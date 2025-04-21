from typing import Any, Dict, Optional, List
from pytorch_lightning.utilities.types import STEP_OUTPUT
import scipy
import numpy as np
from itertools import chain
from copy import deepcopy
import time
import json
import os

import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

import wandb

from transformers import (
    PreTrainedTokenizer, MBartForConditionalGeneration
    )
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation.utils import GenerateOutput
from pl_module.mbart_pl import MBARTPL
from trl.trl import PPOTrainer, PPOConfig, create_reference_model
import adapters
from adapters import setup_adapter_training, AdapterArguments
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from score.score import Scorer

from trl.trl.core import (
    WANDB_PADDING,
    PPODecorators,
    convert_to_scalar,
    logprobs_from_logits,
    stack_dicts,
    stats_to_np,
)

from trl.trl.models import AutoModelForSeq2SeqLMWithValueHead
from custom_dataset import HuggingfaceDataModule
from pl_module.utils import (precompute_corpus_embeddings, precompute_similarity, retrieve_unlabeled_batch_precomputed)

class MBART_TRL_PpoPL(MBARTPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer, datamodule: HuggingfaceDataModule):
        super().__init__(active_config, config, device, tokenizer)
       
        self.tokenizer = tokenizer
        self.datamodule = datamodule
        self.run_id = config['dir_name'].split('/')[-1]

        if self.config['timing_run']:
            with open(f"timing/{self.run_id}/settings.json", 'w') as f:
                json.dump(config, f)
    
        self.ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(self.active_config["model_name_or_path"])
        self.model = self.ppo_model.pretrained_model

        adapters.init(self.ppo_model.pretrained_model)
        if config['adapter']:
            adapter_args = AdapterArguments(train_adapter=True, adapter_config="seq_bn_inv")
            setup_adapter_training(self.ppo_model.pretrained_model, adapter_args, f"{active_config['src']}-{active_config['trg']}_adapter")
        
        self.loss_weights = torch.tensor([1.0, self.config['unsup_wt']])
        self.loss_weights.requires_grad_(False)
        if self.config['weight_schedule_strategy'] == 'grad_norm':
            self.loss_weights.requires_grad_(True)
        self.weight_sum = self.loss_weights.sum().detach() # sum of weights

        self.epoch_raw_scores = {}
        self.epoch_scaled_scores = {}

        self.ppo_config = PPOConfig(batch_size=self.config['unsup_batch_size'] * self.config['num_hypotheses_nmt'], 
                               mini_batch_size=self.config['unsup_batch_size'],
                               gradient_accumulation_steps=self.config['num_hypotheses_nmt'],
                                ppo_epochs=2)
        
        # during initialization ref model is the model itself

        self.ref_ppo_model = create_reference_model(self.ppo_model)
        self.ppo_trainer = CustomPPOTrainer(self.ppo_config, self.ppo_model, self.ref_ppo_model, tokenizer)

        self.automatic_optimization = False

        def unsup_criterion(src, outputs, labels, cross_attention, encoder_attention, decoder_attention, scaled_score_record, raw_score_record, verbose):
            '''
            <parameters>
            src: [batch size, src len] ([src_lang_code] X [eos] [pad] * N )
            outputs: [batch size, trg len, output dim] 
            labels: [batch size, trg len] ([trg_lang_code] Y [eos] [pad] * N)
            attention: [batch size, n heads, trg len, src len]

            <Actor-Critic>
            loss: not used
            '''
            
            #print("Check grad exists")
            #print(src.requires_grad)
            #print(outputs.requires_grad)
            #print(labels.requires_grad)
            #print(attention.requires_grad)

            batch_size = src.shape[0]

            # delete lang code from src, outputs, labels, attention
            src = src[:, 1:]
            outputs = outputs[:, 1:, :]
            labels = labels[:, 1:]
            cross_attention = cross_attention[:, :, 1:, 1:]
            encoder_attention = encoder_attention[:, :, 1:, 1:]
            decoder_attention = decoder_attention[:, :, 1:, 1:]

            # set loss function, ignore pad
            sup_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index = self.model.config.pad_token_id)

            # reshape outputs, labels and calculate loss
            output_dim = outputs.shape[-1]
            outputs = outputs.contiguous().view(-1, output_dim) # [batch size * trg len, output_dim]
            labels = labels.contiguous().view(-1) # [batch size * trg len]
            batch_sup_loss = sup_criterion(outputs, labels).view(batch_size, -1).sum(dim=1) # batch size

            # reshape labels back
            labels = labels.view(batch_size,-1) 
            
            assert labels.shape[1] == cross_attention.shape[2] # trg len
            assert src.shape[1] == cross_attention.shape[-1] # src len

            with torch.no_grad():
                # get batch scores
                
                batch_scaled_scores = {}
                batch_raw_scores = {}
                
                if self.config['score'] == "base":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.base_score(
                                                                                                        cross_attention, 
                                                                                                        scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "uniform":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.uniform_score(src)
                elif self.config['score'] == "fast_align":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.fast_align_alignment_score(
                                                                                                        src, labels, cross_attention, 
                                                                                                        scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "awesome_align":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.awesome_align_alignment_score(
                                                                                                        src, labels, cross_attention, 
                                                                                                        scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "dep_parse_awesome_align":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.dependency_parse_score_awesome_align(
                                                                                                        src, labels, cross_attention, 
                                                                                                        scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "dep_parse_base_align":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.dependency_parse_score_base_align(
                                                                                        src, labels, cross_attention, encoder_attention, decoder_attention, 
                                                                                        scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "comet_kiwi":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.comet_kiwi_score(
                                                                                        src, labels, cross_attention, encoder_attention, decoder_attention, 
                                                                                        scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "ensemble":
                    if 'base' in self.config['score_list']:
                        batch_scaled_scores['base'], batch_raw_scores['base'] = self.scorer.base_score(
                                                                        cross_attention, scaled_score_record.get('base', []), raw_score_record.get('base', [])
                                                                        )
                    if 'awesome_align' in self.config['score_list']:
                        batch_scaled_scores['awesome_align'], batch_raw_scores['awesome_align'] = self.scorer.awesome_align_alignment_score(
                                                                                            src, labels, cross_attention, 
                                                                                            scaled_score_record.get('awesome_align', []), raw_score_record.get('awesome_align', [])
                                                                                            )
                    if 'comet_kiwi' in self.config['score_list']:
                        batch_scaled_scores['comet_kiwi'], batch_raw_scores['comet_kiwi'] = self.scorer.comet_kiwi_score(src, labels, 
                                                                                         cross_attention, encoder_attention, decoder_attention, 
                                                                                         scaled_score_record.get('comet_kiwi', []), raw_score_record.get('comet_kiwi', []))

                
            batch_raw_scores['final'] = torch.mean(torch.stack(tuple(batch_raw_scores.values())), dim=0)
            batch_scaled_scores['final'] = torch.mean(torch.stack(tuple(batch_scaled_scores.values())), dim=0)

            # filter samples whose confidence is lower than mean score
            if 'final' in raw_score_record.keys():
                good_samples = (batch_raw_scores['final'] > np.quantile(raw_score_record['final'], 0.5)) # indices of samples to keep
                print("good_samples: ", torch.sum(good_samples).item())
            else:
                good_samples = (batch_raw_scores['final'] > -np.inf)
                print("good_samples: ", torch.sum(good_samples).item())

            # filter
            if self.config['filter']:
                batch_scaled_scores['final'] = batch_scaled_scores['final'][good_samples]
                batch_raw_scores['final'] = batch_raw_scores['final'][good_samples]
                batch_sup_loss = batch_sup_loss[good_samples]
                
            batch_scaled_scores['final'] = batch_scaled_scores['final'].to(batch_sup_loss)
            batch_raw_scores['final'] = batch_raw_scores['final'].to(batch_sup_loss)

            assert batch_scaled_scores['final'].requires_grad == False
            assert batch_raw_scores['final'].requires_grad == False
            assert batch_sup_loss.requires_grad == True
            
            print("batch unsup loss (cross entropy): ", batch_sup_loss)

            loss = batch_scaled_scores['final'].dot(batch_sup_loss)/batch_size

            print("batch unsup loss (cross entropy * scores): ", loss)

            for score_key, score_values in batch_scaled_scores.items():
                batch_scaled_scores[score_key] = score_values.float()
            for score_key, score_values in batch_raw_scores.items():
                batch_raw_scores[score_key] = score_values.float()

            assert loss.requires_grad == True
                
            return loss, batch_scaled_scores, batch_raw_scores

        self.unsup_criterion = unsup_criterion

        # precompute corpus embeddings, similarity
        if self.config['active_unlabeled_retrieval']:
            encoder = SentenceTransformer(self.config['retrieval_model'])

            if 'source_embedding' not in self.datamodule.dataset['unlabel_train'].column_names:
                input_ids = self.datamodule.dataset["unlabel_train"]["input_ids"]
                src_text = self.tokenizer.batch_decode(input_ids)
                result = precompute_corpus_embeddings(encoder, src_text)
                embeddings = result["embeddings"]
                
                if self.config['timing_run']:
                    with open(f"timing/{self.run_id}/precompute_corpus_embeddings_duration.txt", 'w') as f:
                        f.write(str(result["duration"]))

                self.datamodule.dataset["unlabel_train"] = self.datamodule.dataset["unlabel_train"].add_column('source_embedding', embeddings)

            if 'neighbor' not in self.datamodule.dataset['unlabel_train'].column_names:
                result = precompute_similarity(self.datamodule.dataset["unlabel_train"])
                self.datamodule.dataset["unlabel_train"] = result["corpus"]

                if self.config['timing_run']:
                    with open(f"timing/{self.run_id}/precompute_similarity_duration.txt", 'w') as f:
                        f.write(str(result["duration"]))

                self.datamodule.dataset_id2entry = {}
                self.datamodule.dataset_id2entry["unlabel_train"] = {item["id"].item(): item for item in self.datamodule.dataset["unlabel_train"]}
                print(self.datamodule.dataset_id2entry["unlabel_train"].keys())

    @torch.autocast("cuda")
    def training_step(self, batch, batch_idx):
        start_time = time.time()
        optim = self.optimizers()

        # retrieve better unsupervised data
        if self.config['active_unlabeled_retrieval']:
            result = retrieve_unlabeled_batch_precomputed(self.datamodule, self.tokenizer, self.config, batch['label'])
            batch['unlabel'] = result['batch']

            if self.config['timing_run']:
                with open(f"timing/{self.run_id}/retrieve_unlabeled_duration.txt", 'w') as f:
                    f.write(str(result['duration']))

            batch['unlabel']['input_ids'] = batch['unlabel']['input_ids'].to(self.device)
            batch['unlabel']['attention_mask'] = batch['unlabel']['attention_mask'].to(self.device)

        # decide loss weights
        # manual optimization -> trainer.global_step == every batch (not every actual update step)
        if self.config['weight_schedule_strategy'] == 'constant':
            self.loss_weights[0] = 1.0
            self.loss_weights[1] = self.config['unsup_wt']

        elif self.config['weight_schedule_strategy'] == 'increase_cap':
            self.loss_weights[1] = min(1, self.trainer.global_step / (1000 * self.config['accumulate_grad_batches'])) * self.config['unsup_wt']
            if self.config['sup_wt_constant']:
                self.loss_weights[0] = 1.0
            else:
                self.loss_weights[0] = max(0, 1 - 10 * self.loss_weights[1])

        elif self.config['weight_schedule_strategy'] == 'decrease_cap':
            self.loss_weights[1] = max(0, 1 - self.trainer.global_step / (1000 * self.config['accumulate_grad_batches'])) * self.config['unsup_wt']
            if self.config['sup_wt_constant']:
                self.loss_weights[0] = 1.0
            else:
                self.loss_weights[0] = min(1.0, 1 - 10 * self.loss_weights[1])

        self.loss_weights = self.loss_weights.to(self.ppo_model.pretrained_model.device)

        # TODO: mask padding

        sup_inputs = batch['label'] 
        unsup_inputs = batch['unlabel']
        seq_len = unsup_inputs.input_ids.size(1)
        unsup_batch_size = unsup_inputs.input_ids.size(0)

        print("batch_idx: ", batch_idx)
        print("seq_len: ", seq_len)
        print("unsup_batch_size: ", unsup_batch_size)

        # forward pass through model - sup, unsup separately

        # sup inputs
        sup_inputs.pop('id')
        sup_outputs: Seq2SeqLMOutput = self.ppo_model.pretrained_model(**sup_inputs)
        loss_sup, sup_logits = sup_outputs.loss, sup_outputs.logits

        self.manual_backward(loss_sup * self.loss_weights[0])
        
        # clip gradients
        self.clip_gradients(optim, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        if (batch_idx + 1) % self.config['accumulate_grad_batches'] == 0:
            optim.step()
            optim.zero_grad()

        # apply actor critic (PPO) algorithm to unsup_inputs
        
        # 1. generate rollouts (sequences)
            
        # generate, only get the output ids
        unsup_output_ids: torch.LongTensor = self.train_generate(unsup_inputs, num_hypotheses=self.config['num_hypotheses_nmt'],
                                                                 return_dict_in_generate=False, 
                                                                 output_attentions=False)

        # unsup_output_ids max length: 2 + 50
        unsup_labels = unsup_output_ids[:, 1:].clone() # 51
        # leave out <eos> at the beginning -> [trg_lang_code] Y [eos] [pad] * N

        if batch_idx % 64 == 0:
            sup_src = self.tokenizer.batch_decode(sup_inputs.input_ids)
            print("sup input src: ", sup_src)

            original_labels = torch.where(sup_inputs.labels == -100, self.ppo_model.config.pad_token_id, sup_inputs.labels)
            sup_labels = self.tokenizer.batch_decode(original_labels)
            print("sup input labels:  ", sup_labels)

            sup_decoder_ids = self.tokenizer.batch_decode(sup_inputs.decoder_input_ids)
            print("sup input decoder ids: ", sup_decoder_ids)

            sup_prediction = self.tokenizer.batch_decode(torch.argmax(sup_outputs.logits,-1))
            print("sup input prediction: ", sup_prediction)

            unsup_src = self.tokenizer.batch_decode(unsup_inputs.input_ids)
            print("unsup input src: ", unsup_src)

            unsup_prediction_raw = self.tokenizer.batch_decode(unsup_output_ids)
            print("unsup input prediction raw: (before removing first token): \n", unsup_prediction_raw)

            #print("unsup_outputs:", vars(unsup_outputs))
            unsup_prediction = self.tokenizer.batch_decode(unsup_labels)
            print("unsup input prediction: ", unsup_prediction)

        repeated_unsup_inputs = {}
        repeated_unsup_inputs['input_ids'] = unsup_inputs['input_ids'].repeat_interleave(repeats=self.config['num_hypotheses_nmt'], dim=0)
        repeated_unsup_inputs['attention_mask'] = unsup_inputs['attention_mask'].repeat_interleave(repeats=self.config['num_hypotheses_nmt'], dim=0)
      
        # forward pass, teacher forcing with output ids as the labels 
        unsup_outputs: Seq2SeqLMOutput = self.ppo_model.pretrained_model(**repeated_unsup_inputs, labels=unsup_labels, output_attentions=True)
        unsup_logits = unsup_outputs.logits
        
        # prepare attention (last layer of decoder)
        unsup_cross_attention = unsup_outputs.cross_attentions[self.config['cross_attention_layer']] # (batch_size, num_heads, target seq length, input seq length)
        unsup_encoder_attention = unsup_outputs.encoder_attentions[-1]
        unsup_decoder_attention = unsup_outputs.decoder_attentions[-1]

        # get scores
        _, scaled_scores, raw_scores= self.unsup_criterion(
            repeated_unsup_inputs['input_ids'], unsup_logits, unsup_labels, 
            unsup_cross_attention, unsup_encoder_attention, unsup_decoder_attention,
            self.epoch_scaled_scores, self.epoch_raw_scores,
            batch_idx % 64 == 0)
        
        stats = self.ppo_trainer.step(self, list(repeated_unsup_inputs['input_ids']), list(unsup_labels), list(scaled_scores['final']), self.loss_weights[1].item())
        loss_unsup = stats["ppo/loss/total"]

        for score_key in scaled_scores.keys():
            scaled_scores[score_key] = torch.mean(scaled_scores[score_key].view(-1, self.config['num_hypotheses_nmt']), dim=1)
            raw_scores[score_key] = torch.mean(raw_scores[score_key].view(-1, self.config['num_hypotheses_nmt']), dim=1)
        
        for score_key in scaled_scores.keys(): # including 'final'
            if score_key not in self.epoch_scaled_scores.keys():
                self.epoch_scaled_scores[score_key] = []
                self.epoch_raw_scores[score_key] = []
            self.epoch_scaled_scores[score_key].extend(scaled_scores[score_key].tolist())
            self.epoch_raw_scores[score_key].extend(raw_scores[score_key].tolist())

        self.log("scores mean", scaled_scores['final'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) # batch mean score
        
        print("get total loss")

        self.log("sup_weight", self.loss_weights[0].item())
        self.log("unsup_weight", self.loss_weights[1].item())

        # calculate total loss with current loss weights
        loss_unsup = torch.tensor(loss_unsup).to(loss_sup.device)
        loss = self.loss_weights.dot(torch.stack([loss_sup, loss_unsup]).to(self.loss_weights.dtype))

        self.log("train_loss_sup", loss_sup.item(), sync_dist=True)
        self.log("train_loss_unsup", loss_unsup.item(), sync_dist=True)
        self.log("train_loss", loss.item(), sync_dist=True)

        end_time = time.time()
        duration = end_time - start_time
        if self.config['timing_run']:
            with open(f"timing/{self.run_id}/{self.config['function']}_step.txt", 'w') as f:
                f.write(str(duration))
            if batch_idx == 0:
                exit(0)

        return loss
        
    def train_generate(self, batch: Dict, num_hypotheses: int = 1, override_selfsup_strategy = None,
                       return_dict_in_generate: bool = False, 
                       output_attentions: bool = False) -> (GenerateOutput | LongTensor):
        # to be used in train
         # to be used in train
        strategy = override_selfsup_strategy if override_selfsup_strategy else self.config['selfsup_strategy']

        if strategy == "pretranslate":
            text_translations = []
            for id in batch['id'].tolist():
                print("from pretranslations: trying to find id = ", id)
                if id not in self.pretranslations:
                    text_translations.extend(["ERROR"] * num_hypotheses)
                else:   
                    text_translations.extend([self.pretranslations[id]] * num_hypotheses) #temp
            print(text_translations)
            raw_gen_output = None
            with self.tokenizer.as_target_tokenizer():
                raw_gen_output = self.tokenizer(text_translations, return_tensors='pt', padding=True, truncation=True)['input_ids']
            raw_gen_output = torch.cat((torch.ones(raw_gen_output.size(0), 1).long() * self.tokenizer.eos_token_id, raw_gen_output), dim=1) # add <eos>
            raw_gen_output = raw_gen_output.to(batch['input_ids'].device)
            print(raw_gen_output)
            return raw_gen_output

        if self.config['selfsup_strategy'] == "greedy":
            do_sample = False
            num_beams = 1
            top_p = 1.0 # TODO: None? 
            top_k = 50 # TODO: None? 
        elif self.config['selfsup_strategy'] == "sample":
            do_sample = True
            num_beams = 1
            top_p = 0.9
            top_k = 0
        elif self.config['selfsup_strategy'] == "beam":
            do_sample = False
            num_beams = 5
            top_p = 1.0 # TODO: None? 
            top_k = 50 # TODO: None? 
        elif self.config['selfsup_strategy'] == "beam_sample":
            do_sample = True
            num_beams = 5
            top_p = 1.0 # TODO: None? 
            top_k = 50 # TODO: None?

        if 'id' in batch:
            batch.pop('id')


        # recommended kwargs
        ppo_generation_kwargs = {
            "min_length": -1, # don't ignore the EOS token (see above)
            "top_k": 0.0, # no top-k sampling
            "top_p": 1.0, # no nucleus sampling
            "eos_token_id":-1, # https://github.com/huggingface/trl/issues/235
            "do_sample": True, # yes, we want to sample
            "pad_token_id": self.tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
            "max_new_tokens": self.config['max_length'], # specify how many tokens you want to generate at most
        }

        return self.model.generate(**batch, 
                            min_length=ppo_generation_kwargs['min_length'],
                            top_p=ppo_generation_kwargs['top_p'], 
                            top_k=ppo_generation_kwargs['top_k'],
                            eos_token_id=ppo_generation_kwargs['eos_token_id'], 
                            do_sample=ppo_generation_kwargs['do_sample'],
                            pad_token_id=ppo_generation_kwargs['pad_token_id'],
                            max_new_tokens = ppo_generation_kwargs['max_new_tokens'],

                            num_beams=num_beams,
                            num_return_sequences=num_hypotheses,
                            forced_bos_token_id = self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']],
                            use_cache=False, synced_gpus=True, 
                            early_stopping=True,
                            return_dict_in_generate=return_dict_in_generate, 
                            output_attentions=output_attentions)

class CustomPPOTrainer(PPOTrainer):

    @PPODecorators.empty_device_cache()
    def step(
        self,
        pl_module,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        loss_scale: torch.Tensor,
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = len(queries)

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor =self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                num_mini_batch  = int((bs+1)/self.config.mini_batch_size)
                print("num_mini_batch", num_mini_batch)
                mini_batch = 0
                
                
                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    try:
                        print("mini_batch", mini_batch)
                        if mini_batch == num_mini_batch-1:
                            update = True
                            print("ppo update")
                        else:
                            update = False
                        mini_batch_end = mini_batch_start + self.config.mini_batch_size
                        mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                        mini_batch_dict = {
                            "logprobs": batch_dict["logprobs"][mini_batch_inds],
                            "values": batch_dict["values"][mini_batch_inds],
                            "masks": batch_dict["masks"][mini_batch_inds],
                            # hacks: the queries and responses are ragged.
                            "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                            "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                            "advantages": batch_dict["advantages"][mini_batch_inds],
                            "returns": batch_dict["returns"][mini_batch_inds],
                        }
                        for k in model_inputs_names:
                            mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                        with self.accelerator.accumulate(self.model):
                            model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                            logprobs, logits, vpreds, _ = self.batched_forward_pass(
                                self.model,
                                mini_batch_dict["queries"],
                                mini_batch_dict["responses"],
                                model_inputs,
                                return_logits=True,
                            )

                            train_stats = self.train_minibatch(
                                pl_module,
                                mini_batch_dict["logprobs"],
                                mini_batch_dict["values"],
                                logprobs,
                                logits,
                                vpreds,
                                mini_batch_dict["masks"],
                                mini_batch_dict["advantages"],
                                mini_batch_dict["returns"],
                                loss_scale,
                                update,
                                grad_accum_steps=num_mini_batch
                            )
                            all_stats.append(train_stats)
                        mini_batch += 1
                    except Exception:
                        print("exception in PPO")
                        continue

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        pl_module,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
        loss_scale: torch.Tensor,
        update: bool,
        grad_accum_steps: int
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [mini_batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [mini_batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [mini_batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [mini_batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [mini_batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        loss = loss_p + loss_v
        loss = loss * loss_scale / grad_accum_steps
        pl_module.manual_backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        optim = pl_module.optimizers()
        if update:
            optim.step()
            # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
            # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
            optim.zero_grad()
        return train_stats