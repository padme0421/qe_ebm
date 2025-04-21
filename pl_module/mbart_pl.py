from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import scipy
import numpy as np
from itertools import chain
from copy import deepcopy
from datasets import Dataset
import time
import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import rcParams
import os
from torch.autograd import grad

import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from transformers.generation.utils import GenerateOutput

import wandb

from transformers import (
    PreTrainedTokenizer, MBartForConditionalGeneration, MBartConfig, 
    XLMRobertaModel, MBart50TokenizerFast, AutoTokenizer)

import adapters
from adapters import (setup_adapter_training, AdapterArguments)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation import SampleEncoderDecoderOutput, BeamSearchEncoderDecoderOutput
import huggingface_hub

from comet import download_model, load_from_checkpoint
from comet.models.multitask.unified_metric import UnifiedMetric
from comet.models.utils import Prediction
from comet.encoders.xlmr import XLMREncoder
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from score.score import Scorer
from info_nce import InfoNCE, info_nce
from pl_module.utils import (check_trainable_params, register_param_hooks, 
                             get_length, STL, pad_tensor, load_pretranslations,
                             precompute_corpus_embeddings, precompute_similarity, retrieve_unlabeled_batch_precomputed)

from custom_dataset import HuggingfaceDataModule

class MBARTPL(pl.LightningModule):

    def __init__(
        self,
        active_config,
        config,
        device,
        tokenizer: PreTrainedTokenizer
    ):
        super().__init__()
        self.save_hyperparameters()
        self.active_config = active_config
        self.config = config
        self.run_id = config['dir_name'].split('/')[-1]

        if self.config['timing_run']:
            with open(f"timing/{self.run_id}/settings.json", 'w') as f:
                json.dump(config, f)

        self.model = MBartForConditionalGeneration.from_pretrained(self.active_config["model_name_or_path"])
        adapters.init(self.model)
        if config['adapter']:
            adapter_args = AdapterArguments(train_adapter=True, adapter_config="seq_bn_inv")
            setup_adapter_training(self.model, adapter_args, f"{active_config['src']}-{active_config['trg']}_adapter")

        if self.config['pretranslation_path'] != "":
            self.pretranslations = load_pretranslations(self.config['pretranslation_path'])
        else:
            self.pretranslations = None

        self.tokenizer = tokenizer
        self.scorer = Scorer(self.active_config, self.config, self.tokenizer, None, None, device=torch.device("cuda"))
    
    def forward(self, **inputs):
        """
        **inputs: output of tokenizer, dict("input_ids", "attention_mask", "labels")
        """
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        pass
    
    def rank(self, hypothesis_num, src, labels, cross_attention, encoder_attention, decoder_attention):
        # TODO: fix to incorporate
        batch_num = int(len(src)/hypothesis_num)

        batch_scaled_scores = {}
        batch_raw_scores = {}

        if self.config['score'] == "base":
            batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.base_score(cross_attention) # for ranking, no need to scale scores

        elif self.config['score'] == "uniform":
            batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.uniform_score(src)
            
        elif self.config['score'] == "fast_align":
            batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.fast_align_alignment_score(src, labels, cross_attention)
        
        elif self.config['score'] == "awesome_align":
                batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.awesome_align_alignment_score(src, labels, cross_attention)

        elif self.config['score'] == "dep_parse_awesome_align":
            batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.dependency_parse_score_awesome_align(src, labels, cross_attention)
                                                                                                        
        elif self.config['score'] == "dep_parse_base_align":
            batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.dependency_parse_score_base_align(src, labels, cross_attention, 
                                                                                                                                                      encoder_attention, decoder_attention)
                                                                                                                                                      
        elif self.config['score'] == "comet_kiwi":
            batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.comet_kiwi_score(src, labels, 
                                                                                         cross_attention, encoder_attention, decoder_attention)
        
        elif self.config['score'] == "ensemble":
            if 'base' in self.config['score_list']:
                batch_scaled_scores['base'], batch_raw_scores['base'] = self.scorer.base_score(
                                                                        cross_attention)
                                                                        
            if 'awesome_align' in self.config['score_list']:
                batch_scaled_scores['awesome_align'], batch_raw_scores['awesome_align'] = self.scorer.awesome_align_alignment_score(
                                                            src, labels, cross_attention)
                                                            
            if 'comet_kiwi' in self.config['score_list']:
                batch_scaled_scores['comet_kiwi'], batch_raw_scores['comet_kiwi'] = self.scorer.comet_kiwi_score(src, labels, 
                                                                                         cross_attention, encoder_attention, decoder_attention)
            
        for score_key, score_values in batch_raw_scores.items():
            batch_raw_scores[score_key] = score_values.float()

        batch_raw_scores['final'] = torch.mean(torch.stack(tuple(batch_raw_scores.values())), dim=0)        
        indices = batch_raw_scores['final'].view(-1, hypothesis_num).max(dim=-1)[1]
        labels = labels.view(batch_num, hypothesis_num, -1) # [batch_size, hyp_num, seq len]
        
        cross_attention = cross_attention.view(batch_num, hypothesis_num, cross_attention.size(1), cross_attention.size(2), cross_attention.size(3))
        encoder_attention = encoder_attention.view(batch_num, hypothesis_num, encoder_attention.size(1), encoder_attention.size(2), encoder_attention.size(3))
        decoder_attention = decoder_attention.view(batch_num, hypothesis_num, decoder_attention.size(1), decoder_attention.size(2), decoder_attention.size(3))
        
        return labels[range(0, batch_num), indices], cross_attention[range(0, batch_num), indices], encoder_attention[range(0, batch_num), indices], decoder_attention[range(0, batch_num), indices]
    
    def eval_generate(self, batch, num_hypotheses):
        # to be used in eval
        # batch: {input_ids, attention_mask}
        batch.pop('id')
        return self.model.generate(**batch,
                            num_beams=5,
                            num_return_sequences=num_hypotheses,
                            max_new_tokens = self.config['max_length'],
                            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']],
                            early_stopping=True,
                            use_cache=False, synced_gpus=True)
    
    def train_generate(self, batch: Dict, num_hypotheses: int = 1, override_selfsup_strategy = None,
                       return_dict_in_generate: bool = False, 
                       output_attentions: bool = False) -> (GenerateOutput | LongTensor):

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

        return self.model.generate(**batch, 
                            do_sample=do_sample, num_beams=num_beams,
                            top_p=top_p, top_k=top_k,
                            num_return_sequences=num_hypotheses,
                            max_new_tokens = self.config['max_length'],
                            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']],
                            use_cache=False, synced_gpus=True, 
                            early_stopping=True,
                            return_dict_in_generate=return_dict_in_generate, 
                            output_attentions=output_attentions)

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch.pop("labels")
        batch.pop("decoder_input_ids")

        if not self.config['eval_teacher_forcing']:
            # generated translation
            if self.config['ranking']:
                num_hypotheses = 5
            else:
                num_hypotheses = 1

            pred_ids = self.eval_generate(batch, num_hypotheses)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat_interleave(repeats=num_hypotheses, dim=0)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat_interleave(repeats=num_hypotheses, dim=0)


            # forward pass, teacher forcing with generated translation
            beam_outputs: Seq2SeqLMOutput = self.model(**repeated_batch, labels=pred_labels, output_attentions=True)
            # prepare attention (last layer of decoder)
            cross_attention = beam_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # (batch_size, num_heads, target seq length, input seq length)
            encoder_attention = beam_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = beam_outputs.decoder_attentions[-1][:, :, 1:, 1:]

            if self.config['ranking']:
                pred_labels, cross_attention, encoder_attention, decoder_attention = self.rank(num_hypotheses, repeated_batch['input_ids'], pred_labels, cross_attention, encoder_attention, decoder_attention)

        # calculate loss, teacher forcing with real output
        teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
        if self.config["eval_teacher_forcing"]:
            pred_labels = torch.argmax(teacher_forced_outputs.logits, dim=-1)
            cross_attention = teacher_forced_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # leave out attention for language token ids
            encoder_attention = teacher_forced_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = teacher_forced_outputs.decoder_attentions[-1][:, :, 1:, 1:]
        val_loss = teacher_forced_outputs.loss
        self.log('val_loss', val_loss.item(), sync_dist=True) # loss is unsupervised loss (targets are beam search output, not gold target)

        # pred_labels: [trg lang code] X [eos] [pad] 
        # labels: [trg lang code] X [eos] [pad] 
        return {"loss": val_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention, "encoder_attention": encoder_attention, "decoder_attention": decoder_attention}
    

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch.pop("labels")
        batch.pop("decoder_input_ids")

        if not self.config['eval_teacher_forcing']:

            # generated translation
            if self.config['ranking']:
                num_hypotheses = 5
            else:
                num_hypotheses = 1

            pred_ids = self.eval_generate(batch, num_hypotheses)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat_interleave(repeats=num_hypotheses, dim=0)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat_interleave(repeats=num_hypotheses, dim=0)

            # forward pass, teacher forcing with generated translation
            beam_outputs: Seq2SeqLMOutput = self.model(**repeated_batch, labels=pred_labels, output_attentions=True)
            # prepare attention (last layer of decoder)
            cross_attention = beam_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # (batch_size, num_heads, target seq length, input seq length)
            encoder_attention = beam_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = beam_outputs.decoder_attentions[-1][:, :, 1:, 1:]

            if self.config['ranking']:
                pred_labels, cross_attention, encoder_attention, decoder_attention = self.rank(num_hypotheses, repeated_batch['input_ids'], pred_labels, cross_attention, encoder_attention, decoder_attention)

        # calculate loss, teacher forcing with real output
        teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
        if self.config["eval_teacher_forcing"]:
            pred_labels = torch.argmax(teacher_forced_outputs.logits, dim=-1)
            cross_attention = teacher_forced_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # leave out attention for language token ids
            encoder_attention = teacher_forced_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = teacher_forced_outputs.decoder_attentions[-1][:, :, 1:, 1:]
        test_loss = teacher_forced_outputs.loss
        self.log('test_loss', test_loss.item(), sync_dist=True) # loss is unsupervised loss (targets are beam search output, not gold target)

        # pred_labels: [trg lang code] X [eos] [pad] 
        # labels: [trg lang code] X [eos] [pad] 
        return {"loss": test_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention, "encoder_attention": encoder_attention, "decoder_attention": decoder_attention}
    
    
    
        

        
    def configure_optimizers(self):
        optimizer = Adam(self.trainer.model.parameters(),
                         eps=1e-06,
                         betas=(0.9, 0.98),
                         weight_decay=1e-05,
                         lr=self.config['learning_rate'])
        #scheduler = InverseSqrtScheduler(optimizer, 500)

        #return {
        #    "optimizer": optimizer,
        #    "lr_scheduler": {
		#        "scheduler": scheduler,
		#        "interval": "step",
	    #    }
        #}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": #ReduceLROnPlateau(optimizer=trans_optimizer,
                              #                 mode='max',
                              #                 patience=3),
                              CosineAnnealingWarmRestarts(optimizer, 2),
                "monitor": "val_comet_kiwi",
                }
        }

    '''
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # code borrowed from https://github.com/Lightning-AI/lightning/issues/16117
        """Fix the checkpoint loading issue for deepspeed."""
        if "state_dict" in checkpoint:
            return
        state_dict = checkpoint['module']
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        checkpoint['state_dict'] = state_dict
        return
    '''

class MBARTSupPL(MBARTPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer, warmup: bool = False):
        super().__init__(active_config, config, device, tokenizer)
        self.warmup = warmup

    def training_step(self, batch, batch_idx):
        start_time = time.time()

        outputs = self.model.forward(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])
        loss = outputs[0]
        self.log("train_loss", loss, sync_dist=True)

        return {"loss": loss, "start_time": start_time}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        end_time = time.time()
        duration = end_time-outputs["start_time"]
        if self.warmup:
            if self.trainer.global_step == self.config['warmup_steps']:
                self.trainer.should_stop = True

        if self.config['timing_run']:
            with open(f"timing/{self.run_id}/{self.config['function']}_step.txt", 'w') as f:
                f.write(str(duration))
            if batch_idx == 0:
                exit(0)


class MBARTSslPL(MBARTPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer,
                 datamodule: HuggingfaceDataModule):
        super().__init__(active_config, config, device, tokenizer)
        self.epoch_raw_scores = {}
        self.epoch_scaled_scores = {}
        self.tokenizer = tokenizer
        self.datamodule = datamodule

        self.loss_weights = nn.Parameter(torch.tensor([1.0, self.config['unsup_wt']]).to(self.model.device))
        self.loss_weights.requires_grad_(False)
        if self.config['weight_schedule_strategy'] == 'grad_norm':
            self.loss_weights.requires_grad_(True)
        self.weight_sum = self.loss_weights.sum().detach() # sum of weights

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

        def unsup_criterion(src, outputs, labels, cross_attention, encoder_attention, decoder_attention, 
                            scaled_score_record, raw_score_record, verbose, only_best_sample, num_hypotheses):
            '''
            src: [batch size, src len] ([src_lang_code] X [eos] [pad] * N )
            outputs: [batch size, trg len, output dim] 
            labels: [batch size, trg len] ([trg_lang_code] Y [eos] [pad] * N)
            attention: [batch size, n heads, trg len, src len]
            score record is a dict when score == ensemble, list otherwise
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
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.base_score(cross_attention, 
                                                                                                                               scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "uniform":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.uniform_score(src)
                elif self.config['score'] == "fast_align":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.fast_align_alignment_score(src, labels, cross_attention, 
                                                                                                                                               scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "awesome_align":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.awesome_align_alignment_score(src, labels, cross_attention, 
                                                                                                                                                  scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "dep_parse_awesome_align":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.dependency_parse_score_awesome_align(src, labels, cross_attention,
                                                                                                                                                         scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "dep_parse_base_align":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.dependency_parse_score_base_align(src, labels, cross_attention, 
                                                                                                                                                      encoder_attention, decoder_attention, 
                                                                                                                                                      scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "comet_kiwi":
                    batch_scaled_scores[self.config['score']], batch_raw_scores[self.config['score']] = self.scorer.comet_kiwi_score(src, labels, 
                                                                                         cross_attention, encoder_attention, decoder_attention, 
                                                                                         scaled_score_record.get(self.config['score'], []), raw_score_record.get(self.config['score'], []))
                elif self.config['score'] == "ensemble":
                    if 'base' in self.config['score_list']:
                        batch_scaled_scores['base'], batch_raw_scores['base'] = self.scorer.base_score(
                                                                        cross_attention, scaled_score_record.get('base', []), raw_score_record.get('base', [])
                                                                        )
                    if 'awesome_align' in self.config['score_list']:
                        batch_scaled_scores['awesome_align'], batch_raw_scores['awesome_align'] = self.scorer.awesome_align_alignment_score(
                                                            src, labels, cross_attention, scaled_score_record.get('awesome_align', []), raw_score_record.get('awesome_align', [])
                                                            )
                    if 'comet_kiwi' in self.config['score_list']:
                        batch_scaled_scores['comet_kiwi'], batch_raw_scores['comet_kiwi'] = self.scorer.comet_kiwi_score(src, labels, 
                                                                                         cross_attention, encoder_attention, decoder_attention, 
                                                                                         scaled_score_record.get('comet_kiwi', []), raw_score_record.get('comet_kiwi', []))
                for score_key, score_values in batch_scaled_scores.items():
                    batch_scaled_scores[score_key] = score_values.float()
                for score_key, score_values in batch_raw_scores.items():
                    batch_raw_scores[score_key] = score_values.float()

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
            
            # keep only the best sample among generations for same src
            if only_best_sample:
                batch_raw_scores['final'] = batch_raw_scores['final'].view(-1, num_hypotheses)
                batch_scaled_scores['final'] = batch_scaled_scores['final'].view(-1, num_hypotheses)
                batch_sup_loss = batch_sup_loss.view(-1, num_hypotheses)

                best_sample = (batch_raw_scores['final'] == torch.max(batch_raw_scores['final'], dim=1).values.unsqueeze(1))
                batch_scaled_scores['final'] = batch_scaled_scores['final'][best_sample]
                batch_raw_scores['final'] = batch_raw_scores['final'][best_sample]
                batch_sup_loss = batch_sup_loss[best_sample]

                batch_raw_scores['final'] = batch_raw_scores['final'].view(-1)
                batch_scaled_scores['final'] = batch_scaled_scores['final'].view(-1)
                batch_sup_loss = batch_sup_loss.view(-1)
                
            batch_scaled_scores['final'] = batch_scaled_scores['final'].to(batch_sup_loss)
            batch_raw_scores['final'] = batch_raw_scores['final'].to(batch_sup_loss)

            assert batch_scaled_scores['final'].requires_grad == False
            assert batch_raw_scores['final'].requires_grad == False
            assert batch_sup_loss.requires_grad == True
            
            print("batch unsup loss (cross entropy): ", batch_sup_loss)

            loss = batch_scaled_scores['final'].dot(batch_sup_loss)/batch_size

            print("batch unsup loss (cross entropy * scores): ", loss)

            assert loss.requires_grad == True
            
            return loss, batch_scaled_scores, batch_raw_scores, outputs

        self.unsup_criterion = unsup_criterion

    def training_step(self, batch, batch_idx):
        start_time = time.time()
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


        sup_inputs = batch['label'] # TODO maybe change 'label' to 'sup' or 'parallel'?
        unsup_inputs = batch['unlabel']

        # forward pass through model - sup, unsup separately

        # sup inputs
        sup_inputs.pop('id')
        sup_outputs: Seq2SeqLMOutput = self.model(**sup_inputs)
        loss_sup, sup_logits = sup_outputs.loss, sup_outputs.logits
        
        # generate, only get the output ids
        unsup_output_ids: torch.LongTensor = self.train_generate(unsup_inputs,
                                                                 num_hypotheses=self.config['num_hypotheses_nmt'],
                                                                 return_dict_in_generate=False,
                                                                 output_attentions=False) 

        unsup_labels = unsup_output_ids[:, 1:].clone()
        # leave out <eos> at the beginning -> [trg_lang_code] Y [eos] [pad] * N

        if batch_idx % 64 == 0:
            #print("sup_inputs: ", vars(sup_inputs))
            sup_src = self.tokenizer.batch_decode(sup_inputs.input_ids)
            print("sup input src: ", sup_src)

            original_labels = torch.where(sup_inputs.labels == -100, self.model.config.pad_token_id, sup_inputs.labels)
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
        unsup_outputs: Seq2SeqLMOutput = self.model(**repeated_unsup_inputs, labels=unsup_labels, output_attentions=True)
        unsup_logits = unsup_outputs.logits
        
        # prepare attention (last layer of decoder)
        unsup_cross_attention = unsup_outputs.cross_attentions[self.config['cross_attention_layer']] # (batch_size, num_heads, target seq length, input seq length)
        unsup_encoder_attention = unsup_outputs.encoder_attentions[-1]
        unsup_decoder_attention = unsup_outputs.decoder_attentions[-1]

        loss_unsup, scaled_scores, raw_scores, unsup_logits = self.unsup_criterion(
            repeated_unsup_inputs['input_ids'], unsup_logits, unsup_labels, 
            unsup_cross_attention, unsup_encoder_attention, unsup_decoder_attention,
            self.epoch_scaled_scores, self.epoch_raw_scores,
            batch_idx % 64 == 0, self.config['best_sample'], self.config['num_hypotheses_nmt'])

        for score_key in scaled_scores.keys():
            if not (self.config['best_sample'] and score_key == 'final'):
                scaled_scores[score_key] = torch.mean(scaled_scores[score_key].view(-1, self.config['num_hypotheses_nmt']), dim=1)
                raw_scores[score_key] = torch.mean(raw_scores[score_key].view(-1, self.config['num_hypotheses_nmt']), dim=1)
        
        for score_key in scaled_scores.keys(): # including 'final'
            if score_key not in self.epoch_scaled_scores.keys():
                self.epoch_scaled_scores[score_key] = []
                self.epoch_raw_scores[score_key] = []
            self.epoch_scaled_scores[score_key].extend(scaled_scores[score_key].tolist())
            self.epoch_raw_scores[score_key].extend(raw_scores[score_key].tolist())

        self.log("scores mean", scaled_scores['final'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) # batch mean score

        # calculate total loss with current loss weights
        loss = self.loss_weights.dot(torch.stack([loss_sup, loss_unsup]).to(self.loss_weights.dtype))

        self.log("train_loss_sup", loss_sup.item(), sync_dist=True)
        self.log("train_loss_unsup", loss_unsup.item(), sync_dist=True)
        self.log("train_loss", loss.item(), sync_dist=True)

        return {"loss": loss, "start_time": start_time}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        end_time = time.time()
        duration = end_time - outputs["start_time"]
        if self.config['timing_run']:
            with open(f"timing/{self.run_id}/{self.config['function']}_step.txt", 'w') as f:
                f.write(str(duration))
            if batch_idx == 0:
                exit(0)

    @torch.enable_grad()
    def grad_analysis(self, batch):
        batch_size = len(batch['id'])
        ids: Tensor = batch['id']
        ids = ids.tolist()

        if 'decoder_input_ids' in batch:
            batch.pop('decoder_input_ids')
        if 'labels' in batch:
            labels = batch.pop('labels')

        unsup_inputs = batch
        num_hyp = 1 # NOT self.config['num_hypotheses_nmt']

        # forward pass through model - sup, unsup separately
        # generate, only get the output ids
        unsup_output_ids: torch.LongTensor = self.train_generate(unsup_inputs,
                                                                 num_hypotheses=num_hyp, #self.config['num_hypotheses_nmt'],
                                                                 return_dict_in_generate=False,
                                                                 output_attentions=False) 

        unsup_labels = unsup_output_ids[:, 1:].clone()
        # leave out <eos> at the beginning -> [trg_lang_code] Y [eos] [pad] * N

        #repeated_unsup_inputs = {}
        #repeated_unsup_inputs['input_ids'] = unsup_inputs['input_ids'].repeat_interleave(repeats=self.config['num_hypotheses_nmt'], dim=0)
        #repeated_unsup_inputs['attention_mask'] = unsup_inputs['attention_mask'].repeat_interleave(repeats=self.config['num_hypotheses_nmt'], dim=0)

        repeated_unsup_inputs = unsup_inputs

        # forward pass, teacher forcing with output ids as the labels 
        self.model.requires_grad_(True)
        unsup_outputs: Seq2SeqLMOutput = self.model(**repeated_unsup_inputs, labels=unsup_labels, output_attentions=True)
        unsup_logits = unsup_outputs.logits
        print("before unsupervised criterion: unsup_logits grad? ", unsup_logits.requires_grad)
        
        # prepare attention (last layer of decoder)
        unsup_cross_attention = unsup_outputs.cross_attentions[self.config['cross_attention_layer']] # (batch_size, num_heads, target seq length, input seq length)
        unsup_encoder_attention = unsup_outputs.encoder_attentions[-1]
        unsup_decoder_attention = unsup_outputs.decoder_attentions[-1]

        loss_unsup, scaled_scores, raw_scores, unsup_logits = self.unsup_criterion(
            repeated_unsup_inputs['input_ids'], unsup_logits, unsup_labels, 
            unsup_cross_attention, unsup_encoder_attention, unsup_decoder_attention,
            self.epoch_scaled_scores, self.epoch_raw_scores,
            True, self.config['best_sample'], num_hyp
            )
        
        # unsup_logits
        # [batch size * trg len, output_dim]

        for score_key in scaled_scores.keys():
            if not (self.config['best_sample'] and score_key == 'final'):
                scaled_scores[score_key] = torch.mean(scaled_scores[score_key].view(-1, num_hyp), dim=1)
                raw_scores[score_key] = torch.mean(raw_scores[score_key].view(-1, num_hyp), dim=1)
        
        for score_key in scaled_scores.keys(): # including 'final'
            if score_key not in self.epoch_scaled_scores.keys():
                self.epoch_scaled_scores[score_key] = []
                self.epoch_raw_scores[score_key] = []
            self.epoch_scaled_scores[score_key].extend(scaled_scores[score_key].tolist())
            self.epoch_raw_scores[score_key].extend(raw_scores[score_key].tolist())

        # unsup_labels: [batch_size * num_hypotheses_nmt, seq_length]
        # grad of loss wrt param
        rcParams['font.family'] = 'FreeSerif'

        for index in range(batch_size):
            seq_length = unsup_labels.size(1) - 1 # leave out lang id
            # unsup_logits: [batch_size * seq_len, vocab_size]
            grads: Tensor = grad(loss_unsup, unsup_logits, retain_graph=True, allow_unused=True)[0]
            # grads: [batch_size * seq_len, vocab_size] -> [batch_size, seq_len, vocab_size]
            grads = grads.view(-1, seq_length, grads.size(-1))
            grads = grads[index, torch.arange(seq_length), unsup_labels[index][1:]]
            grads = grads.cpu()

            unsup_labels_tokens = self.tokenizer.convert_ids_to_tokens(unsup_labels[index][1:])
            unsup_labels_str = self.tokenizer.decode(unsup_labels[index][1:], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            print("grads.size(0): ", grads.size(0))
            print("len(unsup_labels_tokens): ", len(unsup_labels_tokens))
            assert grads.size(0) == len(unsup_labels_tokens)
            normalized_grads = grads.abs()/(grads.abs().sum() + 1e-6)
            normalized_grads = (normalized_grads - normalized_grads.min()) / (normalized_grads.max() - normalized_grads.min() + 1e-6)

            valid_indices = [i for i, token in enumerate(unsup_labels_tokens) if token != "<pad>"]
            filtered_tokens = [unsup_labels_tokens[i] for i in valid_indices]
            filtered_grads = normalized_grads[valid_indices]
            filtered_grads: np.ndarray = filtered_grads.numpy()

            colormap = plt.cm.viridis  # Choose your preferred colormap
            colors = colormap(filtered_grads)

            fig, ax = plt.subplots(figsize=(10,2))
            unsup_input_str = self.tokenizer.decode(batch['input_ids'][index],
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
            label_str = self.tokenizer.decode(labels[index],
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
            fig.suptitle(f"Source: {unsup_input_str}\nGold Translation: {label_str}")
            ax: Axes = ax

            ax.set_xticks(range(len(filtered_tokens)))
            ax.set_xticklabels(filtered_tokens, rotation=90)
            ax.set_yticks([])
            im = ax.imshow(colors[np.newaxis, :, :], aspect='auto')
            fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.2)
            
            fig.savefig(f"grad_analysis/{self.run_id}/reinforce_grad_analysis_id{ids[index]}.png")
            with open(f"grad_analysis/{self.run_id}/reinforce_tokens_id{ids[index]}.txt", 'w') as f:
                f.write(json.dumps(unsup_labels_tokens, ensure_ascii=False))
            with open(f"grad_analysis/{self.run_id}/reinforce_strings_id{ids[index]}.json", 'w') as f:
                text_dict = {"input": unsup_input_str, "labels": label_str, "output": unsup_labels_str}
                f.write(json.dumps(text_dict, ensure_ascii=False))
            filtered_grads.dump(f"grad_analysis/{self.run_id}/reinforce_grads_id{ids[index]}.pkl")
        
        self.model.zero_grad()

    def on_train_end(self):
        if self.config['grad_analysis_run']:
            with open(f"grad_analysis/{self.run_id}/settings.txt", 'w') as f:
                f.write("test")
            for batch in self.datamodule.test_dataloader():
                # move batch to cuda
                print(batch)
                for key in batch.keys():
                    batch[key] = batch[key].to(self.model.device)
                self.grad_analysis(batch)

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch.pop("labels")
        batch.pop("decoder_input_ids")
            
        if not self.config['eval_teacher_forcing']:

            # generated translation
            if self.config['ranking']:
                num_hypotheses = 5
            else:
                num_hypotheses = 1

            pred_ids = self.eval_generate(batch, num_hypotheses)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat_interleave(repeats=num_hypotheses, dim=0)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat_interleave(repeats=num_hypotheses, dim=0)

            # forward pass, teacher forcing with generated translation
            beam_outputs: Seq2SeqLMOutput = self.model(**repeated_batch, labels=pred_labels, output_attentions=True)
            # prepare attention (last layer of decoder)
            cross_attention = beam_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # (batch_size, num_heads, target seq length, input seq length)
            encoder_attention = beam_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = beam_outputs.decoder_attentions[-1][:, :, 1:, 1:]

            if self.config['ranking']:
                pred_labels, cross_attention, encoder_attention, decoder_attention = self.rank(num_hypotheses, repeated_batch['input_ids'], pred_labels, cross_attention, encoder_attention, decoder_attention)

        # calculate loss, teacher forcing with real output
        teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
        if self.config["eval_teacher_forcing"]:
            pred_labels = torch.argmax(teacher_forced_outputs.logits, dim=-1)
            cross_attention = teacher_forced_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # leave out attention for language token ids
            encoder_attention = teacher_forced_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = teacher_forced_outputs.decoder_attentions[-1][:, :, 1:, 1:]
        test_loss = teacher_forced_outputs.loss
        self.log('test_loss', test_loss.item(), sync_dist=True) # loss is unsupervised loss (targets are beam search output, not gold target)

        # pred_labels: [trg lang code] X [eos] [pad] 
        # labels: [trg lang code] X [eos] [pad] 
        return {"loss": test_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention, "encoder_attention": encoder_attention, "decoder_attention": decoder_attention}
    
