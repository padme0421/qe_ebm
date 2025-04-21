from typing import Any, Dict

import numpy as np

import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
import wandb
import math
from tqdm import tqdm
import time
import os
import json
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib import rcParams

from transformers import (
    PreTrainedTokenizer, MBartForConditionalGeneration,
    MBart50TokenizerFast, AutoTokenizer)
from datasets import DatasetDict, Dataset
import adapters
from adapters import (setup_adapter_training, AdapterArguments)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation.utils import GenerateOutput
from custom_dataset import HuggingfaceDataModule

from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from score.score import Scorer
from pl_module.utils import (check_trainable_params, register_param_hooks, 
                             get_length, STL, get_adapters, count_optim_params, pad_tensor,
                             load_pretranslations, 
                             precompute_corpus_embeddings, precompute_similarity, retrieve_unlabeled_batch_precomputed
                             )
                             
from energy_model.base_energy_model import EnergyModel
from llm import openai_api

lang_id2str = {"en": "English", "de": "German", "zh": "Chinese", 
                "bn": "Bengali", "ka": "Kazakh", "id": "Indonesian",
                "mr": "Marathi", "az": "Azerbaijani", "mn": "Mongolian"}

class ReparamEmbeddings(nn.Module):
    def __init__(self, original, down_matrix, up_matrix, index_mapping: dict):
        super(ReparamEmbeddings, self).__init__()
        self.index_mapping = index_mapping
        self.original_embeddings = original
        self.down_matrix = down_matrix
        self.up_matrix = up_matrix

    def forward(self, indices: Tensor):
        embeddings = []
        # TODO: check if differentiable
        for index in indices.tolist():
            original_index = self.index_mapping[index]
            embedding = torch.chain_matmul(self.original_embeddings(torch.tensor(original_index).to(self.original_embeddings.weight.device)).unsqueeze(0),
                                           self.down_matrix[index],
                                           self.up_matrix[index])
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
        return embeddings

class MBARTSsl_EBMPL(pl.LightningModule):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer, datamodule: HuggingfaceDataModule, 
                 prepare_energy_model, by_steps: bool = False, warmup: bool = False, 
                ):
        super().__init__()
        self.save_hyperparameters()
        self.active_config = active_config
        self.config = config
        self.tokenizer: MBart50TokenizerFast = tokenizer
        self.datamodule = datamodule
        self.by_steps = by_steps
        self.warmup = warmup
        self.num_hypotheses = 1 # for offline generation
        self.clip_range = 0.5
        self.run_id = config['dir_name'].split('/')[-1]

        if config['timing_run']:
            with open(f"timing/{self.run_id}/settings.json", 'w') as f:
                json.dump(config, f)

        self.llm_translator = openai_api.OPENAI_Translation_API("gpt4", 
                                                                src_lang=lang_id2str[self.active_config['src']],
                                                                trg_lang=lang_id2str[self.active_config['trg']])
        
        if self.config['pretranslation_path'] != "":
            self.pretranslations = load_pretranslations(self.config['pretranslation_path'])
        else:
            self.pretranslations = None

        self.automatic_optimization = False

        # prepare NMT model
        self.model = MBartForConditionalGeneration.from_pretrained(self.active_config['model_name_or_path'])
        self.model.cuda()
        adapters.init(self.model)

        # initialize loss weights 
        self.loss_weights = nn.Parameter(torch.tensor([1.0, self.config['unsup_wt']])
                                         .to(self.model.device))
        self.loss_weights.requires_grad_(False)
        # train loss weights when scheduling with grad norm strategy
        if self.config['weight_schedule_strategy'] == 'grad_norm':
            self.loss_weights.requires_grad_(True)
        self.initial_weight_sum = self.loss_weights.sum().detach()

        # set up NMT adapter
        if config['adapter']:
            adapter_args = AdapterArguments(train_adapter=True, adapter_config="pfeiffer+inv")
            setup_adapter_training(self.model, adapter_args, f"{active_config['src']}-{active_config['trg']}_adapter")

        # prepare energy model
        self.energy_model: EnergyModel = prepare_energy_model(config, active_config)

        # reparameterize energy model embeddings as nmt embeddings
        
        if self.config['reparameterize_embed']:
            self.reparameterize_embeddings(self.energy_model.model.encoder.model.get_input_embeddings(), 
                                           self.model.get_input_embeddings())
        
        self.scorer = Scorer(self.active_config, self.config, self.tokenizer, None, None, device=torch.device("cuda"))
        
        print("nmt model trainable params: ", check_trainable_params(self.model))

        # register backward hooks for NMT model
        if self.config['adapter']:
            register_param_hooks(self.model.get_encoder().get_invertible_adapter(), "nmt model [ADAPTER]")
        else:
            register_param_hooks(self.model.get_encoder().base_model, "nmt model")

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

        # log gradients
        wandb.watch(self.model, log_freq=100)
        wandb.watch(self.energy_model.base_model, log_freq=100)
    
    def reparameterize_embeddings(self, energy_embeddings: nn.Linear, nmt_embeddings: nn.Embedding):
        # find M such that 'energy_embeddings = M * nmt_embeddings'

        nmt_vocab = self.tokenizer.get_vocab()
        energy_vocab = self.energy_model.tokenizer.get_vocab()

        energy_hid_dim = energy_embeddings.weight.size(0)
        nmt_hid_dim = nmt_embeddings.weight.size(1)
        print("energy embeddings size: ", energy_hid_dim, 
              "nmt embeddings size: ", nmt_hid_dim)

        # TODO: better init
        # map: embed low rank -> energy hid dim (for each vocab entry)
        embed_refactor_upmatrix = nn.Parameter(torch.rand(len(energy_vocab), 
                                                    self.config['embed_lowrank'],
                                                    energy_hid_dim, 
                                                    device=energy_embeddings.weight.device))
        
        # map: nmt hid dim -> embed low rank (for each vocab entry)
        embed_refactor_downmatrix = nn.Parameter(torch.rand(len(energy_vocab), 
                                                    nmt_hid_dim,
                                                    self.config['embed_lowrank'], 
                                                    device=energy_embeddings.weight.device))

        index_mapping = {energy_vocab.get(token): nmt_vocab.get(token) for token in energy_vocab}
        reparam_embeddings = ReparamEmbeddings(nmt_embeddings, 
                                               embed_refactor_downmatrix, embed_refactor_upmatrix,
                                               index_mapping)

        # learn up, down matrix for common vocab with MSE loss
        # TODO: what about other vocab??
        embed_optim = torch.optim.Adam(params=[embed_refactor_downmatrix, embed_refactor_upmatrix])
        for token in tqdm(energy_vocab):
            energy_vocab_index = energy_vocab.get(token)
            nmt_vocab_index = nmt_vocab.get(token)
            print(f"vocab entry: {token}, energy vocab index: {energy_vocab_index}, nmt vocab index: {nmt_vocab_index}")
            
            energy_vocab_index_t = F.one_hot(torch.tensor(energy_vocab_index), len(energy_vocab)).to(device=energy_embeddings.weight.device).float()
            nmt_vocab_index_t = torch.tensor(nmt_vocab_index).to(device=nmt_embeddings.weight.device)

            if nmt_vocab_index == -1:
                continue
            for i in range(self.config['embed_match_steps']):
                embed_optim.zero_grad()
                
                predicted_embed = torch.linalg.multi_dot(
                                    [nmt_embeddings(nmt_vocab_index_t).unsqueeze(0),
                                    embed_refactor_downmatrix[energy_vocab_index],
                                    embed_refactor_upmatrix[energy_vocab_index]]
                                    )
                
                # TODO: patch solution -> find better one
                if energy_embeddings.bias.device != energy_embeddings.weight.device:
                    energy_embeddings.bias = nn.Parameter(energy_embeddings.bias.to(energy_embeddings.weight.device))

                correct_embed = energy_embeddings.forward(energy_vocab_index_t).unsqueeze(0)
                embed_loss = F.mse_loss(predicted_embed, correct_embed)
                
                print(f"step: {i}, embed_loss: {embed_loss.item()}, check for decrease")
                embed_loss.backward()
                embed_optim.step()
        
        self.energy_model.encoder.model.set_input_embeddings(reparam_embeddings)


    def eval_generate(self, batch: Dict, num_hypotheses: int) -> LongTensor:
        # to be used in eval
        # batch: {id, input_ids, attention_mask}
        if 'id' in batch:
            batch.pop('id')
        raw_gen_output =  self.model.generate(**batch,
                            num_beams=5,
                            num_return_sequences=num_hypotheses,
                            max_new_tokens = self.config['max_length'],
                            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']],
                            early_stopping=True,
                            use_cache=False, synced_gpus=True)
        
        return raw_gen_output
    
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
            #print(text_translations)
            raw_gen_output = None
            with self.tokenizer.as_target_tokenizer():
                raw_gen_output = self.tokenizer(text_translations, return_tensors='pt', padding=True, truncation=True)['input_ids']
            raw_gen_output = torch.cat((torch.ones(raw_gen_output.size(0), 1).long() * self.tokenizer.eos_token_id, raw_gen_output), dim=1) # add <eos>
            raw_gen_output = raw_gen_output.to(batch['input_ids'].device)
            #print(raw_gen_output)
            return raw_gen_output

        elif strategy == "greedy":
            penalty_alpha = 0
            do_sample = False
            num_beams = 1
            top_p = 1.0 # TODO: None? 
            top_k = 50 # TODO: None? 
        
        elif strategy == "sample":
            penalty_alpha = 0
            do_sample = True
            num_beams = 1
            top_p = self.config['top_p']
            top_k = 0
        
        elif strategy == "k_sample":
            penalty_alpha = 0
            do_sample = True
            num_beams = 1
            top_p = 1.0
            top_k = self.config['top_k']
        
        elif strategy == "contrastive":
            penalty_alpha = self.config['penalty_alpha']
            do_sample = True
            top_p = 1.0
            top_k = self.config['top_k']
            num_beams = 1
        
        elif strategy == "beam":
            penalty_alpha = 0
            do_sample = False
            num_beams = 5
            top_p = 1.0 # TODO: None? 
            top_k = 50 # TODO: None? 
        
        elif strategy == "beam_sample":
            penalty_alpha = 0
            do_sample = True
            num_beams = 5
            top_p = 1.0 # TODO: None? 
            top_k = 50 # TODO: None?
        
        if 'id' in batch:
            batch.pop('id')

        raw_gen_output =  self.model.generate(**batch, 
                            do_sample=do_sample, num_beams=num_beams,
                            top_p=top_p, top_k=top_k, penalty_alpha = penalty_alpha,
                            num_return_sequences=num_hypotheses,
                            max_new_tokens = self.config['max_length'],
                            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']],
                            use_cache=False, synced_gpus=True, 
                            early_stopping=True,
                            return_dict_in_generate=return_dict_in_generate, 
                            output_attentions=output_attentions)
        
        #print(raw_gen_output)
        
        return raw_gen_output
    
    def remove_incompatible_ids(self, token_ids: Tensor, inplace: bool = True):
        """
        remove incompatible ids from mbart tokenized ids before inputting to energy network
        to not modify token ids in place, use `inplace` flag
        """
        # turn lang ids to pad
        incompatible_ids = self.tokenizer.additional_special_tokens_ids + [250053] # <mask>
        incompatible_ids.append(-100)
        incompatible_ids = torch.tensor(incompatible_ids, device=token_ids.device)

        mask_tensor =  torch.isin(token_ids, incompatible_ids)

        if not inplace:
            new_token_ids = token_ids.clone()
            new_token_ids = new_token_ids.masked_fill_(mask_tensor, self.tokenizer.pad_token_id)
            return new_token_ids
        else:
            token_ids = token_ids.masked_fill_(mask_tensor, self.tokenizer.pad_token_id)
            return token_ids

    def set_model_params_grad(self, mode: bool):
        if mode == False:
            # turn all param grads off
            self.model.requires_grad_(False)
        else:
            if self.config['adapter']:
                # turn on only adapter params
                adapters = get_adapters(self.model, f"{self.active_config['src']}-{self.active_config['trg']}_adapter")
                for adapter in adapters:
                    adapter.requires_grad_(True)
            else:
                self.model.requires_grad_(True)
  
    def get_gold_label_effective_energy(self, energy_batch: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        # energy batch input & labels: must not contain incompatible tokens
        # batch input & labels: first token must be energy lang id

        batch_size = len(batch['input_ids'])
        energy_vocab_size = self.energy_model.tokenizer.vocab_size

        # get score of gold labels
        # [batch_size]
        anchor_score  = self.energy_model.forward(
                                    energy_batch['input_ids'][:, 1:], # leave out lang id
                                    F.one_hot(energy_batch['labels'][:, 1:], energy_vocab_size).float(), # leave out lang id
                                    energy_batch['labels'][:, 1:] # leave out lang id
                                    )
        
        assert anchor_score.requires_grad
        
        self.log_dict({'gold_label_score': anchor_score[0].item()})
        
        # get negative sampling probs
        anchor_nmt_logits = self.model.forward(batch['input_ids'], batch['attention_mask'], 
                                               labels=batch['labels']).logits
        # [batch_size]
        anchor_nmt_prob = -F.cross_entropy(anchor_nmt_logits.view(-1, self.tokenizer.vocab_size), 
                                           batch['labels'].view(-1),
                                           reduction='none', ignore_index=-100,
                                           label_smoothing=0.1).view(batch_size, -1).mean(-1)
        
        self.log_dict({'gold_label_sampling_prob': anchor_nmt_prob[0].item()})

        anchor_effective_energy = anchor_score - anchor_nmt_prob

        assert anchor_effective_energy.requires_grad
        return anchor_effective_energy

    def get_positive_sample_effective_energy(self, energy_batch: Dict[str, Tensor], batch: Dict[str, Tensor], 
                                             gen_output_ids: Tensor, energy_gen_output_ids: Tensor) -> Tensor:
        # energy batch input & labels: must not contain incompatible tokens
        # batch input & labels: first token must be energy lang id
        # gen_output_ids: [batch_size * hypothesis_num, seq_len]
        # energy_gen_output_ids: [batch_size * hypothesis_num, seq_len]

        batch_size = len(batch['input_ids'])
        energy_vocab_size = self.energy_model.tokenizer.vocab_size

        # get score for samples
        scores = self.energy_model.forward(
                                    energy_batch['input_ids'][:, 1:].repeat_interleave(repeats=self.config['num_hypotheses_energy'], dim=0), # leave out lang id
                                    F.one_hot(energy_gen_output_ids[:, 1:], energy_vocab_size).float(), # leave out lang id
                                    energy_gen_output_ids[:, 1:] # leave out lang id, [num_hypotheses_energy, seq_length]
                                    ) # [batch_size * hypothesis num]
        scores = scores.view(batch_size, -1) 
        self.log_dict({'positive_sample_score': scores[0].mean().item()})

        gen_output_ids = gen_output_ids.contiguous()
        # get negative sampling probs
        # [batch_size * hypothesis num, seq_len (target), vocab_size]
        batch_logits = self.model.forward(batch['input_ids'].repeat_interleave(repeats=self.config['num_hypotheses_energy'], dim=0),
                                        batch['attention_mask'].repeat_interleave(repeats=self.config['num_hypotheses_energy'], dim=0),
                                        labels=gen_output_ids).logits

        assert not torch.isnan(batch_logits).any()
            
        # [batch_size * hypothesis num]
        sample_nmt_prob = -F.cross_entropy(batch_logits.view(-1, self.tokenizer.vocab_size), gen_output_ids.view(-1),
                                            reduction='none', 
                                            ignore_index=self.tokenizer.pad_token_id, 
                                            label_smoothing=0.1).view(batch_size * self.config['num_hypotheses_energy'], -1).mean(-1)

        sample_nmt_prob = sample_nmt_prob.view(batch_size, -1)
        self.log_dict({'positive_sample_sampling_prob': sample_nmt_prob[0].mean().item()})
    
        sample_effective_energy = scores - sample_nmt_prob # [batch_size, hypothesis num]
        return sample_effective_energy


    def get_diff_source_sample_effective_energy(self, energy_batch: Dict[str, Tensor], batch: Dict[str, Tensor], 
                                             gen_output_ids: Tensor) -> Tensor:
        # energy batch input & labels: must not contain incompatible tokens
        # batch input & labels: first token must be energy lang id
        # gen_output_ids: [batch_size * hypothesis_num, seq_len] 

        batch_size = len(batch['input_ids'])
        energy_vocab_size = self.energy_model.tokenizer.vocab_size
        gen_output_ids = gen_output_ids.view(batch_size, self.config['num_hypotheses_energy'], -1)

        diff_source_sample_logits = []
        for i in range(batch_size):
            # get samples from different source sequence
            diff_source_output_ids = torch.concat([gen_output_ids[0:i], gen_output_ids[i+1:]])
            # [(batch_size - 1) * (hypothesis num), (seq len)]
            diff_source_output_ids = diff_source_output_ids.view((batch_size-1) * self.config['num_hypotheses_energy'], -1)

            # randomly choose negatives 
            chosen_indices = torch.randint(low = 0, high = len(diff_source_output_ids), size = (self.config['num_hypotheses_energy'],))
            diff_source_output_ids = diff_source_output_ids[chosen_indices]
            energy_diff_source_output_ids = self.remove_incompatible_ids(diff_source_output_ids)

            # get scores [hypothesis num]
            diff_source_sample_scores = self.energy_model.forward(
                                    energy_batch['input_ids'][i, 1:].repeat(self.config['num_hypotheses_energy'],1), # leave out lang id
                                    F.one_hot(energy_diff_source_output_ids[:, 1:], energy_vocab_size).float(), # leave out lang id
                                    energy_diff_source_output_ids[:, 1:] # leave out lang id
                                    )
            
            if i == 0:
                self.log_dict({'diff_source_sample_score': diff_source_sample_scores.mean().item()})

            # get negative sampling probs
            # [hypothesis num, seq len, vocab_size]
            diff_source_seq_logits = self.model.forward(batch['input_ids'][i].repeat(self.config['num_hypotheses_energy'], 1),
                                                        batch['attention_mask'][i].repeat(self.config['num_hypotheses_energy'], 1),
                                                    labels=diff_source_output_ids).logits
            # [hypothesis num]
            diff_source_nmt_prob = -F.cross_entropy(diff_source_seq_logits.view(-1, self.tokenizer.vocab_size), 
                                                    diff_source_output_ids.view(-1),
                                                    reduction='none', 
                                                    ignore_index = self.tokenizer.pad_token_id,
                                                    label_smoothing=0.1
                                                    ).view(self.config['num_hypotheses_energy'], -1).mean(-1)
            
            if i == 0:
                self.log_dict({'diff_source_sample_sampling_prob': diff_source_nmt_prob.mean().item()})

            diff_source_sample_logits.append(diff_source_sample_scores - diff_source_nmt_prob)
            
        # batch_size, hypothesis_num
        diff_source_effective_energy = torch.stack(diff_source_sample_logits)
        return diff_source_effective_energy

    @torch.autocast("cuda")
    def update_energy_model(self, batch, update_params: bool):
        '''
        online batch: {id, input_ids, attention_mask, labels}
        offline batch: {id, input_ids, attention_mask, labels, sys_pos, sys_neg}
        score of gold label (ids): anchor
        score of high-ranked model generation: positive
        score of low-ranked model generation: negative
        '''

        torch.cuda.empty_cache()

        print("update energy model")

        energy_optim = self.optimizers()

        # update only energy model params
        if not self.config['joint_optim']:
            self.energy_model.set_params_grad(True)
            self.set_model_params_grad(False)

        batch_size = len(batch['input_ids'])
        if batch_size == 1:
            # skip batch
            return

        vocab_size = self.energy_model.tokenizer.vocab_size

        # prepare batch for energy model
        # do not modify batch in place
        energy_batch = {}
        energy_batch['id'] = batch['id']
        energy_batch['input_ids'] = self.remove_incompatible_ids(batch['input_ids'], inplace=False)
        energy_batch['labels'] = self.remove_incompatible_ids(batch['labels'], inplace=False)
        # first token must be set to lang id for MBART
        assert batch['input_ids'][0, 0] == self.tokenizer.lang_code_to_id[self.active_config['model_src_code']]
        assert batch['labels'][0, 0] == self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']]

        for n, p in self.model.named_parameters():
            if torch.isnan(p).any():
                print(f"model's parameter {n} is nan.")

        # generate samples
        if self.config['offline_generation']:
            gen_output_ids = batch['sys'] # list of list of ints
        else:
            # generate, only get the output ids
            gen_input = {"id": batch['id'], "input_ids": batch['input_ids'], "attention_mask": batch['attention_mask']}
    
            gen_output_ids = self.train_generate(gen_input,
                                            num_hypotheses=self.config['num_hypotheses_energy'],
                                            override_selfsup_strategy='sample')

            gen_output_ids = gen_output_ids[:, 1:] # leave out <eos> at front
        
        # prepare samples for energy model
        energy_gen_output_ids = self.remove_incompatible_ids(gen_output_ids)

        # 1) get gold label (score - negative_sampling_prob)
        anchor_effective_energy = self.get_gold_label_effective_energy(energy_batch, batch)
        assert anchor_effective_energy.requires_grad

        # 2) get positive sample (score - negative_sampling_prob)
        sample_effective_energy = self.get_positive_sample_effective_energy(energy_batch, batch, 
                                                                  gen_output_ids, energy_gen_output_ids)
        sample_effective_energy = sample_effective_energy.to(anchor_effective_energy.device)
        assert sample_effective_energy.requires_grad

        # 3) get negative sample (score - negative_sampling_prob)
        if not self.config['energy_loss'] == 'nce':
            diff_source_effective_energy = self.get_diff_source_sample_effective_energy(energy_batch, batch,
                                                                                 gen_output_ids) 
            diff_source_effective_energy = diff_source_effective_energy.to(anchor_effective_energy.device)
            assert diff_source_effective_energy.requires_grad


        if self.config['energy_loss'] == 'nce':
            logits = torch.concat((anchor_effective_energy.unsqueeze(1), sample_effective_energy), dim=1) # (batch, 1 + sample_num)
            assert logits.requires_grad
            targets = torch.tensor([[1] + [0] * sample_effective_energy.size(1)]*batch_size, device=logits.device).float()
            energy_loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        elif self.config['energy_loss'] == 'soft_nn_loss':
            # [batch_size, 1 + sample_num + diff_source_sample_num]
            logits = torch.concat((anchor_effective_energy.unsqueeze(1), sample_effective_energy, diff_source_effective_energy), dim=1)
            assert logits.requires_grad
            targets = torch.tensor([[1.0] + [0.5] * sample_effective_energy.size(1) 
                                    + [0.0] * diff_source_effective_energy.size(1)]*batch_size, 
                                   device=logits.device).float()
            targets = torch.softmax(targets)
            energy_loss = F.cross_entropy(logits, targets, label_smoothing=0.1)

        elif self.config['energy_loss'] == 'multi_nce':         
            # combine logits for each loss
            # [batch_size, 1 + diff_source_sample_num]
            gold_and_neg_logits = torch.concat((anchor_effective_energy.unsqueeze(1), diff_source_effective_energy), dim=1)
            assert gold_and_neg_logits.requires_grad
            # [batch_size, 1 + diff_source_sample_num] * sample_num
            pos_and_neg_logits = []
            for i in range(self.config['num_hypotheses_energy']):
                pos_and_neg_logits.append(torch.concat((sample_effective_energy[:, i].unsqueeze(1), diff_source_effective_energy), dim=1))
            for logits in pos_and_neg_logits:
                assert logits.requires_grad

            # [batch_size] 
            # correct class is 0th dim
            target = torch.tensor([0] * batch_size, device=gold_and_neg_logits.device).long()

            energy_loss_gold = F.cross_entropy(gold_and_neg_logits, target, label_smoothing=0.1)
            energy_loss_sample = []
            for i in range(self.config['num_hypotheses_energy']):
                energy_loss_sample.append(F.cross_entropy(pos_and_neg_logits[i], target, label_smoothing=0.1))
            energy_loss = energy_loss_gold + torch.mean(torch.stack(energy_loss_sample))
        

        # scale losses by 1/N (for N batches of gradient accumulation)
        energy_loss = energy_loss / self.config['accumulate_grad_batches']
        
        self.log('energy_loss', energy_loss, sync_dist=True)

        if update_params:
            log_src = self.tokenizer.batch_decode(batch['input_ids'])

            log_samples = []
            gen_output_ids = gen_output_ids.view(batch_size * self.config['num_hypotheses_energy'], -1)
            _log_samples = self.tokenizer.batch_decode(gen_output_ids) # (batch_size * seq num) number of strings
            for i in range(0, len(_log_samples), self.config['num_hypotheses_energy']):
                log_samples.append("\n".join(_log_samples[i:i+self.config['num_hypotheses_energy']]))
            
            assert len(log_samples) == batch_size
                
            labels = torch.where(batch["labels"] == -100, self.tokenizer.pad_token_id, batch["labels"])
            log_labels = self.tokenizer.batch_decode(labels)
            
            self.logger.log_text(key="training_energy",columns=["src", "all_samples", "labels"], 
                                 data=[[l1,l2,l3] for (l1,l2,l3) in zip(log_src, log_samples, log_labels)])

        torch.cuda.empty_cache()

        self.manual_backward(energy_loss)

        # if gradients are invalid, skip
        for n, p in self.energy_model.named_parameters():
            if p.grad:
                if torch.isnan(p.grad).any():
                    print(f"energy model's parameter {n}'s gradient contains nan. skipping update step")
                    energy_optim.zero_grad(set_to_none=True)
                    return

        # clip gradients
        self.clip_gradients(energy_optim, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        if update_params:
            energy_optim.step()
            energy_optim.zero_grad(set_to_none=True)


    def compute_supervised_loss(self, batch):
        sup_outputs: Seq2SeqLMOutput = self.model.forward(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])
        loss_sup, sup_logits = sup_outputs.loss, sup_outputs.logits
        assert sup_outputs.logits.requires_grad

        if self.config['offline_generation']:
            sup_outputs_pos: Seq2SeqLMOutput = self.model.forward(batch['input_ids'],
                                                              batch['attention_mask'],
                                                              labels=batch['sys_pos'])
        
            loss_sup_pos, sup_logits_pos = sup_outputs_pos.loss, sup_outputs_pos.logits

            loss_sup += loss_sup_pos
        
        return loss_sup, sup_logits

    def compute_unsupervised_loss(self, batch, num_hyp):
        batch_size = len(batch['input_ids'])

        # generate
        if self.config['offline_generation']:
            unsup_labels = batch['sys']
        else:
            unsup_output_ids: torch.LongTensor = self.train_generate(batch,
                                            num_hypotheses=num_hyp)

            unsup_labels = unsup_output_ids[:, 1:].clone()
            # leave out <eos> at the beginning -> [lang_id] Y [eos] [pad] * N
    
        # translations for same source sentence are grouped together -> must use repeat_interleave NOT repeat
        #print(batch['input_ids'])       
        repeated_batch = {}
        repeated_batch['input_ids'] = torch.repeat_interleave(batch['input_ids'], 
                                                            repeats = num_hyp, dim = 0)
        repeated_batch['attention_mask'] = torch.repeat_interleave(batch['attention_mask'], 
                                                            repeats = num_hyp, dim = 0)

        # forward pass to get gradients, teacher forcing with model generated ids as the labels
    
        unsup_outputs: Seq2SeqLMOutput = self.model.forward(repeated_batch['input_ids'], 
                                                            repeated_batch['attention_mask'], # TODO: fix error
                                                            labels=unsup_labels, 
                                                            output_attentions=True)
        assert unsup_outputs.logits.requires_grad
        assert unsup_outputs.logits.size(0) == batch_size * num_hyp
        # unsup_outputs.logits size: [batch_size * num_hypotheses_nmt, seq_length, vocab_size]

        # straight through estimator
        energy_model_vocab_size = self.energy_model.tokenizer.vocab_size # last token indices: lang id
        energy_model_input = torch.softmax(unsup_outputs.logits[:, 1:, :energy_model_vocab_size], -1) # leave out first lang id
        energy_model_input = STL(energy_model_input)

        assert energy_model_input.requires_grad
        energy_model_input.retain_grad()

        unsup_labels = self.remove_incompatible_ids(unsup_labels)

        # get energy model prediction
        scores = self.energy_model.forward(
                            batch['input_ids'][:, 1:].repeat_interleave(repeats=num_hyp, dim=0), # leave out lang id
                            energy_model_input, # [batch_size * num_hypotheses_nmt, seq_length, energy_vocab_size]
                            unsup_labels[:, 1:] # leave out lang id, [batch_size * num_hypotheses_nmt, seq_length]
                            )
        
        # scores: [batch_size * num_hypotheses_nmt]

        loss_unsup = -(scores)

        loss_unsup = loss_unsup.mean()

        loss_unsup.retain_grad()

        return loss_unsup, unsup_labels, scores, energy_model_input


    def on_train_batch_end(self, outputs, batch, batch_idx):
        '''
        logging
        '''
        if self.by_steps:
            if batch_idx == self.config['updates_per_improve']:
                self.trainer.should_stop = True
        
        if self.warmup:
            if self.trainer.global_step == self.config['warmup_steps']:
                self.warmup = False
                if self.config['offline_generation']:
                    self.trainer.should_stop = True 
            return

        if batch_idx % 64 == 0 and outputs is not None:
            unsup_src = self.tokenizer.batch_decode(batch['unlabel']['input_ids'])
            unsup_batch_size = batch['unlabel']['input_ids'].size(0)
            # log all samples
            unsup_prediction = []
            unsup_labels = []
            for t in outputs['unsup_labels']: # [4, 4 * 5, seq_len]
                # t: [4 * 5, seq_len]
                _unsup_labels = t.view(-1, self.config['num_hypotheses_nmt'], t.size(-1))
                # _unsup_labels: [4, 5, seq_len]
                for seq in _unsup_labels:
                    # seq: [5, seq_len]
                    unsup_labels.append(seq)
            # list of 4 tensors shaped [5, seq_len] -> should be 16
            for i in range(unsup_batch_size):
                print("i: ", i, "unsup_batch_size: ", unsup_batch_size)
                _unsup_prediction = self.tokenizer.batch_decode(unsup_labels[i])
                unsup_prediction.append('\n'.join(_unsup_prediction))

            self.logger.log_text(key="training_unsup", columns=["unsup_src", "unsup_prediction"], 
                                 data = [[x,y] for (x,y) in zip(unsup_src, unsup_prediction)])

            self.logger.log_metrics({"train_loss_sup": outputs['loss_sup'].item(),
                                     "train_loss_unsup": outputs['loss_unsup'].item(),
                                     "train_loss": outputs['loss'].item()})
    

    def warmup_step(self, batch, batch_idx):
        trans_optim = self.optimizers()
        if 'unlabel' in batch.keys():
            batch = batch['label']
        outputs = self.model.forward(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])
        loss = outputs[0]
        self.log("train_loss", loss, sync_dist=True)

        # gradient accumulation
        loss = loss / self.config['accumulate_grad_batches']

        trans_optim.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        
        # clip gradients
        self.clip_gradients(trans_optim, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        if (batch_idx + 1) % self.config['accumulate_grad_batches'] == 0:
            trans_optim.step()

    def grad_project(self, loss_sup: Tensor, loss_unsup: Tensor):
        print("grad project")

        optim = self.optimizers()
        optim.zero_grad(set_to_none=True)

        # get 2D grads for each parameter
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        loss_sup_grad = torch.autograd.grad(loss_sup, model_params, retain_graph=True)
        loss_unsup_grad = torch.autograd.grad(loss_unsup, model_params, retain_graph=True)
            
        # pack into one 1D tensor
        loss_sup_grad = torch.concat([grad.flatten() for grad in loss_sup_grad])
        loss_unsup_grad = torch.concat([grad.flatten() for grad in loss_unsup_grad])

        sup_unsup_dot = loss_unsup_grad.dot(loss_sup_grad)

        if sup_unsup_dot.item() >= 0.0:
            self.loss_weights[1] = min(self.config['unsup_wt'], sup_unsup_dot/(loss_unsup_grad.dot(loss_unsup_grad)))
        else:
            self.loss_weights[1] = 0.0
            
        self.loss_weights[0] = 1.0
              
    def grad_norm(self, loss_sup: Tensor, loss_unsup: Tensor):
        '''
        adapter from LucasBoTang GradNorm repo
        '''
        print("grad norm")

        weights_optim = self.optimizers()
        weights_optim.zero_grad(set_to_none=True)
        self.energy_model.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        self.loss_weights.requires_grad_(True)

        # compute the L2 norm of the model gradients for each task
        # don't set model gradients requires_grad to False before this
        model_params = [p for p in self.model.parameters() if p.requires_grad]

        sup_grads = torch.autograd.grad(self.loss_weights[0] * loss_sup, model_params, retain_graph=True, create_graph=True)
        sup_grads = torch.concat([grad.flatten() for grad in sup_grads]) # tuple of 2D tensors -> 1D tensor
        sup_grad_norm = torch.norm(sup_grads)
        print("sup_grad_norm: ", sup_grad_norm)

        unsup_grads = torch.autograd.grad(self.loss_weights[1] * loss_unsup, model_params, retain_graph=True, create_graph=True)
        unsup_grads = torch.concat([grad.flatten() for grad in unsup_grads]) # tuple of 2D tensors -> 1D tensor
        unsup_grad_norm = torch.norm(unsup_grads)
        print("unsup_grad_norm: ", unsup_grad_norm)

        grad_norms = torch.stack([sup_grad_norm, unsup_grad_norm])
        print("grad_norms: ", grad_norms)

        # compute loss ratio per task
        unweighted_loss = torch.stack([loss_sup, loss_unsup])
        loss_ratio = unweighted_loss.detach() / self.initial_loss

        # compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()

        # compute the average gradient norm
        grad_norm_avg = grad_norms.mean().detach()

        # compute the GradNorm loss
        constant = (grad_norm_avg * rt ** 0.5).detach() # alpha = 0.5
        print("constant: ", constant)
        gradnorm_loss = torch.abs(grad_norms - constant).sum()
        print("gradnorm_loss: ", gradnorm_loss)

        # backward pass and update for GradNorm
        self.manual_backward(gradnorm_loss, retain_graph=True)

        # update only loss weights
        self.energy_model.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)

        weights_optim.step()

        # renormalize weights
        new_weights = torch.min((self.loss_weights / self.loss_weights.sum() * self.initial_weight_sum).detach(), 
                                torch.tensor(5.0).to(self.loss_weights.device))
        self.loss_weights = torch.nn.Parameter(new_weights)
        weights_optim.optimizer.param_groups[2]["params"] = iter([self.loss_weights])
        self.loss_weights.requires_grad_(False)

        print("done with grad norm")


    def update_nmt_model(self, batch, update_params: bool, batch_idx):

        print("update nmt model")

        trans_optim = self.optimizers()

        self.set_model_params_grad(True)
        self.energy_model.set_params_grad(True)
        
        '''
        energy_params = list(self.energy_model.parameters())
        initial_l2_norm = torch.norm(torch.cat([p.flatten() for p in energy_params]), p=2)
        '''
        
        loss_sup, sup_logits = self.compute_supervised_loss(batch['label'])

        # unsup_labels: [4 * 5, seq_len]
        loss_unsup, unsup_labels, energy_model_prediction, energy_model_input = self.compute_unsupervised_loss(batch['unlabel'], self.config['num_hypotheses_nmt'])

        if self.global_step == 0:
            self.initial_loss = torch.stack([loss_sup, loss_unsup])

        if update_params:
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

            elif self.config['weight_schedule_strategy'] == 'grad_projection':
                self.grad_project(loss_sup, loss_unsup)
        
        # do nothing here for grad norm
        
        self.log("sup_weight", self.loss_weights[0].item())
        self.log("unsup_weight", self.loss_weights[1].item())

        # calculate total loss with current loss weights
        loss = self.loss_weights.dot(torch.stack([loss_sup, loss_unsup]))

        # gradient accumulation
        loss = loss / self.config['accumulate_grad_batches']

        # update weights with model grads
        if self.config['weight_schedule_strategy'] == 'grad_norm':
            self.grad_norm(loss_sup, loss_unsup)

        self.manual_backward(loss) # during gradient calculation, let energy model gradients propagate

        # clip gradients
        self.clip_gradients(trans_optim, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # if gradients are invalid, skip
        for n, p in self.model.named_parameters():
            if p.grad:
                if torch.isnan(p.grad).any():
                    print(f"model's parameter {n}'s gradient contains nan. skipping update step")
                    trans_optim.zero_grad(set_to_none=True)
                    return

        # update nmt model
        if update_params:
            # during update, freeze energy model
            self.energy_model.zero_grad(set_to_none=True)
            trans_optim.step()
            trans_optim.zero_grad(set_to_none=True)

        '''
        energy_params = list(self.energy_model.parameters())
        post_update_l2_norm = torch.norm(torch.cat([p.flatten() for p in energy_params]), p=2)
        print(initial_l2_norm)
        print(post_update_l2_norm)
        assert initial_l2_norm == post_update_l2_norm
        '''
        #self.trainer.strategy.barrier()

        return {"loss": loss, "loss_sup": loss_sup, "loss_unsup": loss_unsup, 
                "unsup_labels": unsup_labels}

    def training_step(self, batch, batch_idx):
        '''
        online training: batch['label'] = {'input_ids', 'attention_mask', 'labels'}
                         batch['unlabel] = {'input_ids', 'attention_mask'}
        offline training: batch['label'] = {'input_ids', 'attention_mask', 'labels',
                                            'sys_pos', 'sys_neg'}
                          batch['unlabel'] = {'input_ids', 'attention_mask', 'sys'}
        '''

        start_time = time.time()
        torch.cuda.empty_cache()

        trans_optim = self.optimizers()

        # retrieve better unsupervised data
        if self.config['active_unlabeled_retrieval']:
            result = retrieve_unlabeled_batch_precomputed(self.datamodule, self.tokenizer, self.config, batch['label'])
            batch['unlabel'] = result['batch']

            if self.config['timing_run']:
                with open(f"timing/{self.run_id}/retrieve_unlabeled_duration.txt", 'w') as f:
                    f.write(str(result['duration']))

            batch['unlabel']['input_ids'] = batch['unlabel']['input_ids'].to(self.device)
            batch['unlabel']['attention_mask'] = batch['unlabel']['attention_mask'].to(self.device)

        if self.warmup:
            self.warmup_step(batch, batch_idx)
            return

        effective_sup_batch_size = len(batch['label']['input_ids'])
        effective_unsup_batch_size = len(batch['unlabel']['input_ids'])
        accumulate_sup_grad_batches = min(self.config['accumulate_grad_batches'], 
                                      math.ceil(effective_sup_batch_size/self.config['sup_batch_size']))
        accumulate_unsup_grad_batches = min(self.config['accumulate_grad_batches'],
                                      math.ceil(effective_unsup_batch_size/self.config['unsup_batch_size']))

        train_energy_step = self.config['train_energy'] and (batch_idx % self.config['energy_update_interval'] == 0
            ) and (self.trainer.global_step >= self.config['energy_update_warmup']
            ) and (self.trainer.current_epoch < self.config['train_energy_epoch'])
        
        print("train_energy_step: ", train_energy_step)
        
        if train_energy_step or (self.config['train_energy'] and self.config['timing_run']):
            for i in range(accumulate_sup_grad_batches):
                # update only at end 
                update_params = (i == (accumulate_sup_grad_batches - 1))

                start_idx = i * self.config['sup_batch_size']
                end_idx = min((i+1) * self.config['sup_batch_size'], effective_sup_batch_size)

                small_batch = {'id': batch['label']['id'][start_idx:end_idx],
                                'input_ids': batch['label']['input_ids'][start_idx:end_idx],
                               'attention_mask': batch['label']['attention_mask'][start_idx:end_idx],
                               'labels': batch['label']['labels'][start_idx:end_idx]
                                }
                
                self.update_energy_model(small_batch, update_params)

        accumulate_grad_batches = min(accumulate_sup_grad_batches, accumulate_unsup_grad_batches)

        outputs = {}
        
        for i in range(accumulate_grad_batches):
            # update only at end 
            update_params = (i == (accumulate_grad_batches - 1))

            sup_start_idx = i * self.config['sup_batch_size']
            sup_end_idx = min((i+1) * self.config['sup_batch_size'], effective_sup_batch_size)

            unsup_start_idx = i * self.config['unsup_batch_size']
            unsup_end_idx = min((i+1) * self.config['unsup_batch_size'], effective_unsup_batch_size)

            small_batch = {'label': {
                                'id': batch['label']['id'][sup_start_idx:sup_end_idx],
                                'input_ids': batch['label']['input_ids'][sup_start_idx:sup_end_idx],
                                'attention_mask': batch['label']['attention_mask'][sup_start_idx:sup_end_idx],
                                'labels': batch['label']['labels'][sup_start_idx:sup_end_idx]
                                },
                            'unlabel': {
                                'id': batch['unlabel']['id'][unsup_start_idx:unsup_end_idx],
                                'input_ids': batch['unlabel']['input_ids'][unsup_start_idx:unsup_end_idx],
                                'attention_mask': batch['unlabel']['attention_mask'][unsup_start_idx:unsup_end_idx]
                                }
                            }

            #try:
            small_outputs = self.update_nmt_model(small_batch, update_params, batch_idx)

            #except Exception as e:
            #    print(e)
            #    small_outputs = {}

            for key in small_outputs:
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(small_outputs[key])

        # aggregate outputs
        outputs["loss"] = torch.mean(torch.stack(outputs["loss"]))
        outputs["loss_sup"] = torch.mean(torch.stack(outputs["loss_sup"]))
        outputs["loss_unsup"] = torch.mean(torch.stack(outputs["loss_unsup"]))

        end_time = time.time()
        duration = end_time - start_time

        if self.config['train_energy']:
            method = 'qe_dynamic'
        else:
            method = 'qe_static'
        
        if self.config['timing_run']:
            with open(f"timing/{self.run_id}/{method}_step.txt", 'w') as f:
                f.write(str(duration))
            if batch_idx == 0:
                exit(0)

        return outputs
    

    def predict_step(self, batch, batch_idx: int) -> Any:
        batch_size = len(batch['input_ids'])
        batch = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

        # generate translations
        translation_batch = self.train_generate(batch, self.num_hypotheses)
        translation_batch = translation_batch[:, 1:] # take off <eos>

        return translation_batch.cpu() # to save gpu memory
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        if "labels" in batch:
            labels = batch["labels"]
            batch.pop("labels")
        else:
            labels = None

        if "decoder_input_ids" in batch:
            batch.pop("decoder_input_ids")
        
        if not self.config['eval_teacher_forcing']:
            # generated translation
            if self.config['ranking']:
                num_hypotheses = 5
            else:
                num_hypotheses = 1

            pred_ids = self.eval_generate(batch,
                            num_hypotheses=num_hypotheses)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat_interleave(num_hypotheses, 1)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat_interleave(num_hypotheses, 1)

            # forward pass, teacher forcing with generated translation
            beam_outputs: Seq2SeqLMOutput = self.model(**repeated_batch, labels=pred_labels, output_attentions=True)
            # prepare attention (last layer of decoder)
            cross_attention = beam_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # (batch_size, num_heads, target seq length, input seq length)
            encoder_attention = beam_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = beam_outputs.decoder_attentions[-1][:, :, 1:, 1:]

            if self.config['ranking']:
                pred_labels, cross_attention, encoder_attention, decoder_attention = self.rank(num_hypotheses, repeated_batch['input_ids'], pred_labels, cross_attention, encoder_attention, decoder_attention)

        # calculate loss, teacher forcing with real output
        if labels is not None:
            teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
            if self.config["eval_teacher_forcing"]:
                pred_labels = torch.argmax(teacher_forced_outputs.logits, dim=-1)
                cross_attention = teacher_forced_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # leave out attention for language token ids
                encoder_attention = teacher_forced_outputs.encoder_attentions[-1][:, :, 1:, 1:]
                decoder_attention = teacher_forced_outputs.decoder_attentions[-1][:, :, 1:, 1:]
            val_loss = teacher_forced_outputs.loss
            self.log('val_loss', val_loss.item(), sync_dist=True) # loss is unsupervised loss (targets are beam search output, not gold target)
        else:
            val_loss = None

        # pred_labels: [trg lang code] X [eos] [pad] 
        # labels: [trg lang code] X [eos] [pad] 

        return {"loss": val_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention, "encoder_attention": encoder_attention, "decoder_attention": decoder_attention}

    @torch.enable_grad()
    def grad_analysis(self, batch):
        trans_optim = self.optimizers()

        self.set_model_params_grad(True)
        self.energy_model.set_params_grad(True)
        
        '''
        energy_params = list(self.energy_model.parameters())
        initial_l2_norm = torch.norm(torch.cat([p.flatten() for p in energy_params]), p=2)
        '''
        
        ids: Tensor = batch['id']
        ids = ids.tolist()
        batch_size = len(batch['id'])

        if 'decoder_input_ids' in batch:
            batch.pop('decoder_input_ids')
        if 'labels' in batch:
            labels = batch.pop('labels')
        
        num_hyp = 1 # NOT self.config['num_hypotheses_nmt']
    
        # unsup_labels: [4 * 5, seq_len]
        loss_unsup, unsup_labels, energy_model_prediction, energy_model_input = self.compute_unsupervised_loss(batch, 1)

        self.manual_backward(loss_unsup) # during gradient calculation, let energy model gradients propagate

        # clip gradients
        self.clip_gradients(trans_optim, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # if gradients are invalid, skip
        for n, p in self.model.named_parameters():
            if p.grad:
                if torch.isnan(p.grad).any():
                    print(f"model's parameter {n}'s gradient contains nan. skipping update step")
                    trans_optim.zero_grad(set_to_none=True)
                    return

        # energy_model_input: [batch_size * num_hypotheses_nmt, seq_length, energy_vocab_size]
        # unsup_labels: [batch_size * num_hypotheses_nmt, seq_length]
        rcParams['font.family'] = 'FreeSerif'

        for index in range(batch_size):
            seq_length = unsup_labels.size(1)-1 # leave out lang id
            assert energy_model_input.grad.size(1) == seq_length
            grads = energy_model_input.grad[index, torch.arange(seq_length), unsup_labels[index][1:]]
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
            filtered_grads = filtered_grads.numpy()

            colormap = plt.cm.viridis 
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
            
            fig.savefig(f"grad_analysis/{self.run_id}/ebm_grad_analysis_id{ids[index]}.png")
            with open(f"grad_analysis/{self.run_id}/ebm_tokens_id{ids[index]}.txt", 'w') as f:
                f.write(json.dumps(unsup_labels_tokens, ensure_ascii=False))
            with open(f"grad_analysis/{self.run_id}/ebm_strings_id{ids[index]}.json", 'w') as f:
                text_dict = {"input": unsup_input_str, "labels": label_str, "output": unsup_labels_str}
                f.write(json.dumps(text_dict, ensure_ascii=False))
            filtered_grads.dump(f"grad_analysis/{self.run_id}/ebm_grads_id{ids[index]}.pkl")

        self.energy_model.zero_grad(set_to_none=True)
        trans_optim.zero_grad(set_to_none=True)

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

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        if "labels" in batch:
            labels = batch["labels"]
            batch.pop("labels")
        else:
            labels = None

        if "decoder_input_ids" in batch:
            batch.pop("decoder_input_ids")

        if not self.config['eval_teacher_forcing']:

            # generated translation
            if self.config['ranking']:
                num_hypotheses = 5
            else:
                num_hypotheses = 1

            pred_ids = self.eval_generate(batch,
                           num_hypotheses=num_hypotheses)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat_interleave(num_hypotheses, 1)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat_interleave(num_hypotheses, 1)

            # forward pass, teacher forcing with generated translation
            beam_outputs: Seq2SeqLMOutput = self.model(**repeated_batch, labels=pred_labels, output_attentions=True)
            # prepare attention (last layer of decoder)
            cross_attention = beam_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # (batch_size, num_heads, target seq length, input seq length)
            encoder_attention = beam_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = beam_outputs.decoder_attentions[-1][:, :, 1:, 1:]

            if self.config['ranking']:
                pred_labels, cross_attention, encoder_attention, decoder_attention = self.rank(num_hypotheses, repeated_batch['input_ids'], pred_labels, cross_attention, encoder_attention, decoder_attention)

        # calculate loss, teacher forcing with real output
        if labels is not None:
            teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
            if self.config["eval_teacher_forcing"]:
                pred_labels = torch.argmax(teacher_forced_outputs.logits, dim=-1)
                cross_attention = teacher_forced_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # leave out attention for language token ids
                encoder_attention = teacher_forced_outputs.encoder_attentions[-1][:, :, 1:, 1:]
                decoder_attention = teacher_forced_outputs.decoder_attentions[-1][:, :, 1:, 1:]
            test_loss = teacher_forced_outputs.loss
            self.log('test_loss', test_loss.item(), sync_dist=True) # loss is unsupervised loss (targets are beam search output, not gold target)
        else:
            test_loss = None

        # pred_labels: [trg lang code] X [eos] [pad] 
        # labels: [trg lang code] X [eos] [pad] 

        return {"loss": test_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention, "encoder_attention": encoder_attention, "decoder_attention": decoder_attention}

    def configure_optimizers(self):
        all_params = [{"params": self.model.parameters(), "lr": self.config['learning_rate']},
                    {"params": self.energy_model.parameters(), "lr": self.config['energy_learning_rate']}]
        
        if self.config['weight_schedule_strategy'] == "grad_norm":
            all_params.append({"params": iter([self.loss_weights]), "lr": self.config["loss_weight_learning_rate"]})

        trans_optimizer = Adam(
                         all_params,
                         eps=1e-06,
                         betas=(0.9, 0.98),
                         weight_decay=1e-05,
                         )
        
        '''
        # can't use multiple optimizers for Deepspeed
        energy_optimizer = Adam(self.energy_model.parameters(),
                         eps=1e-06,
                         betas=(0.9, 0.98),
                         weight_decay=1e-05,
                         lr=self.config['learning_rate'])
        
        '''

        print("optim params: ", count_optim_params(trans_optimizer))

        return {
            "optimizer": trans_optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(trans_optimizer, 2),
                "monitor": "val_comet_kiwi",
                }
        }
