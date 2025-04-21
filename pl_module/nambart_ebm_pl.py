from typing import Any, Dict

import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
import wandb
import numpy as np

from transformers import (
    PreTrainedTokenizer, MBart50TokenizerFast, AutoTokenizer)
from adapters import (setup_adapter_training, AdapterArguments)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation.utils import GenerateOutput

from score.score import Scorer
from pl_module.utils import (check_trainable_params, register_param_hooks, 
                             get_length, STL, get_adapters, count_optim_params
                             )
from model.custom_mbart import NAMBartForConditionalGeneration
from pl_module.mbart_ebm_pl import EnergyModel
from generation_strategy.mask_predict import MaskPredict, predict_length_beam, duplicate_encoder_out, _mean_pooling
    

class NAMBARTSsl_EBMPL(pl.LightningModule):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer, prepare_energy_model,
                by_steps: bool = False, warmup: bool = False,
                ):
        super().__init__()
        self.save_hyperparameters()
        self.active_config = active_config
        self.config = config
        self.tokenizer: MBart50TokenizerFast = tokenizer
        self.by_steps = by_steps
        self.warmup = warmup
        self.num_hypotheses = 1 # for offline generation
        self.clip_range = 0.5

        self.automatic_optimization = False

        # prepare NMT model
        self.model = NAMBartForConditionalGeneration.from_pretrained(self.active_config['model_name_or_path'])
        self.model.cuda()

        # NAT setup
        self.strategy = MaskPredict(decoding_iterations=10, tokenizer=self.tokenizer)
        self.dynamic_length = False
        self.random = np.random.RandomState(self.config['seed'])
        self.mask_range = True
        self.embed_lengths = nn.Embedding(self.config['max_length'], self.model.config.d_model)

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

        self.scorer = Scorer(self.active_config, self.config, self.tokenizer, None, None, device=torch.device("cuda"))
        
        print("nmt model trainable params: ", check_trainable_params(self.model))

        # register backward hooks for NMT model
        if self.config['adapter']:
            register_param_hooks(self.model.get_encoder().get_invertible_adapter(), "nmt model [ADAPTER]")
        else:
            register_param_hooks(self.model.get_encoder().base_model, "nmt model")

        # log gradients
        wandb.watch(self.model, log_freq=100)
        wandb.watch(self.energy_model.base_model, log_freq=100)


    def predict_lengths(self, encoder_out, src_masks): # B X T
        enc_feats = encoder_out.transpose(0,1)  # T x B x C
        enc_feats = _mean_pooling(enc_feats, src_masks) # T X B
        length_out = F.linear(enc_feats, self.embed_lengths.weight)
        return F.log_softmax(length_out, -1)

    def mask_predict(self, batch, output_attentions):
        """
        batch: dict("input_ids", "attention_mask", "labels")
        """
        print("mask_predict")
        src_tokens = batch['input_ids']
        src_tokens = src_tokens.new(src_tokens.tolist())
        bsz = src_tokens.size(0)
    
        outputs = self.model.forward(**batch, output_hidden_states=True, output_attentions=True, return_dict=True)
        encoder_out = outputs.encoder_last_hidden_state
        encoder_hidden_states = outputs.encoder_hidden_states
        encoder_attentions = outputs.encoder_attentions
        predicted_lengths = self.predict_lengths(encoder_out, batch["attention_mask"])
        beam = predict_length_beam(None, predicted_lengths, 1)
    
        max_len = beam.max().item()
        length_mask = torch.triu(src_tokens.new(max_len, max_len).fill_(1).long(), 1)
        # triu: returns upper triangle (diagonal retained, other elements: 0)
        length_mask = torch.stack([length_mask[beam[batch] - 1] for batch in range(bsz)], dim=0)
        tgt_tokens = src_tokens.new(bsz, 1, max_len).fill_(self.tokenizer.mask_token_id)
        tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * self.tokenizer.pad_token_id
        tgt_tokens = tgt_tokens.view(bsz * 1, max_len)

        # add target lang code at beginning
        tgt_tokens[:, 0] = self.tokenizer.convert_tokens_to_ids(self.active_config['trg_code'])

        print("tgt_tokens", tgt_tokens)
    
        duplicate_encoder_out(encoder_out, batch['attention_mask'], bsz, 1)
        hypotheses, lprobs, all_token_probs, cross_attention = self.strategy.generate(self.model, encoder_out, encoder_hidden_states, encoder_attentions, tgt_tokens, batch["attention_mask"])
    
        hypotheses = hypotheses.view(bsz, 1, max_len)
        lprobs = lprobs.view(bsz, 1)
        tgt_lengths = (1 - length_mask).sum(-1)
        avg_log_prob = lprobs / tgt_lengths.float()
        best_lengths = avg_log_prob.max(-1)[1]
        hypotheses = torch.stack([hypotheses[b, l, :] for b, l in enumerate(best_lengths)], dim=0)

        if output_attentions:
            return hypotheses, all_token_probs, cross_attention
        return hypotheses, all_token_probs

    def eval_generate(self, batch: Dict, num_hypotheses: int) -> LongTensor:
        # to be used in eval
        # NON-AUTOREGRESSIVE
        # batch: {input_ids, attention_mask}
        repeated_batch = {'input_ids': batch['input_ids'].repeat(num_hypotheses),
                          'attention_mask': batch['input_ids'].repeat(num_hypotheses)}

        pred_labels, outputs, cross_attention = self.mask_predict(repeated_batch, output_attentions=True)
        
        return pred_labels
    
    def train_generate(self, batch: Dict, num_hypotheses: int = 1, 
                       return_dict_in_generate: bool = False, 
                       output_attentions: bool = False) -> LongTensor:
        # to be used in train
        # NON-AUTOREGRESSIVE

        pred_labels, outputs, cross_attention = self.mask_predict(batch, output_attentions=output_attentions)

        return pred_labels
    
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

        anchor_score  = self.energy_model.forward(
                                    energy_batch['input_ids'][:, 1:], # leave out lang id
                                    F.one_hot(energy_batch['labels'][:, 1:], energy_vocab_size).float(), # leave out lang id
                                    energy_batch['labels'][:, 1:] # leave out lang id
                                    )
        
        assert anchor_score.requires_grad
        
        self.log_dict({'gold_label_score': anchor_score.mean().item()})
        
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
                                    energy_batch['input_ids'][:, 1:].repeat(self.config['num_hypotheses_energy'],1), # leave out lang id
                                    F.one_hot(energy_gen_output_ids[:, 1:], energy_vocab_size).float(), # leave out lang id
                                    energy_gen_output_ids[:, 1:] # leave out lang id
                                    )

        scores = scores.view(batch_size, -1) # [batch_size, hypothesis num]
        self.log_dict({'positive_sample_score': scores[0].mean().item()})

        # get negative sampling probs
        sample_nmt_prob = []
        gen_output_ids = gen_output_ids.view(batch_size, self.config['num_hypotheses_energy'], -1)
        energy_gen_output_ids = energy_gen_output_ids.view(batch_size, self.config['num_hypotheses_energy'], -1)
        
        for i in range(batch_size):
            # _gen_output_ids: size = (self.config['num_hypotheses_energy'], seq_len)
            _gen_output_ids = gen_output_ids[i].contiguous()
            
            # [hypothesis num, seq_len (target), vocab_size]
            seq_logits = self.model.forward(batch['input_ids'][i].repeat(self.config['num_hypotheses_energy'], 1),
                                            batch['attention_mask'][i].repeat(self.config['num_hypotheses_energy'], 1),
                                            labels=_gen_output_ids).logits

            assert not torch.isnan(seq_logits).any()
            
            # [hypothesis num]
            seq_prob = -F.cross_entropy(seq_logits.view(-1, self.tokenizer.vocab_size), _gen_output_ids.view(-1),
                                            reduction='none', 
                                            ignore_index=self.tokenizer.pad_token_id, 
                                            label_smoothing=0.1).view(self.config['num_hypotheses_energy'], -1).mean(-1)

            sample_nmt_prob.append(seq_prob)

        sample_nmt_prob = torch.stack(sample_nmt_prob) # [batch_size, hypothesis num]
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


    def update_energy_model(self, batch, batch_idx):
        '''
        online batch: {input_ids, attention_mask, labels}
        offline batch: {input_ids, attention_mask, labels, sys_pos, sys_neg}
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
            gen_input = {"input_ids": batch['input_ids'], "attention_mask": batch['attention_mask']}
    
            gen_output_ids = self.train_generate(gen_input,
                                            num_hypotheses=self.config['num_hypotheses_energy'])

            gen_output_ids = gen_output_ids[:, 1:] # leave out <eos> at front
        
        # prepare samples for energy model
        energy_gen_output_ids = self.remove_incompatible_ids(gen_output_ids)

        # 1) get gold label (score - negative_sampling_prob)
        anchor_effective_energy = self.get_gold_label_effective_energy(energy_batch, batch)

        # 2) get positive sample (score - negative_sampling_prob)
        sample_effective_energy = self.get_positive_sample_effective_energy(energy_batch, batch, 
                                                                  gen_output_ids, energy_gen_output_ids)

        # 3) get negative sample (score - negative_sampling_prob)
        diff_source_effective_energy = self.get_diff_source_sample_effective_energy(energy_batch, batch,
                                                                                 gen_output_ids) 

        # make sure all effective energy are on same device before loss calculation
        sample_effective_energy = sample_effective_energy.to(anchor_effective_energy.device)
        diff_source_effective_energy = diff_source_effective_energy.to(anchor_effective_energy.device)

        assert anchor_effective_energy.requires_grad
        assert sample_effective_energy.requires_grad
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

        if batch_idx % 64 == 0:
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

        energy_optim.zero_grad(set_to_none=True)
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

        if (batch_idx + 1) % self.config['accumulate_grad_batches'] == 0:
            # assume: no gradient accumulation
            energy_optim.step()


    def compute_supervised_loss(self, batch, batch_idx):
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

    def compute_unsupervised_loss(self, batch, batch_idx):

        # generate
        if self.config['offline_generation']:
            unsup_labels = batch['sys']
        else:
            unsup_output_ids: torch.LongTensor = self.train_generate(batch,
                                            num_hypotheses=self.config['num_hypotheses_nmt'])

            unsup_labels = unsup_output_ids[:, 1:].clone()
            # leave out <eos> at the beginning -> [lang_id] Y [eos] [pad] * N

        # forward pass to get gradients, teacher forcing with model generated ids as the labels
        unsup_outputs: Seq2SeqLMOutput = self.model.forward(batch['input_ids'].repeat(self.config['num_hypotheses_nmt'], 1), 
                                                            batch['attention_mask'].repeat(self.config['num_hypotheses_nmt'], 1), 
                                                            labels=unsup_labels, 
                                                            output_attentions=True)
        assert unsup_outputs.logits.requires_grad

        # straight through estimator
        energy_model_vocab_size = self.energy_model.tokenizer.vocab_size # last token indices: lang id
        energy_model_input = torch.softmax(unsup_outputs.logits[:, 1:, :energy_model_vocab_size], -1) # leave out first lang id
        energy_model_input = STL(energy_model_input)

        assert energy_model_input.requires_grad

        unsup_labels = self.remove_incompatible_ids(unsup_labels)
        
        # get energy model prediction
        
        scores = self.energy_model.forward(
                            batch['input_ids'][:, 1:].repeat(self.config['num_hypotheses_nmt'], 1), # leave out lang id
                            energy_model_input,
                            unsup_labels[:, 1:] # leave out lang id
                            )

        if self.config['filter']:
            good_samples = (scores > 0.5) # mask for good samples
            num_good_samples = torch.sum(good_samples).item()
            print("good samples: ", num_good_samples)
            scores[~good_samples] = 0 # make it impossible to backpropagate through bad samples 
        
        if self.config['energy_model_clamp']:
            loss_unsup = -torch.clamp(scores, self.config['energy_clamp_min'], self.config['energy_clamp_max'])

        elif self.config['energy_model_hinge']:
            loss_unsup = -scores + torch.max(torch.zeros_like(scores), scores - 1.0)

        elif self.config['energy_model_scale']:
            loss_unsup = -torch.log(torch.sigmoid(scores))

        elif self.config['energy_sigmoid']:
            loss_unsup = -torch.sigmoid(scores.detach()) * (scores)
        
        else:
            loss_unsup = -(scores)

        if self.config['pseudo_label_loss']:
            # reshape outputs, labels and calculate loss
            batch_size = unsup_outputs.logits.shape[0]
            output_dim = unsup_outputs.logits.shape[-1]
            psuedo_logits = unsup_outputs.logits.contiguous().view(-1, output_dim) # [batch size * trg len, output_dim]
            pseudo_labels = unsup_labels.contiguous().view(-1) # [batch size * trg len]
            pseudo_loss = F.cross_entropy(psuedo_logits, pseudo_labels, 
                                            reduction='none', 
                                            ignore_index = self.model.config.pad_token_id
                                            ).view(batch_size, -1).sum(dim=1) # batch size

            loss_unsup = (loss_unsup + pseudo_loss * torch.sigmoid(scores.detach())).mean()
        else:
            loss_unsup = loss_unsup.mean()

        loss_unsup.retain_grad()

        return loss_unsup, unsup_labels, scores


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

            sup_src = self.tokenizer.batch_decode(batch['label']['input_ids'])
            sup_labels = torch.where(batch['label']['labels'] == -100, self.model.config.pad_token_id, batch['label']['labels'])
            sup_labels = self.tokenizer.batch_decode(sup_labels)
            sup_prediction = self.tokenizer.batch_decode(torch.argmax(outputs['sup_logits'],-1))
            self.logger.log_text(key="training_sup",columns=["sup_src", "sup_prediction", "sup_labels"], 
                                 data=[[x,y,z] for (x,y,z) in zip(sup_src, sup_prediction, sup_labels)])

            unsup_src = self.tokenizer.batch_decode(batch['unlabel']['input_ids'])
            unsup_batch_size = batch['unlabel']['input_ids'].size(0)
            # just log the first sample
            unsup_prediction = self.tokenizer.batch_decode(
                                outputs['unsup_labels'].
                                view(unsup_batch_size, self.config['num_hypotheses_nmt'], -1)[:,0,:]
                                )

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

    def training_step(self, batch, batch_idx):
        '''
        online training: batch['label'] = {'input_ids', 'attention_mask', 'labels'}
                         batch['unlabel] = {'input_ids', 'attention_mask'}
        offline training: batch['label'] = {'input_ids', 'attention_mask', 'labels',
                                            'sys_pos', 'sys_neg'}
                          batch['unlabel'] = {'input_ids', 'attention_mask', 'sys'}
        '''

        torch.cuda.empty_cache()

        trans_optim = self.optimizers()

        if self.warmup:
            self.warmup_step(batch, batch_idx)
            return

        if self.config['train_energy'] and (batch_idx % self.config['energy_update_interval'] == 0
            ) and (self.trainer.global_step >= self.config['energy_update_warmup']
            ) and (self.trainer.current_epoch < self.config['train_energy_epoch']):
            self.update_energy_model(batch['label'], batch_idx)

        # update only nmt model params
        print("update nmt model")
        self.set_model_params_grad(True)
        self.energy_model.set_params_grad(True)
        
        '''
        energy_params = list(self.energy_model.parameters())
        initial_l2_norm = torch.norm(torch.cat([p.flatten() for p in energy_params]), p=2)
        '''
        
        loss_sup, sup_logits = self.compute_supervised_loss(batch['label'], batch_idx)
    
        loss_unsup, unsup_labels, energy_model_prediction = self.compute_unsupervised_loss(batch['unlabel'], batch_idx)

        if self.global_step == 0:
            self.initial_loss = torch.stack([loss_sup, loss_unsup])

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

        trans_optim.zero_grad(set_to_none=True)
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
        if (batch_idx + 1) % self.config['accumulate_grad_batches'] == 0:
            # during update, freeze energy model
            self.energy_model.zero_grad(set_to_none=True)
            trans_optim.step()

        
        # check that energy model is not updated
        # TODO: sometimes, assertion x hold in the middle of training
        '''
        energy_params = list(self.energy_model.parameters())
        post_update_l2_norm = torch.norm(torch.cat([p.flatten() for p in energy_params]), p=2)
        print(initial_l2_norm)
        print(post_update_l2_norm)
        assert initial_l2_norm == post_update_l2_norm
        '''

        #self.trainer.strategy.barrier()

        return {"loss": loss, "loss_sup": loss_sup, "loss_unsup": loss_unsup, 
                "sup_logits": sup_logits, "unsup_labels": unsup_labels, 
                "energy_model_prediction": energy_model_prediction}
    

    def predict_step(self, batch, batch_idx: int) -> Any:
        batch_size = len(batch['input_ids'])
        batch = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

        # generate translations
        translation_batch = self.train_generate(batch, self.num_hypotheses)
        translation_batch = translation_batch[:, 1:] # take off <eos>

        return translation_batch.cpu() # to save gpu memory
    
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

            pred_ids = self.eval_generate(batch,
                            num_hypotheses=num_hypotheses)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat(num_hypotheses, 1)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat(num_hypotheses, 1)

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

            pred_ids = self.eval_generate(batch,
                           num_hypotheses=num_hypotheses)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat(num_hypotheses, 1)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat(num_hypotheses, 1)

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
