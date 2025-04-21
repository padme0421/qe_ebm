import argparse
import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam

import pytorch_lightning as pl

from transformers import PreTrainedTokenizer

from mask_predict.fairseq.models.bert_seq2seq import Transformer_nonautoregressive
from mask_predict.fairseq.tasks.translation_self import TranslationSelfTask
from mask_predict.generate_cmlm import generate
from mask_predict.fairseq import utils
from mask_predict.fairseq.strategies.mask_predict import MaskPredict
from mask_predict.fairseq.data.dictionary import Dictionary

from score.score import Scorer

class InverseSqrtScheduler(LambdaLR):
    # code borrowed from https://blog.hjgwak.com/posts/learning-rate/

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            
            decay_factor = warmup_steps ** 0.5
            return decay_factor * step ** -0.5

        super(InverseSqrtScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

class CMLMPL(pl.LightningModule):

    def __init__(
        self,
        active_config,
        config,
        device,
        tokenizer: PreTrainedTokenizer,
        fairseq_dict: Dictionary
    ):
        super().__init__()
        self.save_hyperparameters()
        self.active_config = active_config
        self.config = config

        fairseq_args = { # FIX
            "data": "mask_predict/ML50_dict",
            "lang_tokens_file": "mask_predict/ML50_dict/ML50_langs.txt",
            "source_lang": self.active_config['src'],
            "target_lang": self.active_config['trg'],
            "shared_dict": self.config['dict'],
            "arch": "bert_transformer_seq2seq",
            "share_all_embeddings": True,
            #"criterion": "label_smoothed_length_cross_entropy",
            #"label_smoothing": 0.1,
            #"lr": 5e-4,
            #"warmup-init-lr": 1e-7,
            #"min-lr": 1e-9,
            #"lr-scheduler": "inverse_sqrt",
            #"warmup-updates": 10000,
            #"optimizer": "adam",
            #"adam-betas": '(0.9, 0.999)',
            #"adam-eps": 1e-6, 
            #"task": "translation_self",
            "max_tokens": 800,
            #"weight-decay": 0.01,
            "dropout": 0.3, 
            "encoder_layers": 6, 
            "encoder_embed-dim": 512, 
            "decoder_layers": 6, 
            "decoder_embed-dim": 512,  
            #"fp16": True, 
            "max_source_positions": 50,
            "max_target_positions": 50,
            "seed": self.config['seed'],
            "left_pad_source": False,
            "left_pad_target": False
        }

        fairseq_args = argparse.Namespace(**fairseq_args)

        utils.import_user_module(fairseq_args)

        task = TranslationSelfTask.setup_shared_dict(fairseq_args, fairseq_dict)
        
        self.model = Transformer_nonautoregressive.build_model(fairseq_args, task)

        generation_args = {
            "decoding_strategy": "mask_predict",
            "remove_bpe": True, 
            "max_sentences": 20,
            "decoding_iterations": 10
        }
        
        generation_args = argparse.Namespace(**generation_args)

        self.strategy = MaskPredict(generation_args)
        #strategies.setup_strategy(generation_args)

        self.tokenizer = tokenizer
        self.scorer = Scorer(self.active_config, self.config, self.tokenizer)
        self.shared_dict = task.source_dictionary
        self.sup_criterion = nn.CrossEntropyLoss(ignore_index = self.tokenizer.pad_token_id) #self.shared_dict.pad()
        self.dynamic_length = False
        self.random = np.random.RandomState(self.config['seed'])
        self.mask_range = False

    def preprocess_dataset_batch(self, batch, train: bool, unlabeled: bool):
        '''
        change dataset batch (dict("input_ids", "attention_mask", "labels")) into
        (dict("src_tokens", "src_lengths", "prev_output_tokens", "labels"))
        '''
        batch["src_tokens"] = batch["input_ids"]
        batch.pop("input_ids")
        
        batch.pop("attention_mask")

        # get src lengths
        nonpad_mask = (batch["src_tokens"] != self.tokenizer.pad_token_id) #self.shared_dict.pad
        col_indices = torch.arange(batch["src_tokens"].shape[1], device=nonpad_mask.device)
        max_nonpad_indices = torch.argmax(nonpad_mask * col_indices, dim=1)
        batch["src_lengths"] = max_nonpad_indices.add(1)
        
        if train and not unlabeled:
            batch["prev_output_tokens"] = [] 
            for source, target in zip(batch["src_tokens"], batch["labels"]):
                batch["prev_output_tokens"].append(self.mask(source, target)[1])
            batch["prev_output_tokens"] = torch.stack(batch["prev_output_tokens"])

        return batch


    def forward(self, **inputs):
        """
        inputs: dict("src_tokens", "src_lengths", "prev_output_tokens")
        """

        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        print("CMLM validation step")

        batch = self.preprocess_dataset_batch(batch, train=False, unlabeled=False)
        labels = batch.pop("labels")

        pred_labels, lprobs, all_token_probs, cross_attention = generate(self.active_config, self.strategy, batch, [self.model], self.shared_dict, 1, None)

        # prepare attention (last layer of decoder)
        cross_attention = torch.unsqueeze(cross_attention[:, 1:, 1:],dim=1) # (batch_size, target seq length, input seq length)
        
        # TODO: how to define loss?
        # calculate loss, teacher forcing with real labels
        batch["prev_output_tokens"] = [] 
        for source, target in zip(batch["src_tokens"], labels):
            batch["prev_output_tokens"].append(self.mask(source, target)[1])
        batch["prev_output_tokens"] = torch.stack(batch["prev_output_tokens"])
        teacher_forced_outputs, _ = self.model(**batch)

        batch_size = teacher_forced_outputs.shape[0]
        output_dim = teacher_forced_outputs.shape[-1]

        teacher_forced_outputs = teacher_forced_outputs.contiguous().view(-1, output_dim) # [batch size * trg len, output_dim]
        labels = labels.contiguous().view(-1) # [batch size * trg len]

        val_loss = self.sup_criterion(teacher_forced_outputs, labels)
        self.log('val_loss', val_loss.item(), sync_dist=True)

        labels = labels.view(batch_size,-1)
        # pred_labels: [trg lang code] X [eos] [pad] 
        # labels: [trg lang code] X [eos] [pad] 
        return {"loss": val_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention}
    

    def test_step(self, batch, batch_idx):
        batch = self.preprocess_dataset_batch(batch, train=False, unlabeled=False)
        labels = batch.pop("labels")

        pred_labels, lprobs, all_token_probs, cross_attention = generate(self.active_config, self.strategy, batch, [self.model], self.shared_dict, 1, None)

        pred_labels = pred_labels
        # prepare attention (last layer of decoder)
        cross_attention = torch.unsqueeze(cross_attention[:, 1:, 1:],dim=1) # (batch_size, 1, target seq length, input seq length)
        
        # TODO: how to define loss?
        # calculate loss, teacher forcing with real labels
        batch["prev_output_tokens"] = [] 
        for source, target in zip(batch["src_tokens"], labels):
            batch["prev_output_tokens"].append(self.mask(source, target)[1])
        batch["prev_output_tokens"] = torch.stack(batch["prev_output_tokens"])

        teacher_forced_outputs, _ = self.model(**batch)

        batch_size = teacher_forced_outputs.shape[0]
        output_dim = teacher_forced_outputs.shape[-1]

        teacher_forced_outputs = teacher_forced_outputs.contiguous().view(-1, output_dim) # [batch size * trg len, output_dim]
        labels = labels.contiguous().view(-1) # [batch size * trg len]

        test_loss = self.sup_criterion(teacher_forced_outputs, labels)
        self.log('test_loss', test_loss.item(), sync_dist=True)

        # pred_labels: [trg lang code] X [eos] [pad] 
        # labels: [trg lang code] X [eos] [pad] 
        labels = labels.view(batch_size,-1)
        return {"loss": test_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention}
    
        
    def configure_optimizers(self):
        optimizer = Adam(self.trainer.model.parameters(),
                         eps=1e-06,
                         betas=(0.9, 0.98),
                         lr=1e-6)
        #scheduler = InverseSqrtScheduler(optimizer, 500)

        #return {
        #    "optimizer": optimizer,
        #    "lr_scheduler": {
		#        "scheduler": scheduler,
		#        "interval": "step",
	    #    }
        #}

        return optimizer

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

    def mask(self, source, target):
        if self.dynamic_length:
            max_len = 3 * len(source) // 2 + 1
            target = target.new((target.tolist() + ([self.tokenizer.eos_token_id] * (max_len - len(target))))[:max_len]) #self.shared_dict.eos()
        
        min_num_masks = 1
        
        enc_source = source
        dec_source = target.new(target.tolist())
        dec_target_cp = target.new(target.tolist())
        dec_target = target.new([self.tokenizer.pad_token_id] * len(dec_source)) #self.shared_dict.pad()
        
        if self.trainer.training:
            if min_num_masks < len(dec_source):
                sample_size = self.random.randint(min_num_masks, len(dec_source))
            else:
                sample_size = len(dec_source)

            if self.mask_range:
                start = self.random.randint(len(dec_source) - sample_size + 1)
                ind = list(range(start, start + sample_size))
            else:
                ind = self.random.choice(len(dec_source) , size=sample_size, replace=False)
            
            dec_source[ind] = self.tokenizer.mask_token_id # self.shared_dict.mask()
            dec_target[ind] = dec_target_cp[ind]
        else:
            dec_target = dec_target_cp
            dec_source[:] = self.tokenizer.mask_token_id # self.shared_dict.mask()

        ntokens = dec_target.ne(self.tokenizer.pad_token_id).sum(-1).item() # self.shared_dict.pad()
        #print ("masked tokens", self.tgt_dict.string(dec_source))
        #print ("original tokens", self.tgt_dict.string(dec_target))
        #print ("source tokens", self.src_dict.string(enc_source))

        return enc_source, dec_source, dec_target, ntokens

class CMLMSupPL(CMLMPL):
    def training_step(self, batch, batch_idx):

        print("CMLM training step")
        
        batch = self.preprocess_dataset_batch(batch, train=True, unlabeled=False)

        print(batch)

        labels = batch.pop("labels")
        
        outputs, _ = self(**batch)

        batch_size = outputs.shape[0]
        output_dim = outputs.shape[-1]

        outputs = outputs.contiguous().view(-1, output_dim) # [batch size * trg len, output_dim]
        labels = labels.contiguous().view(-1) # [batch size * trg len]

        loss = self.sup_criterion(outputs, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss


class CMLMSslPL(CMLMPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer, fairseq_dict: Dictionary):
        super().__init__(active_config, config, device, tokenizer)
        self.epoch_scaled_scores = {}
        self.epoch_raw_scores = {}
        self.tokenizer = tokenizer

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
            if encoder_attention:
                encoder_attention = encoder_attention[:, :, 1:, 1:]
            if decoder_attention:
                decoder_attention = decoder_attention[:, :, 1:, 1:]

            # set loss function, ignore pad
            sup_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index = self.tokenizer.pad_token_id) #self.shared_dict.pad()

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
            #good_samples = (batch_raw_scores['final'] > np.quantile(raw_score_record['final'], 0.5)) # indices of samples to keep
            #print("good_samples: ", torch.sum(good_samples).item())

            # filter
            #batch_scaled_scores['final'] = batch_scaled_scores['final'][good_samples]
            #batch_raw_scores['final'] = batch_raw_scores['final'][good_samples]
            #batch_sup_loss = batch_sup_loss[good_samples]
                
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


    def training_step(self, batch, batch_idx):

        sup_inputs = self.preprocess_dataset_batch(batch['label'], train=True, unlabeled=False)
        sup_labels = sup_inputs.pop("labels")
        unsup_inputs = self.preprocess_dataset_batch(batch['unlabel'], train=True, unlabeled=True)

        # forward pass through model - sup, unsup separately

        # sup inputs
        sup_outputs, _ = self.model(**sup_inputs)

        batch_size = sup_outputs.shape[0]
        output_dim = sup_outputs.shape[-1]

        sup_outputs = sup_outputs.contiguous().view(-1, output_dim) # [batch size * trg len, output_dim]
        sup_labels = sup_labels.contiguous().view(-1) # [batch size * trg len]

        loss_sup = self.sup_criterion(sup_outputs, sup_labels)
        
        # [batch_size,1] filled with decoder_start_token_id
        decoder_start_tokens = torch.ones_like(unsup_inputs["src_tokens"])[:, :1] * self.tokenizer.eos_token_id #self.shared_dict.eos()
        # [batch_size,1] filled with target lang id
        #decoder_forced_start_tokens = torch.ones_like(unsup_inputs["src_tokens"])[:, :1] * self.tokenizer.lang_code_to_id[self.active_config['trg_code']]
        #decoder_start_tokens = torch.cat((decoder_start_tokens, decoder_forced_start_tokens), dim=1)

        #decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)
        
        unsup_output_ids, unsup_lprobs, _, _ = generate(self.active_config, self.strategy, unsup_inputs, [self.model], self.shared_dict, 1, None)

        unsup_labels = unsup_output_ids[:, 1:].clone()
        # leave out <eos> at the beginning -> [trg_lang_code] Y [eos] [pad] * N

        sup_outputs = sup_outputs.view(batch_size,-1,output_dim)
        sup_labels = sup_labels.view(batch_size,-1)

        #if batch_idx % 64 == 0:
            #print("sup_inputs: ", vars(sup_inputs))
            #sup_src = self.tokenizer.batch_decode(sup_inputs["src_tokens"])
            #print("sup input src: ", sup_src)

            #original_labels = torch.where(sup_labels == -100, self.shared_dict.pad(), sup_labels)
            #sup_labels_str = self.tokenizer.batch_decode(original_labels)
            #print("sup input labels:  ", sup_labels_str)

            #sup_prediction = self.tokenizer.batch_decode(torch.argmax(sup_outputs,-1))
            #print("sup input prediction: ", sup_prediction)

            #unsup_src = self.tokenizer.batch_decode(unsup_inputs["src_tokens"])
            #print("unsup input src: ", unsup_src)

            #unsup_prediction_raw = self.tokenizer.batch_decode(unsup_output_ids)
            #print("unsup input prediction raw: (before removing first token): \n", unsup_prediction_raw)

            #print("unsup_outputs:", vars(unsup_outputs))
            #unsup_prediction = self.tokenizer.batch_decode(unsup_labels)
            #print("unsup input prediction: ", unsup_prediction)

        # forward pass, teacher forcing with unsup_labels
        
        unsup_inputs["prev_output_tokens"] = []
        for source, target in zip(unsup_inputs["src_tokens"], unsup_labels):
            unsup_inputs["prev_output_tokens"].append(self.mask(source, target)[1])
        unsup_inputs["prev_output_tokens"] = torch.stack(unsup_inputs["prev_output_tokens"])

        unsup_outputs = self.model(**unsup_inputs)
        unsup_logits = unsup_outputs[0]
        
        # prepare attention (last layer of decoder)
        unsup_attention = unsup_outputs[1]['attn'] # (batch_size, target seq length, source seq length)
        if unsup_attention is not None:
            print("unsup attention not None")
            unsup_attention = torch.unsqueeze(unsup_attention, dim=1) # (batch_size, 1, target seq length, source seq length)

        loss_unsup, scaled_scores, raw_scores= self.unsup_criterion(
            unsup_inputs["src_tokens"], unsup_logits, unsup_labels, unsup_attention,
            None, None,
            self.epoch_scaled_scores, self.epoch_raw_scores,
            batch_idx % 64 == 0)
       
        for score_key in scaled_scores.keys(): # including 'final'
            if score_key not in self.epoch_scaled_scores.keys():
                self.epoch_scaled_scores[score_key] = []
                self.epoch_raw_scores[score_key] = []
            self.epoch_scaled_scores[score_key].extend(scaled_scores[score_key].tolist())
            self.epoch_raw_scores[score_key].extend(raw_scores[score_key].tolist())
        
        self.log("scores mean", scaled_scores['final'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) # batch mean score
        
        # total loss
        loss = loss_sup + self.config['unsup_wt'] * loss_unsup 
        self.log("train_loss_sup", loss_sup.item(), sync_dist=True)
        self.log("train_loss_unsup", loss_unsup.item(), sync_dist=True)
        self.log("train_loss", loss.item(), sync_dist=True)
        return loss