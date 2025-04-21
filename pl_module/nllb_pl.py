import numpy as np
import scipy

import torch
from torch import nn
from torch.optim.adam import Adam
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from peft import LoraConfig, TaskType, get_peft_model

from score.score import Scorer

class NLLBPL(pl.LightningModule):

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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.active_config["model_name_or_path"])
        # "facebook/nllb-200-distilled-600M"
        
        # lora
        peft_config = LoraConfig(
            target_modules=['q_proj','v_proj'], task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, 
            r=32, lora_alpha=32, lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, peft_config)

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
        

    def validation_step(self, batch, batch_idx):
        
        labels = batch["labels"]
        batch.pop("labels")
        #batch.pop("decoder_input_ids")

        if not self.config['eval_teacher_forcing']:
            # [batch_size,1] filled with decoder_start_token_id
            decoder_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.model.config.decoder_start_token_id
            # [batch_size,1] filled with target lang id
            decoder_forced_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']]
            decoder_start_tokens = torch.cat((decoder_start_tokens, decoder_forced_start_tokens), dim=1)

            decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)

            # generated translation
            if self.config['ranking']:
                num_hypotheses = 5
            else:
                num_hypotheses = 1

            pred_ids = self.model.generate(**batch,
                            num_beams=5,
                            num_return_sequences=num_hypotheses,
                            max_new_tokens = self.config['max_length'],
                            decoder_input_ids=decoder_start_tokens, decoder_attention_mask=decoder_attention_mask, 
                            use_cache=False, synced_gpus=True)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning

            print(pred_labels.shape)
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat(num_hypotheses, 1)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat(num_hypotheses, 1)
            print(repeated_batch['input_ids'].shape)
            print(repeated_batch['attention_mask'].shape)

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
        #batch.pop("decoder_input_ids")

        if not self.config['eval_teacher_forcing']:
            # [batch_size,1] filled with decoder_start_token_id
            decoder_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.model.config.decoder_start_token_id
            # [batch_size,1] filled with target lang id
            decoder_forced_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']]
            decoder_start_tokens = torch.cat((decoder_start_tokens, decoder_forced_start_tokens), dim=1)

            decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)

            # generated translation
            if self.config['ranking']:
                num_hypotheses = 5
            else:
                num_hypotheses = 1

            print("batch input_ids.device: ", batch["input_ids"].device)
            print("decoder_start_tokens.device: ", decoder_start_tokens.device)
            print("decoder_attention_mask.device: ", decoder_attention_mask.device)

            pred_ids = self.model.generate(**batch,
                            num_beams=5,
                            num_return_sequences=num_hypotheses,
                            max_new_tokens = self.config['max_length'],
                            decoder_input_ids=decoder_start_tokens, decoder_attention_mask=decoder_attention_mask, 
                            use_cache=False, synced_gpus=True)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning

            print(pred_labels.shape)
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat(num_hypotheses, 1)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat(num_hypotheses, 1)
            print(repeated_batch['input_ids'].shape)
            print(repeated_batch['attention_mask'].shape)

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
                         lr=3e-04)
        #scheduler = InverseSqrtScheduler(optimizer, 500)

        #return {
        #    "optimizer": optimizer,
        #    "lr_scheduler": {
		#        "scheduler": scheduler,
		#        "interval": "step",
	    #    }
        #}

        return optimizer
    
class NLLBSupPL(NLLBPL):
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, sync_dist=True)
        return loss


class NLLBSslPL(NLLBPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer):
        super().__init__(active_config, config, device, tokenizer)
        self.epoch_raw_scores = {}
        self.epoch_scaled_scores = {}
        self.tokenizer = tokenizer
        self.unsup_weight = 0.0 if self.config['weight_schedule'] else self.config['unsup_wt']

        def unsup_criterion(src, outputs, labels, cross_attention, encoder_attention, decoder_attention, scaled_score_record, raw_score_record, verbose):
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
                
            batch_scaled_scores['final'] = batch_scaled_scores['final'].to(batch_sup_loss)
            batch_raw_scores['final'] = batch_raw_scores['final'].to(batch_sup_loss)

            assert batch_scaled_scores['final'].requires_grad == False
            assert batch_raw_scores['final'].requires_grad == False
            assert batch_sup_loss.requires_grad == True
            
            print("batch unsup loss (cross entropy): ", batch_sup_loss)

            loss = batch_scaled_scores['final'].dot(batch_sup_loss)/batch_size

            print("batch unsup loss (cross entropy * scores): ", loss)

            assert loss.requires_grad == True
                
            return loss, batch_scaled_scores, batch_raw_scores

        self.unsup_criterion = unsup_criterion


    def training_step(self, batch, batch_idx):

        # decay unsup weight (increase with steps)
        if self.config['weight_schedule']:
            self.unsup_weight = (self.trainer.global_step / 1000) * self.config['unsup_wt']
        else:
            self.unsup_weight = self.config['unsup_wt']

        sup_inputs = batch['label'] # TODO maybe change 'label' to 'sup' or 'parallel'?
        unsup_inputs = batch['unlabel']

        # forward pass through model - sup, unsup separately

        # sup inputs
        sup_outputs: Seq2SeqLMOutput = self.model(**sup_inputs)
        loss_sup, sup_logits = sup_outputs.loss, sup_outputs.logits
        output_dim = sup_logits.shape[-1]

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
        
        # [batch_size,1] filled with decoder_start_token_id
        decoder_start_tokens = torch.ones_like(unsup_inputs.input_ids)[:, :1] * self.model.config.decoder_start_token_id
        # [batch_size,1] filled with target lang id
        decoder_forced_start_tokens = torch.ones_like(unsup_inputs.input_ids)[:, :1] * self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']]
        decoder_start_tokens = torch.cat((decoder_start_tokens, decoder_forced_start_tokens), dim=1)

        #decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)
        
        # generate, only get the output ids
        unsup_output_ids: torch.LongTensor = self.model.generate(**unsup_inputs, 
                                            do_sample=do_sample, num_beams=num_beams,
                                            num_return_sequences=1,
                                            max_new_tokens = self.config['max_length'],
                                            decoder_input_ids=decoder_start_tokens, #decoder_attention_mask=decoder_attention_mask, 
                                            use_cache=False, synced_gpus=True, early_stopping=True, 
                                            top_p=top_p, top_k=top_k)

        unsup_labels = unsup_output_ids[:, 1:].clone()
        # leave out <eos> at the beginning -> [trg_lang_code] Y [eos] [pad] * N

        if batch_idx % 64 == 0:
            #print("sup_inputs: ", vars(sup_inputs))
            sup_src = self.tokenizer.batch_decode(sup_inputs.input_ids)
            print("sup input src: ", sup_src)

            original_labels = torch.where(sup_inputs.labels == -100, self.model.config.pad_token_id, sup_inputs.labels)
            sup_labels = self.tokenizer.batch_decode(original_labels)
            print("sup input labels:  ", sup_labels)

            sup_prediction = self.tokenizer.batch_decode(torch.argmax(sup_outputs.logits,-1))
            print("sup input prediction: ", sup_prediction)

            unsup_src = self.tokenizer.batch_decode(unsup_inputs.input_ids)
            print("unsup input src: ", unsup_src)

            unsup_prediction_raw = self.tokenizer.batch_decode(unsup_output_ids)
            print("unsup input prediction raw: (before removing first token): \n", unsup_prediction_raw)

            #print("unsup_outputs:", vars(unsup_outputs))
            unsup_prediction = self.tokenizer.batch_decode(unsup_labels)
            print("unsup input prediction: ", unsup_prediction)

        # forward pass, teacher forcing with output ids as the labels 
        unsup_outputs: Seq2SeqLMOutput = self.model(**unsup_inputs, labels=unsup_labels, output_attentions=True)
        unsup_logits = unsup_outputs.logits
        
        # prepare attention (last layer of decoder)
        unsup_cross_attention = unsup_outputs.cross_attentions[self.config['cross_attention_layer']] # (batch_size, num_heads, target seq length, input seq length)
        unsup_encoder_attention = unsup_outputs.encoder_attentions[-1]
        unsup_decoder_attention = unsup_outputs.decoder_attentions[-1]

        loss_unsup, scaled_scores, raw_scores= self.unsup_criterion(
            unsup_inputs.input_ids, unsup_logits, unsup_labels, 
            unsup_cross_attention, unsup_encoder_attention, unsup_decoder_attention,
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
        #if loss_sup < 5.5:
        #    unsup_wt = self.config['unsup_wt'] * 10
        loss = max(0, 1 - 10 * self.unsup_weight) * loss_sup + self.unsup_weight * loss_unsup 
        self.log("train_loss_sup", loss_sup.item(), sync_dist=True)
        self.log("train_loss_unsup", loss_unsup.item(), sync_dist=True)
        self.log("train_loss", loss.item(), sync_dist=True)
        return loss

class NLLBPpoPL(NLLBPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer):
        super().__init__(active_config, config, device, tokenizer)
       
        self.tokenizer = tokenizer
        #self.critic_model = AutoModelForSeq2SeqLM.from_pretrained(self.active_config["model_name_or_path"])
        self.critic_model_head = nn.Linear(
            self.model.config.hidden_size, 1, bias=False
            #self.critic_model.config.hidden_size, 1, bias=False
        )
        self.gamma = 1.0
        self.gae_lambda = 0.95
        self.beta = 0.5 # value function loss coef
        self.ent_coef = 0.1
        self.unsup_weight = 0.0 if self.config['weight_schedule'] else self.config['unsup_wt']

        self.epoch_raw_scores = {}
        self.epoch_scaled_scores = {}

        self.clip_range = 0.2

        self.reset_buffers(self.config['batch_size'], self.config['max_length'])

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

    def reset_buffers(self, batch_size, seq_len):
        self.epoch_values = [0.0] * (batch_size * (seq_len+1)) # length: batch_size * seq length
        self.epoch_advantages = [0.0] * (batch_size * (seq_len+1)) # length: batch_size * seq length
        self.epoch_log_prob = [0.0] * (batch_size * (seq_len+1)) # length: batch_size * seq length
        self.epoch_returns = [0.0] * (batch_size * (seq_len+1)) # length: batch_size * seq length

    @torch.autocast("cuda")
    def training_step(self, batch, batch_idx):

        # decay unsup weight (increase with steps)
        if self.config['weight_schedule']:
            self.unsup_weight = (self.trainer.global_step / 1000) * self.config['unsup_wt']
        else:
            self.unsup_weight = self.config['unsup_wt']

        # TODO: mask padding

        sup_inputs = batch['label'] 
        unsup_inputs = batch['unlabel']
        seq_len = unsup_inputs.input_ids.size(1)
        unsup_batch_size = unsup_inputs.input_ids.size(0)

        print("batch_idx: ", batch_idx)
        print("seq_len: ", seq_len)
        print("unsup_batch_size: ", unsup_batch_size)

        start_sample_idx = batch_idx * self.config['batch_size']
        end_sample_idx = batch_idx * self.config['batch_size'] + unsup_batch_size
        batch_slice = slice(start_sample_idx, end_sample_idx)
        print("batch_slice: ", batch_slice)

        rollout_batch_slice = slice(start_sample_idx * (self.config['max_length']+1), end_sample_idx * (self.config['max_length']+1))
        print("rollout_batch_slice: ", rollout_batch_slice)

        # forward pass through model - sup, unsup separately

        # sup inputs
        sup_outputs: Seq2SeqLMOutput = self.model(**sup_inputs)
        loss_sup, sup_logits = sup_outputs.loss, sup_outputs.logits
        output_dim = sup_logits.shape[-1]

        # set generation parameters for unsup output generation
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

        # apply actor critic (PPO) algorithm to unsup_inputs
        
        # 1. generate rollouts (sequences)

        # [batch_size,1] filled with decoder_start_token_id
        decoder_start_tokens = torch.ones_like(unsup_inputs.input_ids)[:, :1] * self.model.config.decoder_start_token_id
        # [batch_size,1] filled with target lang id
        decoder_forced_start_tokens = torch.ones_like(unsup_inputs.input_ids)[:, :1] * self.tokenizer.lang_code_to_id[self.active_config['model_trg_code']]
        decoder_start_tokens = torch.cat((decoder_start_tokens, decoder_forced_start_tokens), dim=1)

        #decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)
        
        # generate, only get the output ids
        unsup_output_ids: torch.LongTensor = self.model.generate(**unsup_inputs, 
                                            do_sample=do_sample, num_beams=num_beams,
                                            num_return_sequences=1,
                                            max_new_tokens = self.config['max_length'],
                                            decoder_input_ids=decoder_start_tokens, #decoder_attention_mask=decoder_attention_mask, 
                                            use_cache=False, synced_gpus=True, early_stopping=True, 
                                            top_p=top_p, top_k=top_k)
        # unsup_output_ids max length: 2 + 50
        unsup_labels = unsup_output_ids[:, 1:].clone() # 51
        # leave out <eos> at the beginning -> [trg_lang_code] Y [eos] [pad] * N
        unsup_labels = pad_to_fixed_length(unsup_labels, self.config['max_length']+1, self.tokenizer.pad_token_id)
        
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

        # 2. get policy loss
        # forward pass through policy network, teacher forcing with output ids as the labels 
        unsup_policy_outputs: Seq2SeqLMOutput = self.model(**unsup_inputs, labels=unsup_labels, output_attentions=True)
        unsup_policy_logits = unsup_policy_outputs.logits # [batch_size, seq_length, vocab_size]
        unsup_log_prob = get_token_probs_from_logits(unsup_policy_logits, unsup_labels) #  [batch_size, seq_length]
        # TODO: just use compute transition scores

        # forward pass through value network
        unsup_value_outputs: Seq2SeqLMOutput = self.model(**unsup_inputs, labels=unsup_labels, output_attentions=True, output_hidden_states=True)
        #self.critic_model(**unsup_inputs, labels=unsup_labels, output_attentions=True)
        last_hidden_state = unsup_value_outputs.decoder_hidden_states[-1] # [batch_size, seq_length, hidden_size]
        unsup_values = self.critic_model_head.forward(last_hidden_state).squeeze() # [batch size, seq_length]
        
        # prepare attention (last layer of decoder)
        unsup_cross_attention = unsup_policy_outputs.cross_attentions[self.config['cross_attention_layer']] # (batch_size, num_heads, target seq length, input seq length)
        unsup_encoder_attention = unsup_policy_outputs.encoder_attentions[-1]
        unsup_decoder_attention = unsup_policy_outputs.decoder_attentions[-1]

        # get scores
        _, scaled_scores, raw_scores= self.unsup_criterion(
            unsup_inputs.input_ids, unsup_policy_logits, unsup_labels, 
            unsup_cross_attention, unsup_encoder_attention, unsup_decoder_attention,
            self.epoch_scaled_scores, self.epoch_raw_scores,
            batch_idx % 64 == 0)
        
        for score_key in scaled_scores.keys(): # including 'final'
            if score_key not in self.epoch_scaled_scores.keys():
                self.epoch_scaled_scores[score_key] = []
                self.epoch_raw_scores[score_key] = []
            self.epoch_scaled_scores[score_key].extend(scaled_scores[score_key].tolist())
            self.epoch_raw_scores[score_key].extend(raw_scores[score_key].tolist())

        self.log("scores mean", scaled_scores['final'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) # batch mean score
        
        
        # calculate advantages
        # TODO: batch calculation
        batch_returns = []
        batch_advantages = []
        for i in range(unsup_batch_size):
            vals = unsup_values[i].squeeze().clone().cpu().detach().numpy()
            reward = scaled_scores['final'][i].clone().cpu().detach().numpy()
            _returns, _advantages = self.compute_returns_and_advantages(0, reward, vals) # TODO: fix 0
            batch_returns.extend(_returns)
            batch_advantages.extend(_advantages)

        batch_returns = torch.tensor(batch_returns, device=unsup_labels.device).float() # [batch_size * seq length]
        batch_advantages = torch.tensor(batch_advantages, device=unsup_labels.device).float() # [batch_size * seq length]
        
        # normalize advantages
        print("normalize advantages")
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

        # ratio between policies (token level)
        print("get_policy_ratio")
        if self.trainer.current_epoch == 0:
            policy_ratio = torch.ones(unsup_batch_size * (self.config['max_length']+1), device=unsup_log_prob.device)
        else:
            old_log_prob = self.epoch_log_prob[rollout_batch_slice] # [batch_size * sequence length]
            diff = unsup_log_prob.flatten() - torch.tensor(old_log_prob, device=unsup_log_prob.device)
            policy_ratio = torch.exp(torch.min(diff, torch.tensor(8, device=diff.device)))
            
        # get policy loss
        print("get policy loss")
        self.log("policy_ratio", policy_ratio.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        policy_loss_1 = batch_advantages * policy_ratio 
        self.log("policy_loss_1", policy_loss_1.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        policy_loss_2 = batch_advantages * torch.clamp(policy_ratio, 1 - self.clip_range, 1 + self.clip_range)
        self.log("policy_loss_2", policy_loss_2.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # get entropy loss
        # higher entropy (or lower entropy_loss) means more exploration
        entropy_loss = -torch.mean(-unsup_log_prob)

        # get value loss
        print("get value loss")
        value_loss = F.mse_loss(batch_returns, unsup_values.flatten())
        print("batch_returns: ", batch_returns.dtype)

        print("get total loss")
        self.log("policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("entropy_loss", entropy_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("value_loss", value_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss_unsup = policy_loss + self.ent_coef * entropy_loss + self.beta * value_loss

        # total loss
        #if loss_sup < 5.5:
        #    unsup_wt = self.config['unsup_wt'] * 10
        
        loss = max(0, 1 - 10 * self.unsup_weight) * loss_sup + self.unsup_weight * loss_unsup 
        self.log("train_loss_sup", loss_sup.item(), sync_dist=True)
        self.log("train_loss_unsup", loss_unsup.item(), sync_dist=True)
        self.log("train_loss", loss.item(), sync_dist=True)

        # update buffers
        self.epoch_advantages.extend(batch_advantages.tolist())
        self.epoch_returns.extend(batch_returns.tolist())
        self.epoch_log_prob.extend(unsup_log_prob.flatten().tolist())
        self.epoch_values.extend(unsup_values.flatten().tolist())

        return loss

    def compute_returns_and_advantages(self, last_value, reward, values):
        '''
        Generalized Advantage Estimation, code borrowed from OpenAI SpinningUp
        for one sequence

        last_value: scalar 
        reward: scalar
        values: [seq_length]
        '''
        # TODO batching for faster calculation
        rews = np.append([reward] * len(values), last_value)
        vals = np.append(values, last_value)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        advantages = discount_cumsum(deltas, self.gamma * self.gae_lambda)
        
        # the next line computes rewards-to-go, to be targets for the value function
        returns = discount_cumsum(rews, self.gamma)[:-1] 

        return advantages, returns


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def get_token_probs_from_logits(logits: torch.Tensor, token_ids: torch.Tensor):
    # logits: [batch_size, seq_length, vocab_size]
    _logits = logits.clone().reshape(logits.size(0) * logits.size(1),-1)
    _ids = token_ids.clone().reshape(-1).unsqueeze(-1)
    probs = _logits.gather(-1, _ids)
    return probs.reshape(logits.size(0),-1) # [batch_size, seq_length]

def pad_to_fixed_length(matrix: torch.Tensor, fixed_length, pad_token_id):
    print("pad to fixed length")
    print("seq length", matrix.size(1))
    ones_tensor = torch.ones(matrix.size(0), fixed_length - matrix.size(1), device=matrix.device).int() * pad_token_id
    padded = torch.concat((matrix, ones_tensor), dim=1)
    return padded
