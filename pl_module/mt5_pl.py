import numpy as np

import torch
from torch import nn
from torch.optim.adam import Adam

import pytorch_lightning as pl

from transformers import PreTrainedTokenizer, MT5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from score.score import Scorer

# TODO
class MT5PL(pl.LightningModule):
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
        self.model = MT5ForConditionalGeneration.from_pretrained(self.active_config["model_name_or_path"])
        #self.model._init_weights = True
        #self.model.init_weights()
        self.tokenizer = tokenizer
        self.scorer = Scorer(self.active_config, self.config, self.tokenizer)
    
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
        batch.pop("decoder_input_ids")

        if not self.config['eval_teacher_forcing']:
            # [batch_size,1] filled with decoder_start_token_id
            decoder_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.model.config.decoder_start_token_id
            # [batch_size,1] filled with target lang id
            #decoder_forced_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.tokenizer.lang_code_to_id[self.active_config['trg_code']]
            #decoder_start_tokens = torch.cat((decoder_start_tokens, decoder_forced_start_tokens), dim=1)

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
        batch.pop("decoder_input_ids")

        if not self.config['eval_teacher_forcing']:
            # [batch_size,1] filled with decoder_start_token_id
            decoder_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.model.config.decoder_start_token_id
            # [batch_size,1] filled with target lang id
            #decoder_forced_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.tokenizer.lang_code_to_id[self.active_config['trg_code']]
            #decoder_start_tokens = torch.cat((decoder_start_tokens, decoder_forced_start_tokens), dim=1)

            decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)

            # generated translation
            pred_ids = self.model.generate(**batch,
                            num_beams=5,
                            num_return_sequences=1,
                            max_new_tokens = self.config['max_length'],
                            decoder_input_ids=decoder_start_tokens, decoder_attention_mask=decoder_attention_mask, 
                            use_cache=False, synced_gpus=True)
        
            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning
        
            # forward pass, teacher forcing with generated translation
            beam_outputs: Seq2SeqLMOutput = self.model(**batch, labels=pred_labels, output_attentions=True)
            # prepare attention (last layer of decoder)
            cross_attention = beam_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # (batch_size, num_heads, target seq length, input seq length)
            encoder_attention = beam_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = beam_outputs.decoder_attentions[-1][:, :, 1:, 1:]

        # calculate loss, teacher forcing with real output
        teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
        if self.config["eval_teacher_forcing"]:
            pred_labels = torch.argmax(teacher_forced_outputs.logits, dim=-1)
            cross_attention = teacher_forced_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:]
            encoder_attention = teacher_forced_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = teacher_forced_outputs.decoder_attentions[-1][:, :, 1:, 1:]
            
        test_loss = teacher_forced_outputs.loss
        
        self.log('test_loss', test_loss.item(), sync_dist=True) # loss is unsupervised loss (targets are beam search outputs)

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
    

    
    
        
    


class MT5SupPL(MT5PL):
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, sync_dist=True)
        return loss


class MT5SslPL(MT5PL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer):
        super().__init__(active_config, config, device, tokenizer)
        self.epoch_scaled_scores = {}
        self.epoch_raw_scores = {}
        self.tokenizer = tokenizer

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
        #decoder_forced_start_tokens = torch.ones_like(unsup_inputs.input_ids)[:, :1] * self.tokenizer.lang_code_to_id[self.active_config['trg_code']]
        #decoder_start_tokens = torch.cat((decoder_start_tokens, decoder_forced_start_tokens), dim=1)

        decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)
        
        # generate, only get the output ids
        unsup_output_ids: torch.LongTensor = self.model.generate(**unsup_inputs, 
                                            do_sample=do_sample, num_beams=num_beams,
                                            num_return_sequences=1,
                                            max_new_tokens = self.config['max_length'],
                                            decoder_input_ids=decoder_start_tokens, decoder_attention_mask=decoder_attention_mask, 
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
        unsup_wt = self.config['unsup_wt']
        loss = loss_sup + unsup_wt * loss_unsup 
        self.log("train_loss_sup", loss_sup.item(), sync_dist=True)
        self.log("train_loss_unsup", loss_unsup.item(), sync_dist=True)
        self.log("train_loss", loss.item(), sync_dist=True)
        return loss