import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from model.custom_mbart import NAMBartForConditionalGeneration
from score.score import Scorer
from generation_strategy.mask_predict import MaskPredict, _mean_pooling, predict_length_beam, duplicate_encoder_out


class NAMBARTPL(pl.LightningModule):

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
        self.model = NAMBartForConditionalGeneration.from_pretrained(self.active_config["model_name_or_path"])
        
        # freeze encoder
        for param in self.model.get_encoder().parameters():
            param.requires_grad = False

        # initialize decoder weights
        #self.model.get_decoder().init_weights()        

        self.tokenizer = tokenizer
        self.scorer = Scorer(self.active_config, self.config, self.tokenizer)

        #self.shared_dict = Dictionary.load("dict_250k.txt")
        # add special lang tokens
        #with open('ML50_langs.txt') as f:
        #    lang_tokens = f.readlines()
        #for lang_token in lang_tokens:
        #    self.shared_dict.add_symbol(lang_token)

        self.strategy = MaskPredict(decoding_iterations=10, tokenizer=self.tokenizer)
        self.dynamic_length = False
        self.random = np.random.RandomState(self.config['seed'])
        self.mask_range = True
        self.embed_lengths = nn.Embedding(self.config['max_length'], self.model.config.d_model)

    def predict_lengths(self, encoder_out, src_masks): # B X T
        enc_feats = encoder_out.transpose(0,1)  # T x B x C
        enc_feats = _mean_pooling(enc_feats, src_masks) # T X B
        length_out = F.linear(enc_feats, self.embed_lengths.weight)
        return F.log_softmax(length_out, -1)

    def mask(self, source, target):
        if self.dynamic_length:
            max_len = 3 * len(source) // 2 + 1
            target = target.new((target.tolist() + ([self.tokenizer.eos_token_id] * (max_len - len(target))))[:max_len])
        
        min_num_masks = 1
        
        enc_source = source
        dec_source = target.new(target.tolist())
        dec_target_cp = target.new(target.tolist())
        dec_target = target.new([self.tokenizer.pad_token_id] * len(dec_source))

        pad_indices = torch.nonzero(dec_source.eq(self.tokenizer.pad_token_id))
        pad_start_ind = pad_indices.min().item() if pad_indices.size(0) > 0 else len(dec_source)
        valid_mask_index = np.arange(1, pad_start_ind) # don't mask lang id token and pad
        
        if self.trainer.training:
            if min_num_masks < len(valid_mask_index):
                sample_size = self.random.randint(min_num_masks, len(valid_mask_index))
            else:
                sample_size = len(valid_mask_index)

            if self.mask_range:
                start = self.random.randint(pad_start_ind - sample_size)
                ind = list(range(start, start + sample_size))
            else:
                ind = self.random.choice(valid_mask_index, size=sample_size, replace=False)
            
            dec_source[ind] = self.tokenizer.mask_token_id
            dec_target[ind] = dec_target_cp[ind]
        else:
            dec_target = dec_target_cp
            dec_source[:pad_start_ind] = self.tokenizer.mask_token_id

        ntokens = dec_target.ne(self.tokenizer.pad_token_id).sum(-1).item()
        print ("masked tokens", self.tokenizer.convert_ids_to_tokens(dec_source))
        print ("original tokens", self.tokenizer.convert_ids_to_tokens(dec_target))
        print ("source tokens", self.tokenizer.convert_ids_to_tokens(enc_source))

        """
        enc_source: encoder tokens
        dec_source: mix of [mask] and actual tokens that are fed into model as clue for predicting dec_target
        dec_target: masked tokens in dec_source are shown, other tokens are replaced with pad
        """
        return enc_source, dec_source, dec_target, ntokens
    
    def forward(self, **inputs):
        """
        **inputs: output of tokenizer, dict("input_ids", "attention_mask", "labels")
        """
        return self.model(**inputs)
    

    def preprocess_dataset_batch(self, batch, train: bool, unlabeled: bool):
        '''
        add keys "masked_labels" and "new_labels" in batch,
        to feed into the model
        '''
        if train and not unlabeled:
            batch["masked_labels"] = [] 
            batch["new_labels"] = []
            for source, target in zip(batch["input_ids"], batch["labels"]):
                enc_source, dec_source, dec_target, ntokens = self.mask(source, target)
                batch["masked_labels"].append(dec_source)
                batch["new_labels"].append(dec_target)
            batch["labels"] = torch.stack(batch["masked_labels"]) # input to the decoder
            batch.pop("masked_labels")
            #batch["original_labels"] = batch["labels"] # full labels
            batch["new_labels"] = torch.stack(batch["new_labels"]) # labels with all tokens except mask padded

        return batch

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

    def mlm_loss(self, outputs, labels):
        # outputs: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len] with pads in non-mask tokens 
        output_dim = outputs.shape[-1]
        outputs = outputs.contiguous().view(-1, output_dim) # [batch size * trg len, output_dim]
        labels = labels.contiguous().view(-1) # [batch size * trg len]
        loss = F.cross_entropy(outputs, labels, ignore_index=self.tokenizer.pad_token_id)
        return loss

    def ctc_loss(self, outputs, labels, mask_index):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch.pop("labels")
        batch.pop("decoder_input_ids")

        # forward pass, teacher forcing with generated translation

        pred_labels, outputs, cross_attention = self.mask_predict(batch, output_attentions=True)

        # prepare attention (last layer of decoder)
        cross_attention = cross_attention.unsqueeze(1)
        cross_attention = cross_attention[:, :, 1:, 1:] # (batch_size, num_heads, target seq length, input seq length)

        # calculate loss, teacher forcing with real labels
        
        teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
        val_loss = teacher_forced_outputs.loss

        self.log('val_loss', val_loss.item(), sync_dist=True) # loss is unsupervised loss (targets are beam search output, not gold target)

        # pred_labels: [trg lang code] X [eos] [pad] 
        # labels: [trg lang code] X [eos] [pad] 
        return {"loss": val_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention}
    

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch.pop("labels")
        batch.pop("decoder_input_ids")

        # forward pass, teacher forcing with generated translation

        pred_labels, outputs, cross_attention = self.mask_predict(batch, output_attentions=True)
        # outputs: Seq2SeqLMOutput = self.model(**batch, output_attentions=True)
        # pred_labels = torch.argmax(outputs.logits, dim=-1)

        # prepare attention (last layer of decoder)
        cross_attention = cross_attention.unsqueeze(1)
        cross_attention = cross_attention[:, :, 1:, 1:] # (batch_size, num_heads, target seq length, input seq length)

        # calculate loss, teacher forcing with real labels
        teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
        test_loss = teacher_forced_outputs.loss
        
        # teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
        # test_loss = teacher_forced_outputs.loss

        self.log('test_loss', test_loss.item(), sync_dist=True) # loss is unsupervised loss (targets are beam search output, not gold target)

        # pred_labels: [trg lang code] X [eos] [pad] 
        # labels: [trg lang code] X [eos] [pad] 
        return {"loss": test_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention}
    
        
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

class NAMBARTSupPL(NAMBARTPL):
    def training_step(self, batch, batch_idx):
        batch = self.preprocess_dataset_batch(batch, train=True, unlabeled=False)
        # batch: dict("input_ids", "attention_mask", "labels") -> dict("input_ids", "attention_mask", "labels", "new_labels")
        new_labels = batch.pop("new_labels")

        # run through model
        output: Seq2SeqLMOutput = self.model.forward(**batch, return_dict=True)
        
        mlm_loss = self.mlm_loss(output.logits, new_labels)
        self.log("train_loss", mlm_loss, sync_dist=True)
        return mlm_loss


class NAMBARTSslPL(NAMBARTPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer):
        super().__init__(active_config, config, device, tokenizer)
        self.epoch_scaled_scores = []
        self.epoch_raw_scores = []
        self.tokenizer = tokenizer

        def unsup_criterion(src: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor, attention: torch.Tensor, scaled_score_record, raw_score_record, verbose):
            '''
            src: [batch size, src len] ([src_lang_code] X [eos] [pad] * N )
            outputs: [batch size, trg len, output dim] 
            labels: [batch size, trg len] ([trg_lang_code] Y [eos] [pad] * N) (all non-mask tokens padded out)
            attention: [batch size, n heads, trg len, src len]
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
            # TODO: to calculate scores, only consider attention for non-padded (masked) tokens
            attention = attention[:, :, 1:, 1:]

            # set loss function, ignore pad
            sup_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index = self.model.config.pad_token_id)

            # reshape outputs, labels and calculate loss
            output_dim = outputs.shape[-1]
            outputs = outputs.contiguous().view(-1, output_dim) # [batch size * trg len, output_dim]
            labels = labels.contiguous().view(-1) # [batch size * trg len]
            batch_sup_loss = sup_criterion(outputs, labels).view(batch_size, -1).sum(dim=1) # batch size

            # reshape labels back
            labels = labels.view(batch_size,-1) 
            
            assert labels.shape[1] == attention.shape[2] # trg len
            assert src.shape[1] == attention.shape[-1] # src len

            # don't use attention score for non-mask tokens
            #non_mask_indices = labels.ne(self.tokenizer.mask_token_id) # [bsz, target len]
            #non_mask_indices = non_mask_indices.unsqueeze(1) # add head dim
            #non_mask_indices = non_mask_indices.unsqueeze(-1) # add src len dim 
            #valid_attention = attention * non_mask_indices
            valid_attention = attention

            with torch.no_grad():
                # get batch scores

                if self.config['score'] == "base":
                    batch_scaled_scores, batch_raw_scores = self.scorer.base_score(valid_attention, scaled_score_record, raw_score_record)
                elif self.config['score'] == "uniform":
                    batch_scaled_scores, batch_raw_scores = self.scorer.uniform_score(src)
                elif self.config['score'] == "fast_align":
                    batch_scaled_scores, batch_raw_scores = self.scorer.fast_align_alignment_score(src, labels, attention, scaled_score_record, raw_score_record)
                elif self.config['score'] == "awesome_align":
                    batch_scaled_scores, batch_raw_scores = self.scorer.awesome_align_alignment_score(src, labels, attention, scaled_score_record, raw_score_record)
                elif self.config['score'] == "dep_parse_awesome_align":
                    batch_scaled_scores, batch_raw_scores = self.scorer.dependency_parse_score_awesome_align(src, labels, attention, scaled_score_record, raw_score_record)
                elif self.config['score'] == "dep_parse_base_align":
                    batch_scaled_scores, batch_raw_scores = self.scorer.dependency_parse_score_base_align(src, labels, attention, scaled_score_record, raw_score_record)

                batch_scaled_scores = batch_scaled_scores.float()
                batch_raw_scores = batch_raw_scores.float()

            # filter samples whose confidence is lower than mean score
            good_samples = (batch_raw_scores > np.quantile(raw_score_record, 0.5)) # indices of samples to keep
            print("good_samples: ", torch.sum(good_samples).item())

            # filter
            batch_scaled_scores = batch_scaled_scores[good_samples]
            batch_raw_scores = batch_raw_scores[good_samples]
            batch_sup_loss = batch_sup_loss[good_samples]

            batch_scaled_scores = batch_scaled_scores.to(batch_sup_loss)
            batch_raw_scores = batch_raw_scores.to(batch_sup_loss)

            assert batch_scaled_scores.requires_grad == False
            assert batch_raw_scores.requires_grad == False
            assert batch_sup_loss.requires_grad == True
            
            print("batch unsup loss (cross entropy): ", batch_sup_loss)

            loss = batch_scaled_scores.dot(batch_sup_loss)/len(batch_scaled_scores)

            print("batch unsup loss (cross entropy * scores): ", loss)
            assert loss.requires_grad == True
                
            return loss, batch_scaled_scores, batch_raw_scores

        self.unsup_criterion = unsup_criterion


    def training_step(self, batch, batch_idx):

        sup_inputs = batch['label'] # TODO maybe change 'label' to 'sup' or 'parallel'?
        unsup_inputs = batch['unlabel']

        # forward pass through model - sup, unsup separately

        # sup inputs
        sup_inputs = self.preprocess_dataset_batch(sup_inputs, train=True, unlabeled=False)
        new_labels = sup_inputs.pop("new_labels")

        sup_outputs: Seq2SeqLMOutput = self.model(**sup_inputs, return_dict=True)
        loss_sup = self.mlm_loss(sup_outputs.logits, new_labels)
        
        # unsup inputs
        unsup_output_ids, _ = self.mask_predict(unsup_inputs, output_attentions=False)
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
        unsup_inputs["labels"] = unsup_labels # full labels
        unsup_inputs = self.preprocess_dataset_batch(unsup_inputs, train=True, unlabeled=False)
        unsup_labels = unsup_inputs["labels"] # labels with all non-mask tokens padded 
        unsup_new_labels = unsup_inputs.pop("new_labels") 

        unsup_outputs: Seq2SeqLMOutput = self.model.forward(**unsup_inputs, output_attentions=True, return_dict=True)

        unsup_logits = unsup_outputs.logits
        unsup_attention = unsup_outputs.cross_attentions[-1] # (batch_size, num_heads, target seq length, input seq length)

        # calculate unsupervised loss
        loss_unsup, scaled_scores, raw_scores = self.unsup_criterion(
            unsup_inputs["input_ids"], unsup_logits, unsup_new_labels, unsup_attention, 
            self.epoch_scaled_scores, self.epoch_raw_scores,
            batch_idx % 64 == 0)
        
        self.epoch_scaled_scores.extend(scaled_scores.tolist())
        self.epoch_raw_scores.extend(raw_scores.tolist())
        self.log("scores mean", scaled_scores.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) # batch mean score
        
        # total loss
        loss = loss_sup + self.config['unsup_wt'] * loss_unsup 
        self.log("train_loss_sup", loss_sup.item(), sync_dist=True)
        self.log("train_loss_unsup", loss_unsup.item(), sync_dist=True)
        self.log("train_loss", loss.item(), sync_dist=True)
        return loss