from typing import Dict

import torch
from torch import nn, optim, Tensor

import pytorch_lightning as pl

from transformers import PreTrainedTokenizer, BartForConditionalGeneration, XLMRobertaModel, BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
import huggingface_hub

from comet import download_model, load_from_checkpoint
from comet.models.multitask.unified_metric import UnifiedMetric
from comet.encoders.xlmr import XLMREncoder
from comet.models.utils import Prediction

from score.score import Scorer

class BARTPL(pl.LightningModule):

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

        self.tokenizer = tokenizer

        # bart architecture (but different vocab size)
        bart_model_config = BartConfig.from_pretrained("facebook/bart-base")
        bart_model_config.vocab_size = self.tokenizer.vocab_size
        self.model = BartForConditionalGeneration(bart_model_config)
        self.model.apply(self.model._init_weights)

        # download energy model
        with open("hf_token.txt") as f:
            token = f.readline().strip()
            huggingface_hub.login(token)

        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        self.energy_model: UnifiedMetric = load_from_checkpoint(model_path)
        self.energy_model.eval() # don't train the energy model
        encoder_model: XLMRobertaModel = self.energy_model.encoder.model

        # initialize model embeddings with XLM embeddings
        encoder_model_embeddings = encoder_model.get_input_embeddings()
        original_embedding_matrix = encoder_model.get_input_embeddings().weight

        input_embedding_adapter = nn.Linear(encoder_model.config.hidden_size, self.model.config.hidden_size)
        encoder_input_embeddings = encoder_model_embeddings.requires_grad_(False)
        self.model.set_input_embeddings(nn.Sequential(encoder_input_embeddings, input_embedding_adapter))

        output_embedding_adapter = nn.Linear(self.model.config.hidden_size, encoder_model.config.hidden_size)
        encoder_output_embeddings = nn.Linear(original_embedding_matrix.size(1), original_embedding_matrix.size(0)) # [in, out] = [hid_dim, vocab_size]
        encoder_output_embeddings.weight = original_embedding_matrix
        encoder_output_embeddings = encoder_output_embeddings.requires_grad_(False)
        self.model.set_output_embeddings(nn.Sequential(output_embedding_adapter, encoder_output_embeddings))

        self.scorer = Scorer(self.active_config, self.config, self.tokenizer)
    
    def forward(self, **inputs):
        """
        **inputs: output of tokenizer, dict("input_ids", "attention_mask", "labels")
        """
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        pass
    
    @torch.autocast("cuda")
    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch.pop("labels")
        batch.pop("decoder_input_ids")

        if not self.config['eval_teacher_forcing']:
            # [batch_size,1] filled with decoder_start_token_id
            decoder_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.model.config.decoder_start_token_id
            decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)

            pred_ids = self.model.generate(**batch,
                            do_sample=True,
                            num_beams=5,
                            num_return_sequences=1,
                            max_new_tokens = self.config['max_length'],
                            decoder_input_ids=decoder_start_tokens, decoder_attention_mask=decoder_attention_mask, 
                            use_cache=False, synced_gpus=True, min_length=4)

            # exclude eos token at the beginning
            pred_labels = pred_ids[:, 1:].clone()

            # forward pass, teacher forcing with generated translation
            beam_outputs: Seq2SeqLMOutput = self.model.forward(**batch, labels=pred_labels, output_attentions=True)
            if type(beam_outputs) is tuple:
                beam_outputs = beam_outputs[0]

            # prepare attention (last layer of decoder)
            # TODO: exclude attention for pad tokens
            cross_attention = beam_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, :, :] # (batch_size, num_heads, target seq length, input seq length)
            encoder_attention = beam_outputs.encoder_attentions[-1][:, :, :, :]
            decoder_attention = beam_outputs.decoder_attentions[-1][:, :, :, :]

            print("cross attention shape", cross_attention.shape)
            print("encoder_attention shape", encoder_attention.shape)
            print("decoder_attention shape", decoder_attention.shape)
            
        # calculate loss, teacher forcing with real output
        teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
        if type(teacher_forced_outputs) is tuple:
            teacher_forced_outputs = teacher_forced_outputs[0]

        if self.config["eval_teacher_forcing"]:
            pred_labels = torch.argmax(teacher_forced_outputs.logits, dim=-1)
            cross_attention = teacher_forced_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # leave out attention for language token ids
            encoder_attention = teacher_forced_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = teacher_forced_outputs.decoder_attentions[-1][:, :, 1:, 1:]
        val_loss = teacher_forced_outputs.loss
        self.log('val_loss', val_loss.item(), sync_dist=True) # loss is unsupervised loss (targets are beam search output, not gold target)

        return {"loss": val_loss, "preds": pred_labels, "labels": labels, "cross_attention": cross_attention, "encoder_attention": encoder_attention, "decoder_attention": decoder_attention}
    
    @torch.autocast("cuda")
    def test_step(self, batch, batch_idx):
        # TODO: update to be the same as validation step
        labels = batch["labels"]
        batch.pop("labels")
        batch.pop("decoder_input_ids")

        if not self.config['eval_teacher_forcing']:
            # [batch_size,1] filled with decoder_start_token_id
            decoder_start_tokens = torch.ones_like(batch.input_ids)[:, :1] * self.model.config.decoder_start_token_id
            decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)


            print("batch input_ids.device: ", batch["input_ids"].device)
            print("decoder_start_tokens.device: ", decoder_start_tokens.device)
            print("decoder_attention_mask.device: ", decoder_attention_mask.device)

            pred_ids = self.model.generate(**batch,
                                           do_sample=True,
                            num_beams=5,
                            num_return_sequences=1,
                            max_new_tokens = self.config['max_length'],
                            decoder_input_ids=decoder_start_tokens, decoder_attention_mask=decoder_attention_mask, 
                            use_cache=False, synced_gpus=True)

            pred_labels = pred_ids[:, 1:].clone()
            # leave out <eos> at the beginning

            print(pred_labels.shape)
        
            repeated_batch = {}
            repeated_batch['input_ids'] = batch['input_ids'].repeat(1,1)
            repeated_batch['attention_mask'] = batch['attention_mask'].repeat(1,1)
            print(repeated_batch['input_ids'].shape)
            print(repeated_batch['attention_mask'].shape)

            # forward pass, teacher forcing with generated translation
            beam_outputs: Seq2SeqLMOutput = self.model(**repeated_batch, labels=pred_labels, output_attentions=True)
            # prepare attention (last layer of decoder)
            if type(beam_outputs) is tuple:
                beam_outputs = beam_outputs[0]
            cross_attention = beam_outputs.cross_attentions[self.config['cross_attention_layer']][:, :, 1:, 1:] # (batch_size, num_heads, target seq length, input seq length)
            encoder_attention = beam_outputs.encoder_attentions[-1][:, :, 1:, 1:]
            decoder_attention = beam_outputs.decoder_attentions[-1][:, :, 1:, 1:]

            
        # calculate loss, teacher forcing with real output
        teacher_forced_outputs: Seq2SeqLMOutput = self.model(**batch, labels=labels, output_attentions=self.config['eval_teacher_forcing'])
        if type(teacher_forced_outputs) is tuple:
            teacher_forced_outputs = teacher_forced_outputs[0]
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
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=0.00001)
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

class BARTSupPL(BARTPL):
    def __init__(self, active_config, config, device, tokenizer):
        super().__init__(active_config, config, device, tokenizer)
        del self.energy_model

    @torch.autocast("cuda")
    def training_step(self, batch, batch_idx):
        print("training step")
        print(batch)
        outputs = self(**batch)
        loss = outputs[0]
        if type(loss) is Seq2SeqLMOutput:
            loss = loss.loss
        self.log("train_loss", loss, sync_dist=True)
        return loss


class BARTSslPL(BARTPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer):
        super().__init__(active_config, config, device, tokenizer)
        self.epoch_scores = []
        self.tokenizer = tokenizer

        def unsup_criterion(src, outputs, labels, attention, score_record):
            # TODO: inspect tokenizer more carefully and determine the following
            '''
            src: [batch size, src len - 1]
            outputs: [batch size * (trg len), output dim]
            labels: [batch size * (trg len)]
            attention: [batch size, n heads, trg len, src len]
            '''
            
            #print("Check grad exists")
            #print(src.requires_grad)
            #print(outputs.requires_grad)
            #print(labels.requires_grad)
            #print(attention.requires_grad)

            batch_size = src.shape[0]
            sup_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index = self.model.config.pad_token_id)

            batch_sup_loss = sup_criterion(outputs, labels).view(batch_size, -1).sum(dim=1) # batch size

            labels = labels.view(batch_size,-1) # reshape
            labels = labels[:, 1:].clone() # exclude pad token

            with torch.no_grad():
                # get batch scores

                if self.config['score'] == "base":
                    batch_scores = self.scorer.base_score(attention, score_record)
                    batch_scores = batch_scores.float()
                elif self.config['score'] == "uniform":
                    batch_scores = self.scorer.uniform_score(src)
                    batch_scores = batch_scores.float()
                elif self.config['score'] == "fast_align": 
                    batch_scores = self.scorer.fast_align_alignment_score(src, labels, attention, score_record) # batch_size
                    batch_scores = batch_scores.float()
                elif self.config['score'] == "awesome_align":
                    batch_scores = self.scorer.awesome_align_alignment_score(src, labels, attention, score_record)
                    batch_scores = batch_scores.float()
                elif self.config['score'] == "dep_parse_awesome_align":
                    batch_scores = self.scorer.dependency_parse_score_awesome_align(src, labels, attention, score_record)
                    batch_scores = batch_scores.float()
                elif self.config['score'] == "dep_parse_base_align":
                    batch_scores = self.scorer.dependency_parse_score_base_align(src, labels, attention, score_record)
                    batch_scores = batch_scores.float()
                
            batch_scores = batch_scores.to(batch_sup_loss)

            assert batch_scores.requires_grad == False
            assert batch_sup_loss.requires_grad == True

            loss = batch_scores.dot(batch_sup_loss)
            assert loss.requires_grad == True
                
            return loss, batch_scores

        self.unsup_criterion = unsup_criterion


    def training_step(self, batch, batch_idx):

        sup_inputs = batch['label'] # TODO maybe change 'label' to 'sup' or 'parallel'?
        unsup_inputs = batch['unlabel']

        # forward pass through model - sup, unsup separately

        # labeled inputs
        # in labels, change pad token id (=0) to -100 so that they will be ignored during loss calculation
        
        # print("labels: ", sup_inputs['labels']) # check that pad token id is not -100
        original_labels = sup_inputs['labels']
        sup_inputs['labels'] = torch.where(sup_inputs['labels'] == self.model.config.pad_token_id, -100, sup_inputs['labels'])
        
        sup_outputs: Seq2SeqLMOutput = self.model(**sup_inputs)
        loss_sup, sup_logits = sup_outputs.loss, sup_outputs.logits
        output_dim = sup_logits.shape[-1]
        
        # unsup_inputs
        if self.config['selfsup_strategy'] == "greedy":
            do_sample = False
            num_beams = 1
        elif self.config['selfsup_strategy'] == "sample":
            do_sample = True
            num_beams = 1
        elif self.config['selfsup_strategy'] == "beam":
            do_sample = False
            num_beams = 5

        decoder_start_tokens = torch.ones_like(unsup_inputs.input_ids)[:, :1] * self.model.config.decoder_start_token_id
        decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)

        # generate, only get the output ids
        unsup_output_ids: torch.LongTensor = self.model.generate(**unsup_inputs, 
                                            do_sample=do_sample, num_beams=num_beams,
                                            num_return_sequences=1,
                                            max_new_tokens = self.config['max_length'],
                                            decoder_input_ids=decoder_start_tokens, decoder_attention_mask=decoder_attention_mask, 
                                            use_cache=False, synced_gpus=True)

        '''
        unsup_outputs: GenerateOutput = self.model.generate_with_grad(unsup_inputs.input_ids, 
                                            do_sample=do_sample, num_beams=num_beams, 
                                            output_attentions=True, output_scores=True, # output_hidden_states=True,
                                            return_dict_in_generate=True, num_return_sequences=1,
                                            decoder_input_ids=decoder_start_tokens, decoder_attention_mask=decoder_attention_mask, 
                                            max_new_tokens = self.config['max_length'],
                                            use_cache=False,synced_gpus=True)
        '''

        unsup_labels = unsup_output_ids[:, 1:].clone()
        if batch_idx % 64 == 0:
            #print("sup_inputs: ", vars(sup_inputs))
            sup_src = self.tokenizer.batch_decode(sup_inputs.input_ids)
            print("sup input src: ", sup_src)

            original_labels = torch.where(original_labels == -100, self.model.config.pad_token_id, original_labels)
            sup_labels = self.tokenizer.batch_decode(original_labels) # use original labels to avoid decoding -100
            print("sup input labels:  ", sup_labels)

            sup_decoder_ids = self.tokenizer.batch_decode(sup_inputs.decoder_input_ids)
            print("sup input decoder ids: ", sup_decoder_ids)

            sup_prediction = self.tokenizer.batch_decode(torch.argmax(sup_outputs.logits,-1))
            print("sup input prediction: ", sup_prediction)

            unsup_src = self.tokenizer.batch_decode(unsup_inputs.input_ids)
            print("unsup input src: ", unsup_src)

            #print("unsup_outputs:", vars(unsup_outputs))
            unsup_prediction = self.tokenizer.batch_decode(unsup_labels)
            print("unsup input prediction: ", unsup_prediction)

        # forward pass, teacher forcing with output ids as the labels 
        unsup_outputs: Seq2SeqLMOutput = self.model(**unsup_inputs, labels=unsup_labels, output_attentions=True)
        unsup_logits = unsup_outputs.logits
        
         # reshape output ids, logits to calculate loss
        unsup_labels = unsup_labels.view(-1)
        unsup_logits = unsup_logits.contiguous().view(-1, output_dim)  # (batch size * seq length, output_dim)
        
        # prepare attention (last layer of decoder)
        unsup_attention = unsup_outputs.cross_attentions[-1] # (batch_size, num_heads, generated_length, sequence_length)

        loss_unsup, scores = self.unsup_criterion(unsup_inputs["input_ids"][:, :], unsup_logits, unsup_labels, unsup_attention, self.epoch_scores)
        self.epoch_scores.extend(scores.tolist())
        self.log("scores mean", scores.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True) # batch mean score
        
        # total loss
        loss = loss_sup + self.config['unsup_wt'] * loss_unsup 
        self.log("train_loss_sup", loss_sup.item(), sync_dist=True)
        self.log("train_loss_unsup", loss_unsup.item(), sync_dist=True)
        self.log("train_loss", loss.item(), sync_dist=True)
        return loss
    
class BARTSsl_EBMPL(BARTPL):
    def __init__(self, active_config, config, device, tokenizer):
        super().__init__(active_config, config, device, tokenizer)

        '''
        # download energy model
        with open("hf_token.txt") as f:
            token = f.readline().strip()
            huggingface_hub.login(token)

        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        self.energy_model: UnifiedMetric = load_from_checkpoint(model_path)
        self.energy_model.eval() # don't train the energy model
        '''
        encoder_model: XLMRobertaModel = self.energy_model.encoder.model

        encoder_model_embeddings = encoder_model.get_input_embeddings()
        original_embedding_matrix = encoder_model.get_input_embeddings().weight
        '''
        input_embedding_adapter = nn.Linear(encoder_model.config.hidden_size, self.model.config.hidden_size)
        encoder_input_embeddings = encoder_model_embeddings.requires_grad_(False)
        self.model.set_input_embeddings(nn.Sequential(encoder_input_embeddings, input_embedding_adapter))

        output_embedding_adapter = nn.Linear(self.model.config.hidden_size, encoder_model.config.hidden_size)
        encoder_output_embeddings = nn.Linear(original_embedding_matrix.size(1), original_embedding_matrix.size(0)) # [in, out] = [hid_dim, vocab_size]
        encoder_output_embeddings.weight = original_embedding_matrix
        encoder_output_embeddings = encoder_output_embeddings.requires_grad_(False)
        self.model.set_output_embeddings(nn.Sequential(output_embedding_adapter, encoder_output_embeddings))
        '''
        # set embedding to linear layer
        new_embedding = nn.Linear(original_embedding_matrix.size(0), original_embedding_matrix.size(1)) # [in, out] = [vocab_size, hid_dim]
        new_embedding.weight = nn.Parameter(original_embedding_matrix.t().contiguous())
        print("embedding weight matrix shape", new_embedding.weight.shape) # [out, in] = [hid_dim, vocab_size]
        encoder_model.set_input_embeddings(new_embedding)

    def pad_tensor(
        self, tensor: torch.Tensor, length: torch.Tensor, padding_index: int
    ) -> torch.Tensor:
        """Pad a tensor to length with padding_index.

        Args:
            tensor (torch.Tensor): Tensor to pad.
            length (torch.Tensor): Sequence length after padding.
            padding_index (int): Index to pad tensor with.

        Returns:
            torch.Tensor: Input batch
        """
        n_padding = length - tensor.shape[0]
        assert n_padding >= 0
        if n_padding == 0:
            return tensor
        padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
        return torch.cat((tensor, padding), dim=0)
        
    def energy_model_concat_inputs(self, src_input_ids: Tensor, mt_input_logits: Tensor, mt_input_ids: Tensor):
        # src_input_ids: 
        # [CLS] X [SEP] [PAD], size = [bsz, src_length]
        # mt_input_logits:
        # [CLS] Y [SEP] [PAD], size = [bsz, mt_length, vocab_size]
        # result:
        # [CLS] X [SEP] [SEP] Y [SEP] [PAD], size = [bsz, concated length, vocab_size]
        
        vocab_size = mt_input_logits.size(-1)
        print("vocab_size: ", vocab_size)

        concat_logits = []
        src_input_ids = [
            x.masked_select(x.ne(self.tokenizer.pad_token_id))[1:-1]
            for x in src_input_ids.unbind(dim=0)
        ]
        concat_logits.append(src_input_ids)
        
        mt_input_logits = [
            x.masked_select(y.ne(self.tokenizer.pad_token_id).unsqueeze(1)).view(-1, vocab_size)[1:-1]
            for x,y in zip(mt_input_logits.unbind(dim=0), mt_input_ids.unbind(dim=0))
        ]
        concat_logits.append(mt_input_logits)

        # Concatenate inputs into a single batch
        batch_size = len(concat_logits[0])
        batch = []
        for i in range(batch_size):
            cls_one_hot = torch.zeros(vocab_size, device=concat_logits[0][i].device)
            print("cls token id: ", self.tokenizer.cls_token_id)
            print("cls one hot shape: ", cls_one_hot.shape)
            cls_one_hot[self.tokenizer.cls_token_id] = 1
            
            print("cls one hot prepared")
            sep_one_hot = torch.zeros(vocab_size, device=concat_logits[0][i].device)
            sep_one_hot[self.tokenizer.sep_token_id] = 1

            concat_logits[1][i] = concat_logits[1][i].to(concat_logits[0][i])
            if concat_logits[1][i].size(0) > self.tokenizer.max_len_single_sentence - 2:
                concat_logits[1][i] = concat_logits[1][i][:self.tokenizer.max_len_single_sentence-2]

            # change ids into one hot vectors
            id_to_one_hot = torch.zeros(concat_logits[0][i].size(0), vocab_size, device=concat_logits[0][i].device)
            id_to_one_hot[torch.arange(concat_logits[0][i].size(0)), concat_logits[0][i]] = 1
            concat_logits[0][i] = id_to_one_hot
            if concat_logits[0][i].size(0) > self.tokenizer.max_len_single_sentence - 2:
                concat_logits[0][i] = concat_logits[1][i][:self.tokenizer.max_len_single_sentence-2]

            print("prepare logits done")

            new_logits = torch.concat([cls_one_hot.unsqueeze(0),
                                       concat_logits[0][i],
                                       sep_one_hot.unsqueeze(0), sep_one_hot.unsqueeze(0),
                                       concat_logits[1][i],
                                       sep_one_hot.unsqueeze(0)]) # [concated length, vocab_size]

            batch.append(new_logits)
        
        print(batch[0].shape)

        lengths = [t.shape[0] for t in batch]
        max_len = max(lengths)
        padded = [
            self.pad_tensor(t, max_len, self.tokenizer.pad_token_id) for t in batch
        ]

        lengths = torch.tensor(lengths, dtype=torch.long)
        padded = torch.stack(padded, dim=0).contiguous()
        
        print(padded.shape)

        attention_mask = torch.arange(max_len)[None, :] < lengths[:, None]
        
        encoder_input = {"input_ids": padded, "attention_mask": attention_mask}

        return encoder_input, lengths, max_len
        
    @torch.autocast("cuda")
    def energy_model_forward(self, input_dist: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward function.

        Args:
            input_dist (torch.Tensor): Input distributions
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (Optional[torch.Tensor], optional): Token type ids for
                BERT-like models. Defaults to None.

        Raises:
            Exception: Invalid model word/sent layer if self.{word/sent}_layer are not
                valid encoder model layers .

        Returns:
            Dict[str, torch.Tensor]: Sentence scores and word-level logits (if
                word_level_training = True)
        """
        
        encoder: XLMREncoder = self.energy_model.encoder

        inputs_embeds = encoder.model.get_input_embeddings().forward(input_dist) 
        # input_dist: [bsz, seq_len, vocab_size]
        # linear layer: map [vocab_size -> hid_dim]
        # input_embeds: [bsz, seq_len, hid_dim]

        inputs_embeds = inputs_embeds.to(encoder.model.device)
        attention_mask = attention_mask.to(encoder.model.device)

        last_hidden_states, _, all_layers = encoder.model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )

        encoder_out = {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }

        # Word embeddings used for the sentence-level regression task
        if self.energy_model.layerwise_attention:
            sentemb = self.energy_model.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )[:, 0, :]

        elif (
            isinstance(self.energy_model.hparams.sent_layer, int)
            and 0 <= self.energy_model.hparams.sent_layer < self.energy_model.encoder.num_layers
        ):
            sentemb = encoder_out["all_layers"][self.energy_model.hparams.sent_layer][:, 0, :]
        else:
            raise Exception(
                "Invalid model sent layer {}.".format(self.energy_model.hparams.word_layer)
            )

        return Prediction(score=self.energy_model.estimator(sentemb).view(-1))


    def training_step(self, batch, batch_idx):
        sup_inputs = batch['label']
        unsup_inputs = batch['unlabel']

        # forward pass through model - sup, unsup separately

        # sup inputs
        sup_outputs: Seq2SeqLMOutput = self.model(**sup_inputs)
        loss_sup, sup_logits = sup_outputs.loss, sup_outputs.logits

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
        decoder_attention_mask = torch.where(decoder_start_tokens==self.model.config.pad_token_id, 0, 1)
        
        # generate, only get the output ids
        unsup_output_ids: torch.LongTensor = self.model.generate(**unsup_inputs, 
                                            do_sample=do_sample, num_beams=num_beams,
                                            num_return_sequences=1,
                                            max_new_tokens = self.config['max_length'],
                                            decoder_input_ids=decoder_start_tokens, decoder_attention_mask=decoder_attention_mask, 
                                            use_cache=False, synced_gpus=True, min_length=4,
                                            top_p=top_p, top_k=top_k)

        unsup_labels = unsup_output_ids[:, 1:].clone()
        # leave out <eos> at the beginning -> [sos] Y [eos] [pad] * N

        # forward pass, teacher forcing with output ids as the labels 
        unsup_outputs: Seq2SeqLMOutput = self.model(**unsup_inputs, labels=unsup_labels, output_attentions=True)
        energy_model_input = torch.softmax(unsup_outputs.logits, -1)

        energy_model_concat_inputs = self.energy_model_concat_inputs(unsup_inputs.input_ids, energy_model_input, unsup_labels)
        
        prediction = self.energy_model_forward(energy_model_concat_inputs[0]['input_ids'],
                                               energy_model_concat_inputs[0]['attention_mask'])

        loss_unsup = -(prediction['score'].mean())
        loss = loss_sup + self.config['unsup_wt'] * loss_unsup
        
        assert loss_unsup.requires_grad == True
        assert energy_model_input.requires_grad == True

        self.log("train_loss_sup", loss_sup.item(), sync_dist=True)
        self.log("train_loss_unsup", loss_unsup.item(), sync_dist=True)
        self.log("train_loss", loss.item(), sync_dist=True)

        return loss