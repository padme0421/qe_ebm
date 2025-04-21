from energy_model.base_energy_model import EnergyModel
import huggingface_hub

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from transformers import (
    XLMRobertaModel, AutoTokenizer)
import adapters
from adapters import (setup_adapter_training, AdapterArguments)

import huggingface_hub

from comet import download_model, load_from_checkpoint
from comet.models.multitask.unified_metric import UnifiedMetric
from comet.encoders.xlmr import XLMREncoder
from peft import LoraConfig, TaskType, get_peft_model

from score.score import Scorer
from pl_module.utils import (check_trainable_params, register_param_hooks, 
                             get_length, STL, get_adapters, count_optim_params
                             )

class COMET_EnergyModel(EnergyModel):
    def __init__(self, config, active_config):
        '''
        1) load energy model
        2) make embeddings linear
        3) configure adapters & param requires_grad attibutes
        4) register backward hooks
        '''
        super(COMET_EnergyModel, self).__init__(config, active_config)
        self.config = config
        self.active_config = active_config

        # 1) load energy model
        with open("hf_token.txt") as f:
            token = f.readline().strip()
            huggingface_hub.login(token)

        energy_model_path = download_model('Unbabel/wmt22-cometkiwi-da')
        self.model: UnifiedMetric = load_from_checkpoint(energy_model_path)
        self.base_model = self.model
        self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/infoxlm-large')

        # 2) make embeddings linear
        encoder_model: XLMRobertaModel = self.model.encoder.model
        original_embedding_matrix = encoder_model.get_input_embeddings().weight
        new_embedding = nn.Linear(original_embedding_matrix.size(0), original_embedding_matrix.size(1)) # [in, out] = [vocab_size, hid_dim]
        new_embedding.weight = nn.Parameter(original_embedding_matrix.t().contiguous(), requires_grad=True)
        print("embedding weight matrix shape", new_embedding.weight.shape) # [out, in] = [hid_dim, vocab_size]
        encoder_model.set_input_embeddings(new_embedding)

        # 3) configure adapters & param requires_grad attibutes
        # SEAL STATIC
        if not config['train_energy']:
            self.model.requires_grad_(True) # for registering backward hook
            register_param_hooks(self.model.encoder, "energy model [ENCODER]")
            self.model.freeze_encoder()

        # SEAL-DYNAMIC
        else:
            if config['train_energy_encoder']:
                self.model.requires_grad_(True)
                if config['energy_model_adapter']:
                    adapters.init(encoder_model)
                    # lora
                    #peft_config = LoraConfig(
                    #    target_modules=['query','value'], task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, 
                    #    r=self.config['energy_lora_rank'], lora_alpha=self.config['energy_lora_rank'], lora_dropout=0.1
                    #)
                    #encoder_model = get_peft_model(encoder_model, peft_config)

                    adapter_args = AdapterArguments(train_adapter=True)
                    setup_adapter_training(encoder_model, adapter_args, f"{active_config['src']}-{active_config['trg']}_energy_adapter")
            else:
                self.model.freeze_encoder()

        # 4) register backward hooks
        if config['energy_model_adapter']:
            energy_adapter_modules = get_adapters(self.model.encoder.model, f"{active_config['src']}-{active_config['trg']}_energy_adapter")
            register_param_hooks(energy_adapter_modules[0], "energy model [ENCODER ADAPTER]")        
        register_param_hooks(self.model.estimator, "energy model [ESTIMATOR]")

        print("energy model trainable params: ", check_trainable_params(self.model))
        
    
    def set_optimizers(self, optim):
        self.optim = optim

    @torch.autocast("cuda")
    def prepare_model_input(self, src_input_ids: Tensor, mt_input_logits: Tensor, mt_input_ids: Tensor) -> Tensor:
        '''
        src_input_ids: 
        X [SEP] [PAD], size = [bsz, src_length]

        mt_input_logits:
        Y [SEP] [PAD], size = [bsz, mt_length, vocab_size]

        mt_input_ids: 
        Y [SEP] [PAD], size  = [bsz, mt_length]
        
        return:
            [CLS] X [SEP] [SEP] Y [SEP] [PAD], size = [bsz, concated length, vocab_size]
        '''

        mt_input_logits.requires_grad_(True)

        batch_size = src_input_ids.size(0)
        vocab_size = self.tokenizer.vocab_size

        cls_tensor = torch.tensor(self.tokenizer.cls_token_id).repeat(batch_size)
        cls_tensor = cls_tensor.to(mt_input_logits.device)
        cls_one_hot: Tensor = F.one_hot(cls_tensor, vocab_size).unsqueeze(1).float()
        cls_one_hot.requires_grad_(True) # so that concated inputs will have requires_grad set to True
        
        sep_tensor = torch.tensor(self.tokenizer.sep_token_id).repeat(batch_size)
        sep_tensor = sep_tensor.to(mt_input_logits.device)
        sep_one_hot: Tensor = F.one_hot(sep_tensor, vocab_size).unsqueeze(1).float()
        sep_one_hot.requires_grad_(True)
    
        src_one_hot: Tensor = F.one_hot(src_input_ids, vocab_size).float()
        src_one_hot = src_one_hot.to(mt_input_logits.device)
        src_one_hot.requires_grad_(True)
            
        concat_input = torch.concat([cls_one_hot, # [batch size, 1, vocab_size]
                                    src_one_hot, # [batch size, src input len, vocab_size]
                                    sep_one_hot, # [batch size, 1, vocab_size]
                                    mt_input_logits, # [batch size, mt input len, vocab_size]
                                    ], dim=1) # [batch size, seq length, vocab_size]

        discrete_concat_input = torch.concat([cls_tensor.unsqueeze(1), # [batch size, 1]
                                    src_input_ids, # [batch size, src input len]
                                    sep_tensor.unsqueeze(1), # [batch size, 1]
                                    mt_input_ids, # [batch size, mt input len]
                                    ], dim=1) # [batch size, seq length]

        attention_mask = discrete_concat_input.ne(self.tokenizer.pad_token_id)
        
        encoder_input = {"input_ids": concat_input, "attention_mask": attention_mask}

        assert concat_input.requires_grad

        return encoder_input
    
    @torch.autocast("cuda")
    def forward(self, src_input_ids: Tensor, mt_input_logits: Tensor, mt_input_ids: Tensor) -> Tensor:
        """Forward function.

        Args:
            src_input_ids: 
            X [SEP] [PAD], size = [bsz, src_length]

            mt_input_logits:
            Y [SEP] [PAD], size = [bsz, mt_length, vocab_size]

            mt_input_ids: 
            Y [SEP] [PAD], size  = [bsz, mt_length]            

        Raises:
            Exception: Invalid model word/sent layer if self.{word/sent}_layer are not
                valid encoder model layers .

        Returns:
            Tensor
        """

        encoder_input = self.prepare_model_input(src_input_ids, mt_input_logits, mt_input_ids)
        input_dist = encoder_input["input_ids"]
        attention_mask = encoder_input["attention_mask"]
        
        encoder: XLMREncoder = self.model.encoder

        input_dist = input_dist.to(encoder.model.get_input_embeddings().weight.dtype)

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

        # Word embeddings used for the word-level classification task
        if self.model.word_level:
            if (
                isinstance(self.model.hparams.word_layer, int)
                and 0 <= self.model.hparams.word_layer < self.model.encoder.num_layers
            ):
                wordemb = encoder_out["all_layers"][self.model.hparams.word_layer]
            else:
                raise Exception(
                    "Invalid model word layer {}.".format(self.model.hparams.word_layer)
                )

        # Word embeddings used for the sentence-level regression task
        if self.model.layerwise_attention:
            sentemb = self.model.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )[:, 0, :]

        elif (
            isinstance(self.model.hparams.sent_layer, int)
            and 0 <= self.model.hparams.sent_layer < self.model.encoder.num_layers
        ):
            sentemb = encoder_out["all_layers"][self.model.hparams.sent_layer][:, 0, :]
        else:
            raise Exception(
                "Invalid model sent layer {}.".format(self.model.hparams.sent_layer)
            )
        
        return self.model.estimator(sentemb).view(-1)

    def set_params_grad(self, mode: bool):
        if mode == False:
            # turn all params off
            self.model.requires_grad_(False)

        else:
            if not self.config['train_energy']: # SEAL-STATIC
                # use all gradients from the base energy model
                self.model.requires_grad_(True)
                return

            # SEAL-DYNAMIC
            self.model.requires_grad_(True)
            if self.config['train_energy_encoder']:
                if self.config['energy_model_adapter']:
                    # in the encoder base model, turn off all non-adapter params
                    self.model.encoder.model.requires_grad_(False)
                    adapters = get_adapters(self.model.encoder.model, f"{self.active_config['src']}-{self.active_config['trg']}_energy_adapter")
                    for adapter in adapters:
                        adapter.requires_grad_(True)
            else:
                self.model.freeze_encoder()