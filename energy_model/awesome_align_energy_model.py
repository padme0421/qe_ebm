from torch import nn
from score.awesome_align_module import AwesomeAlignerXLMR
from awesome_align.modeling import return_extended_attention_mask
from awesome_align.modeling_xlmr import RobertaModel, XLMRobertaForMaskedLM
from pathlib import Path

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from transformers import AutoTokenizer

import huggingface_hub

from peft import LoraConfig, TaskType, get_peft_model
from energy_model.base_energy_model import EnergyModel
from pl_module.utils import register_param_hooks, check_trainable_params

def convert_roberta_embeddings_to_linear(roberta_model: RobertaModel):
    original_word_embedding_matrix = roberta_model.embeddings.word_embeddings.weight
    original_position_embedding_matrix = roberta_model.embeddings.position_embeddings.weight
    original_token_type_embedding_matrix = roberta_model.embeddings.token_type_embeddings.weight

    new_word_embedding = nn.Linear(original_word_embedding_matrix.size(0), original_word_embedding_matrix.size(1)) # [in, out] = [vocab_size, hid_dim]
    new_word_embedding.weight = nn.Parameter(original_word_embedding_matrix.t().contiguous(), requires_grad=True)
    print("word embedding weight matrix shape", new_word_embedding.weight.shape) # [out, in] = [hid_dim, vocab_size]
    roberta_model.embeddings.word_embeddings = new_word_embedding

    new_position_embedding = nn.Linear(original_position_embedding_matrix.size(0), original_position_embedding_matrix.size(1)) # [in, out] = [vocab_size, hid_dim]
    new_position_embedding.weight = nn.Parameter(original_position_embedding_matrix.t().contiguous(), requires_grad=True)
    print("position embedding weight matrix shape", new_position_embedding.weight.shape) # [out, in] = [hid_dim, vocab_size]
    roberta_model.embeddings.position_embeddings = new_position_embedding

    new_token_type_embedding = nn.Linear(original_token_type_embedding_matrix.size(0), original_token_type_embedding_matrix.size(1)) # [in, out] = [vocab_size, hid_dim]
    new_token_type_embedding.weight = nn.Parameter(original_token_type_embedding_matrix.t().contiguous(), requires_grad=True)
    print("token type embedding weight matrix shape", new_token_type_embedding.weight.shape) # [out, in] = [hid_dim, vocab_size]
    roberta_model.embeddings.token_type_embeddings = new_token_type_embedding

    # forward function can stay the same
    return roberta_model

class AwesomeAlign_EnergyModel(EnergyModel):
    def __init__(self, config, active_config):
        super(AwesomeAlign_EnergyModel, self).__init__(config, active_config)
        self.config = config 
        self.active_config = active_config

        # 1) load energy model
        with open(f"{Path.home()}/reinforce/hf_token.txt") as f:
            token = f.readline().strip()        
            huggingface_hub.login(token)

        self.model = AwesomeAlignerXLMR("xlm-roberta-large")
        self.base_model: XLMRobertaForMaskedLM = self.model.model
        self.base_model = self.base_model.bfloat16()
        self.base_model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
        
        # 2) make embeddings linear
        self.base_model.roberta = convert_roberta_embeddings_to_linear(self.base_model.roberta)
        
        # 3) configure adapters & param requires_grad attibutes
        # SEAL STATIC
        if not config['train_energy']:
            self.base_model.requires_grad_(True) # for registering backward hook
            register_param_hooks(self.base_model, "energy model [BASE MODEL]")
            self.base_model.requires_grad_(False)

        # SEAL DYNAMIC
        else:
            if config['train_energy_encoder']:
                if config['energy_model_adapter']:
                    peft_config = LoraConfig(
                        target_modules=['query','value'], task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, 
                        r=4, lora_alpha=4, lora_dropout=0.1
                    )
                    self.base_model = get_peft_model(self.base_model, peft_config)
                else:
                    self.base_model.requires_grad_(True)
            else:
                self.base_model.requires_grad_(False)

        # 4) register backward hooks
        register_param_hooks(self.base_model, "energy model [BASE MODEL]")

        print("energy model trainable params: ", check_trainable_params(self.base_model))

    def encode_seq(self, logits: Tensor, ids: Tensor):
        # logits: [batch_size, seq_len, vocab_size]
        # ids: [batch_size, seq_len]

        input_shape = ids.size()

        batch_size = input_shape[0]
        vocab_size = self.tokenizer.vocab_size

        cls_tensor = torch.tensor(self.tokenizer.cls_token_id).repeat(batch_size)
        cls_tensor = cls_tensor.to(logits.device)
        cls_one_hot: Tensor = F.one_hot(cls_tensor, vocab_size).unsqueeze(1).float()
        cls_one_hot.requires_grad_(True) # so that concated inputs will have requires_grad set to True

        ids = torch.concat([cls_tensor.unsqueeze(1), ids], dim=1)
        logits = torch.concat([cls_one_hot, logits], dim=1)

        attention_mask=(ids != self.tokenizer.pad_token_id) # [batch_size, seq_len]

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = return_extended_attention_mask(attention_mask, next(self.base_model.parameters()).dtype)

        logits = logits.to(self.base_model.get_input_embeddings().weight.dtype)

        embeds = self.base_model.get_input_embeddings().forward(logits)

        outputs = self.base_model.roberta.encoder.forward(
            embeds,
            attention_mask=extended_attention_mask,
            align_layer=8,
        )

        return outputs


    def forward(self, src_input_ids: Tensor, mt_input_logits: Tensor, mt_input_ids: Tensor) -> Tensor:
        """Forward function.

        Args:
            src_input_ids: 
            X [SEP] [PAD], size = [bsz, src_length]

            mt_input_logits:
            Y [SEP] [PAD], size = [bsz, mt_length, vocab_size]

            mt_input_ids: 
            Y [SEP] [PAD], size  = [bsz, mt_length]            

        Returns:
            Tensor [bsz]
        """

        batch_size = src_input_ids.size(0)
        bpelen_src, bpelen_tgt = src_input_ids.size(1)-2, mt_input_ids.size(1)-2
        
        src_logits: Tensor = F.one_hot(src_input_ids, self.tokenizer.vocab_size).float()
        src_logits.requires_grad_(True)

        mt_input_logits.requires_grad_(True)

        outputs_src = self.encode_seq(src_logits, src_input_ids)
        outputs_tgt = self.encode_seq(mt_input_logits, mt_input_ids)

        assert outputs_src.requires_grad
        assert outputs_tgt.requires_grad

        # put cls in front of src, tgt ids
        cls_tensor = torch.tensor(self.tokenizer.cls_token_id).repeat(batch_size)
        cls_tensor = cls_tensor.to(src_input_ids.device)

        src_input_ids = torch.concat([cls_tensor.unsqueeze(1), src_input_ids], dim=1)
        mt_input_ids = torch.concat([cls_tensor.unsqueeze(1), mt_input_ids], dim=1)

        attention_probs_inter = self.base_model.guide_layer(outputs_src, outputs_tgt, src_input_ids, mt_input_ids,
                                                 extraction="softmax", softmax_threshold=0.001,
                                                 output_prob=True)
        
        attention_probs_inter, alignment_probs = attention_probs_inter
        alignment_probs = alignment_probs[:, 0, 1:-1, 1:-1]

        attention_probs_inter = attention_probs_inter.float()
                
        align_scores = []
        attention_probs_inter = attention_probs_inter[:, 0, 1:-1, 1:-1]

        assert alignment_probs.requires_grad

        for idx, attention in enumerate(attention_probs_inter):
            aligns = dict()
            non_zeros = torch.nonzero(attention)
            for i, j in non_zeros:
                word_pair = (i,j)
                prob = alignment_probs[idx, i, j] 
                if not word_pair in aligns:
                    aligns[word_pair] = prob
                else:
                    aligns[word_pair] = max(aligns[word_pair], prob)

            if list(aligns.values()) == []:
                print("no alignments. setting align score to 0")
                align_score = torch.tensor(0.0).cuda().requires_grad_(True)
            else:
                align_score = torch.mean(torch.stack(list(aligns.values())))

            assert align_score.requires_grad

            align_scores.append(align_score)
        
        align_scores = torch.stack(align_scores)
        align_scores = align_scores.to(mt_input_logits.dtype)
        return align_scores
    

    def set_params_grad(self, mode: bool):
        if mode == False:
            # turn all params off
            self.base_model.requires_grad_(False)
        else:
            if not self.config['train_energy']: # SEAL-STATIC
                # use all gradients from the base energy model
                self.base_model.requires_grad_(True)
                return

            # SEAL-DYNAMIC           
            if self.config['train_energy_encoder']:
                if self.config['energy_model_adapter']:
                    self.base_model.requires_grad_(False) # disable all non-adapter layers
                    self.base_model.enable_adapter_layers()
                else:
                    self.base_model.requires_grad_(True)
            else:
                self.base_model.requires_grad_(False)