from typing import Dict
from pathlib import Path
import pprint

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import huggingface_hub
from comet import download_model, load_from_checkpoint
from comet.encoders.xlmr import XLMREncoder, XLMRobertaModel
from comet.models.utils import Prediction

from transformers import (MBartForConditionalGeneration, AutoTokenizer, MBart50TokenizerFast)
from transformers.generation.utils import Seq2SeqLMOutput

import wandb

from main import make_data_opus, parse_arguments
from config import configs

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

run = wandb.init()

# download comet kiwi model
# log in to huggingface
with open(f"{Path.home()}/reinforce/hf_token.txt") as f:
    token = f.readline().strip()        
huggingface_hub.login(token)
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
energy_model = load_from_checkpoint(model_path)
energy_model = energy_model.cuda(1)
energy_model.requires_grad_(False)

encoder_model: XLMRobertaModel = energy_model.encoder.model
original_embedding_matrix = encoder_model.get_input_embeddings().weight
# set energy model embedding to linear layer
new_embedding = nn.Linear(original_embedding_matrix.size(0), original_embedding_matrix.size(1), device=encoder_model.device) # [in, out] = [vocab_size, hid_dim]
new_embedding.weight = nn.Parameter(original_embedding_matrix.t().contiguous())
print("embedding weight matrix shape", new_embedding.weight.shape) # [out, in] = [hid_dim, vocab_size]
encoder_model.set_input_embeddings(new_embedding)

# download mbart model (pretrained checkpoint)
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
translation_model = translation_model.cuda(0)
translation_model.requires_grad_(False)
tokenizer: MBart50TokenizerFast = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")

# get dataset
active_config = configs["iwslt17_en_de_mbart50_mmt_config"]
config = {
    "model": "mbart",
    "tokenizer": "",
    "separate_monolingual_dataset": False,
    "mono_dataset_path": "",
    "ml50_path": "",
    "seed": 231,
    "label_train_size": 200, # arbitrary
    "unlabel_train_size": 200, # arbitrary
    "train_size": 4000,
    "val_size": 200,
    "test_size": 200,
    "label_keep": 0.2, # check
    "batch_size": 1,
    "max_length": 30,
    "dir_name": "en_dev0",
    "function": "supervised_train"
    }

energy_model_tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
datamodule = make_data_opus(active_config, config, torch.device("cuda"))

datamodule.setup_datacollator(translation_model)
        
@torch.no_grad()
def pad_tensor(
    tensor: torch.Tensor, length: torch.Tensor, padding_index: int
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

@torch.no_grad()
def energy_model_concat_inputs(src_input_ids: Tensor, mt_input_logits: Tensor, mt_input_ids: Tensor):
    # src_input_ids: 
    # X [SEP] [PAD], size = [bsz, src_length]
    # mt_input_logits:
    # Y [SEP] [PAD], size = [bsz, mt_length, vocab_size]
    # mt_input_ids:
    # Y [SEP] [PAD], size = [bsz, mt_length]
    # result:
    # [CLS] X [SEP] [SEP] Y [SEP] [PAD], size = [bsz, concated length, vocab_size]

    vocab_size = mt_input_logits.size(-1)
    print("vocab_size: ", vocab_size)

    concat_logits = []
    src_input_ids = [
        x.masked_select(x.ne(energy_model_tokenizer.pad_token_id))[:-1] #strip SEP
        for x in src_input_ids.unbind(dim=0)
    ]
    concat_logits.append(src_input_ids)
    
    mt_input_logits = [
        x.masked_select(y.ne(energy_model_tokenizer.pad_token_id).unsqueeze(1)).view(-1, vocab_size)[:-1] #strip SEP
        for x,y in zip(mt_input_logits.unbind(dim=0), mt_input_ids.unbind(dim=0))
    ]
    concat_logits.append(mt_input_logits)

    # Concatenate inputs into a single batch
    batch_size = len(concat_logits[0])
    batch = []
    for i in range(batch_size):
        cls_one_hot = torch.zeros(vocab_size, device=concat_logits[0][i].device)
        print("cls token id: ", energy_model_tokenizer.cls_token_id)
        print("cls one hot shape: ", cls_one_hot.shape)
        cls_one_hot[energy_model_tokenizer.cls_token_id] = 1
        
        print("cls one hot prepared")
        sep_one_hot = torch.zeros(vocab_size, device=concat_logits[0][i].device)
        sep_one_hot[energy_model_tokenizer.sep_token_id] = 1

        concat_logits[1][i] = concat_logits[1][i].to(concat_logits[0][i])
        if concat_logits[1][i].size(0) > energy_model_tokenizer.max_len_single_sentence - 2:
            concat_logits[1][i] = concat_logits[1][i][:energy_model_tokenizer.max_len_single_sentence-2]

        # change ids into one hot vectors
        id_to_one_hot = torch.zeros(concat_logits[0][i].size(0), vocab_size, device=concat_logits[0][i].device)
        id_to_one_hot[torch.arange(concat_logits[0][i].size(0)), concat_logits[0][i]] = 1
        concat_logits[0][i] = id_to_one_hot
        if concat_logits[0][i].size(0) > energy_model_tokenizer.max_len_single_sentence - 2:
            concat_logits[0][i] = concat_logits[1][i][:energy_model_tokenizer.max_len_single_sentence-2]

        print("prepare logits done")

        new_logits = torch.concat([cls_one_hot.unsqueeze(0),
                                    concat_logits[0][i],
                                    sep_one_hot.unsqueeze(0), sep_one_hot.unsqueeze(0),
                                    concat_logits[1][i],
                                    sep_one_hot.unsqueeze(0)]) # [concated length, vocab_size]

        batch.append(new_logits)

    lengths = [t.shape[0] for t in batch]
    max_len = max(lengths)
    padded = [
        pad_tensor(t, max_len, energy_model_tokenizer.pad_token_id) for t in batch
    ]

    lengths = torch.tensor(lengths, dtype=torch.long)
    padded = torch.stack(padded, dim=0).contiguous()

    attention_mask = torch.arange(max_len)[None, :] < lengths[:, None]
    
    # move tensors to energy model 
    padded = padded.to(energy_model.device)
    attention_mask = attention_mask.to(energy_model.device)
    encoder_input = {"input_ids": padded, "attention_mask": attention_mask}

    return encoder_input, lengths, max_len

@torch.autocast("cuda")
@torch.no_grad()
def energy_model_forward(input_dist: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
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
    
    encoder: XLMREncoder = energy_model.encoder

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
    if energy_model.layerwise_attention:
        sentemb = energy_model.layerwise_attention(
            encoder_out["all_layers"], attention_mask
        )[:, 0, :]

    elif (
        isinstance(energy_model.hparams.sent_layer, int)
        and 0 <= energy_model.hparams.sent_layer < energy_model.encoder.num_layers
    ):
        sentemb = encoder_out["all_layers"][energy_model.hparams.sent_layer][:, 0, :]
    else:
        raise Exception(
            "Invalid model sent layer {}.".format(energy_model.hparams.word_layer)
        )

    return Prediction(score=energy_model.estimator(sentemb).view(-1))

all_predictions = {'x_y_relaxed': [], 'x_y': [], 'x_x': []}

for batch in datamodule.labeled_trainloader():
    torch.cuda.empty_cache()

    # 1) (x, x)
    # 2) (x, y)
    # 3) (x, y_relaxed)
    
    # move batch to gpu
    for k,v in batch.items():
        batch[k] = v.to(device=translation_model.device)

    # forward pass, teacher forcing with output ids as the labels 
    outputs: Seq2SeqLMOutput = translation_model(**batch)
    energy_model_vocab_size = energy_model_tokenizer.vocab_size # last token indices: lang id
    
    energy_model_input = {}
    energy_model_input['y_relaxed'] = torch.softmax(outputs.logits[:, 1:, :energy_model_vocab_size], -1) # leave out lang token id
    output_logits_contig = outputs.logits[:, 1:, :energy_model_vocab_size].contiguous()
    print(torch.argmax(output_logits_contig, dim=2).shape)
    energy_model_input['y'] = F.one_hot(torch.argmax(output_logits_contig, dim=2), energy_model_vocab_size)
    # [batch_size, seq_length, vocab_size] -> [batch_size, seq_length] -> [batch_size, seq_length, vocab_size]
    energy_model_input['x'] = F.one_hot(batch.input_ids[:, 1:], energy_model_vocab_size)

    # leave out lang ids
    # first token of input ids, first token of labels
    # turn lang ids to pad
    lang_ids = torch.tensor(tokenizer.additional_special_tokens_ids, device=batch.labels.device)
    mask_tensor =  torch.isin(batch.labels, lang_ids)
    batch.labels = batch.labels.masked_fill_(mask_tensor, tokenizer.pad_token_id)
    
    concat_inputs = {}
    concat_inputs['x_y_relaxed'] = energy_model_concat_inputs(batch.input_ids[:, 1:], energy_model_input['y_relaxed'], batch.labels[:, 1:])
    concat_inputs['x_y'] = energy_model_concat_inputs(batch.input_ids[:, 1:], energy_model_input['y'], batch.labels[:, 1:])
    concat_inputs['x_x'] = energy_model_concat_inputs(batch.input_ids[:, 1:], energy_model_input['x'], batch.labels[:, 1:])

    prediction = {}
    # (x, y_relaxed)
    prediction['x_y_relaxed'] = energy_model_forward(concat_inputs['x_y_relaxed'][0]['input_ids'],
                                    concat_inputs['x_y_relaxed'][0]['attention_mask'])
    prediction['x_y'] = energy_model_forward(concat_inputs['x_y'][0]['input_ids'],
                                    concat_inputs['x_y'][0]['attention_mask'])
    prediction['x_x'] = energy_model_forward(concat_inputs['x_x'][0]['input_ids'],
                                    concat_inputs['x_x'][0]['attention_mask'])
    
    all_predictions['x_y_relaxed'].append(prediction['x_y_relaxed'].score.item())
    all_predictions['x_y'].append(prediction['x_y'].score.item())
    all_predictions['x_x'].append(prediction['x_x'].score.item())


# get score distribution for (x, x), (x, y), (x, y_relaxed)
run.log({'x_x': sum(all_predictions['x_x'])/len(all_predictions['x_x']), 
         'x_y': sum(all_predictions['x_y'])/len(all_predictions['x_y']), 
         'x_y_relaxed': sum(all_predictions['x_y_relaxed'])/len(all_predictions['x_y_relaxed']) 
         })