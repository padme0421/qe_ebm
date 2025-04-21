from typing import List
import time
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
import os
import json
from pandas import DataFrame
import re
from datasets import Dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from custom_dataset import HuggingfaceDataModule
from transformers.tokenization_utils import PreTrainedTokenizer

def plot_grad_flow(named_parameters, logger: WandbLogger, save_path: str):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad:
                ave_grads.append(p.grad.abs().mean())
            else:
                ave_grads.append(torch.tensor(0.0))
    
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    
    os.makedirs(save_path.split('/')[0], exist_ok=True)
    plt.savefig(save_path)
    
    ext = "png"
    logger.log_image('grad_flow', [save_path + "." + ext])


def get_label_logits(logits: Tensor, labels: Tensor):
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    seq_len = logits.size(1)
    result = []
    for i, seq_logits in enumerate(logits):
        logits = seq_logits[torch.arange(seq_len), labels[i]]
        result.append(logits)
    result = torch.stack(result)
    return result

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
        if len(tensor.shape) > 1: # more than one dimension
            padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
        else:
            padding = tensor.new(n_padding).fill_(padding_index)
        return torch.cat((tensor, padding), dim=0)

def get_length(t: Tensor, pad_token):
    # t: padded sequence, [batch_size, seq length]
    # pad mask
    mask = t.not_equal(pad_token)
    lengths = mask.sum(dim=1)
    return lengths

def assert_param_no_update(module: nn.Module, initial_norm):
    # Assuming 'model' is your PyTorch model
    parameters = list(module.parameters())
    l2_norm = torch.norm(torch.cat([p.flatten() for p in parameters]), p=2)

    assert l2_norm == initial_norm

def register_param_hooks(module: nn.Module, name: str = None):
    if not name:
        name = module._get_name()
    for p in module.parameters():
        if p.requires_grad:
            p.register_hook(lambda x: print(f"grads accumulated in {name} module"))
            break

def check_trainable_params(model: nn.Module):
    num = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
           num += param.numel()
    return num

def STL(logits):

    index = logits.max(dim=-1, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(2, index, 1.0)
    ret = (y_hard - logits).detach() + logits

    return ret

def get_adapters(model: nn.Module, adapter_name: str) -> List[nn.Module]:
    adapter_dict = model.get_adapter(adapter_name)
    adapter_modules = []
    for layer in adapter_dict.keys():
        for module_location in adapter_dict[layer].keys():
            adapter_module: nn.Module = adapter_dict[layer][module_location]
            adapter_modules.append(adapter_module)
    return adapter_modules

def count_optim_params(optim: torch.optim.Optimizer):
        count = 0
        for param_group in optim.param_groups:
            for p in param_group["params"]:
                count += p.numel()
        return count

class InverseSqrtScheduler(LambdaLR):
    # code borrowed from https://blog.hjgwak.com/posts/learning-rate/

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            
            decay_factor = warmup_steps ** 0.5
            return decay_factor * step ** -0.5

        super(InverseSqrtScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

def load_pretranslations(trans_file_path):
    print("load pretranslations")
    pretranslations = {}
    with open(trans_file_path) as trans_file:
        lines = trans_file.readlines()
        for line in lines:
            try:
                #line_dict = json.loads(line) # many errors in parsing json
                pattern = r'\{"(\d+)"\: "(.*)"\}'
                match = re.match(pattern, line)     
                id = int(match.group(1))           
                sent = match.group(2)

                pretranslations[id] = sent
            except Exception as e:
                print(e)
                print("Error in loading json pretranslations: ", line)

    print(pretranslations.keys())
    return pretranslations

def precompute_corpus_embeddings(encoder: SentenceTransformer, corpus: List[str]) -> dict:
    """
    Given a sentence encoder and a list of sentences
    return dict of {"embeddings": list of embeddings, "duration": duration"}
    """
    # TODO batch encode
    start_time = time.time()
    embeddings = encoder.encode(corpus)
    embeddings = list(embeddings)
    end_time = time.time()
    duration = end_time - start_time
    result = {"embeddings": embeddings, "duration": duration}
    return result

def precompute_similarity(corpus: Dataset) -> Dataset:
    """
    Given an unlabeled dataset, add a new column `neighbor` 
    with the id of the nearest neighbor for each entry 
    and return dict of {"corpus": corpus, "duration": duration}
    """

    start_time = time.time()
    corpus_embeddings = torch.tensor(np.array(corpus['source_embedding']))
    exemplars = semantic_search(corpus_embeddings, corpus_embeddings, top_k=2)

    nearest_neighbors = []
    for query_idx, query_candidates in enumerate(exemplars):
        query = corpus[query_idx]
        best_cand = corpus[query_candidates[0]['corpus_id']] # row num
        second_best_cand = corpus[query_candidates[1]['corpus_id']] # row num
        if (best_cand['id'].item() == query['id'].item()):
            # if the candidate sentence and query sentence are the same, choose the second best candidate
            nearest_neighbors.append(second_best_cand['id'].item())
        else:
            # if the candidate sentence and query sentence are different, choose the best candidate
            nearest_neighbors.append(best_cand['id'].item())
        
    corpus = corpus.add_column('neighbor', nearest_neighbors)
    end_time = time.time()
    duration = end_time - start_time
    result = {"corpus": corpus, "duration": duration}
    return result

def retrieve_unlabeled_batch_precomputed(datamodule: HuggingfaceDataModule, tokenizer: PreTrainedTokenizer, 
                                         config, labeled_batch: dict):
    """ retrieve unlabeled batch by utilizing precomputed nearest neighbors of labeled batch """
    
    start_time = time.time()
    neighbor_src = []
    neighbor_ids = []

    for id in labeled_batch['id']:
        labeled_entry = datamodule.dataset_id2entry['unlabel_train'][id.item()]
        neighbor = datamodule.dataset_id2entry['unlabel_train'][labeled_entry['neighbor'].item()] # neighbor id

        neighbor_src.append(tokenizer.decode(neighbor['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        neighbor_ids.append(neighbor['id'].item())
    
    unlabeled_batch = tokenizer(neighbor_src, return_tensors='pt', padding=True,
                                        truncation=True, max_length=config['max_length'])
    
    if 'labels' in unlabeled_batch:
        unlabeled_batch.pop('labels')

    unlabeled_batch.update({'id': torch.tensor(neighbor_ids)})
    end_time = time.time()
    duration = end_time - start_time
    result = {"batch": unlabeled_batch, "duration": duration}
    return result
