import random
import os
import pandas as pd
import subprocess
import argparse
import yaml
import json
import time

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import datasets
from datasets import load_dataset, DatasetDict, concatenate_datasets
from datasets import Dataset as HFDataset

from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding

from config import configs

from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
import numpy as np
import dotenv
import huggingface_hub
from pathlib import Path
from comet import download_model, load_from_checkpoint
from comet.models import UnifiedMetric

ML50_PATH = dotenv.get_key(".env", "ML50_PATH")
MTNT_PATH = dotenv.get_key(".env", "MTNT_PATH")
ROBUST_WMT20_PATH = dotenv.get_key(".env", "ROBUST_WMT20_PATH")
DOMAIN_DATA_PATH = dotenv.get_key(".env", "DOMAIN_DATA_PATH")
NEWSCRAWL_DATA_PATH = dotenv.get_key(".env", "NEWSCRAWL_DATA_PATH")
WMT19_TEST_DATA = dotenv.get_key(".env", "WMT19_TEST_DATA_PATH")
FEEDBACKMT_DATA_PATH = dotenv.get_key(".env", "FEEDBACKMT_DATA_PATH")

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

@dataclass
class CustomDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        '''
        features: list of dict of list
        return: dict of padded tensors
        '''
     
        all_keys = features[0].keys()
        result = {}
        for key in all_keys:
            if key == 'id':
                result['id'] = torch.tensor([item_features['id'] for item_features in features])
            else:
                if key == "labels":
                    # pad with -100
                    pad_token_id = -100
                else:
                    pad_token_id = self.tokenizer.pad_token_id
            
                value_tensors = [torch.tensor(item_features[key]) for item_features in features]
                max_length = max([t.size(0) for t in value_tensors])
                result[key] = torch.stack([pad_tensor(t, max_length, pad_token_id) for t in value_tensors])

        return result


def get_tokenizer(active_config, config):
    if config['tokenizer'] != "":
        # use separate tokenizer if specified
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
            
    elif active_config['model_name_or_path'].startswith("facebook/mbart") or active_config['model_name_or_path'].startswith("facebook/nllb"):
        # multilingual model -> specify lang codes
        tokenizer = AutoTokenizer.from_pretrained(active_config['model_name_or_path'], src_lang=active_config["model_src_code"], tgt_lang=active_config["model_trg_code"])
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(active_config['model_name_or_path'])

    return tokenizer

class HuggingfaceDataModule(LightningDataModule):
    def __init__(self, active_config, config):
        self.active_config = active_config
        self.config = config

        self.run_id = config['dir_name'].split('/')[-1]

        self.number_columns_keep = ["id", "input_ids", "attention_mask", "labels"]

        self.lang_pair = self.active_config["src"] + "-" + self.active_config["trg"]
        self.model_name_or_path = self.active_config["model_name_or_path"]

        self.tokenizer = get_tokenizer(active_config, config)

        # download comet kiwi model
        # log in to huggingface
        with open(f"{Path.home()}/reinforce/hf_token.txt") as f:
            token = f.readline().strip()        
        huggingface_hub.login(token)
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        self.comet_kiwi_model: UnifiedMetric = load_from_checkpoint(model_path)
        self.comet_kiwi_model.requires_grad_(False)

        self.comet_kiwi_tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")


    def setup_datacollator(self, model):
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model, padding=True)
        self.padding_data_collator = CustomDataCollatorWithPadding(self.tokenizer, model, padding=True)
        
    def load_parallel_data_splits(self, stage):
        # TODO: for test stage, just load test split
        self.parallel_dataset = DatasetDict()
       
        if self.active_config["dataset"] == "opus100":
            if stage != 'fit':
                self.parallel_dataset['validation'] = load_dataset(self.active_config["dataset"], self.lang_pair, split='validation')
                self.parallel_dataset['test'] = load_dataset(self.active_config["dataset"], self.lang_pair, split='test')
            else:
                self.parallel_dataset = load_dataset(self.active_config["dataset"], self.lang_pair)
        
        elif self.active_config["dataset"] == "iwslt2017":
            
            if stage != 'fit':
                self.parallel_dataset['validation'] = load_dataset(self.active_config["dataset"], self.active_config["dataset"]+"-"+self.lang_pair, split='validation')
                self.parallel_dataset['test'] = load_dataset(self.active_config["dataset"], self.active_config["dataset"]+"-"+self.lang_pair, split='test')
            else:
                self.parallel_dataset = load_dataset(self.active_config["dataset"], self.active_config["dataset"]+"-"+self.lang_pair)
            for split in self.parallel_dataset.keys():
                self.parallel_dataset[split] = self.parallel_dataset[split].add_column('id', list(range(len(self.parallel_dataset[split]))))

        elif self.active_config["dataset"] == "wmt19":
            if stage != 'fit':
                self.parallel_dataset['validation'] = load_dataset(self.active_config["dataset"], self.active_config["trg"] + "-" + self.active_config["src"], split='validation')
            else:
                self.parallel_dataset = load_dataset(self.active_config["dataset"], self.active_config["trg"] + "-" + self.active_config["src"])
            # load test split from local
            def gen():
                test_file = f"data/wmt19.{self.active_config['trg']}-{self.active_config['src']}.test"
                with open(test_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        trg = line.split('\t')[0]
                        src = line.split('\t')[1] # english
                        yield {"translation": {
                            f"{self.active_config['src']}": src,
                            f"{self.active_config['trg']}": trg
                        }}
            self.parallel_dataset["test"] = HFDataset.from_generator(gen)
        
        elif self.active_config["dataset"] == "wmt16":
            if stage != 'fit':
                self.parallel_dataset['validation'] = load_dataset(self.active_config["dataset"], self.active_config["trg"] + "-" + self.active_config["src"], split='validation')
                self.parallel_dataset['test'] = load_dataset(self.active_config["dataset"], self.active_config["trg"] + "-" + self.active_config["src"], split='test')
            else:
                self.parallel_dataset = load_dataset(self.active_config["dataset"], self.active_config["trg"] + "-" + self.active_config["src"])
        
        elif self.active_config["dataset"] == "facebook/flores":
            self.parallel_dataset['validation'] = load_dataset(self.active_config["dataset"], self.lang_pair, split='dev')
            self.parallel_dataset['test'] = load_dataset(self.active_config["dataset"], self.lang_pair, split='devtest')
            if stage != "test":
                self.parallel_dataset["train"] = load_dataset("yhavinga/ccmatrix", self.active_config["src_code"].split("_")[0] + "-" + self.active_config["trg_code"].split("_")[0])["train"]
        
        elif self.active_config["dataset"] == "ML50":
            def gen_train():
                train_file_src = f"{ML50_PATH}/ML50/clean/train.{self.active_config['data_src_code']}-{self.active_config['data_trg_code']}.{self.active_config['data_src_code']}"
                if not os.path.exists(train_file_src):
                    train_file_src = f"{ML50_PATH}/ML50/clean/train.{self.active_config['data_trg_code']}-{self.active_config['data_src_code']}.{self.active_config['data_src_code']}"

                train_file_trg = f"{ML50_PATH}/ML50/clean/train.{self.active_config['data_src_code']}-{self.active_config['data_trg_code']}.{self.active_config['data_trg_code']}"
                if not os.path.exists(train_file_trg):
                    train_file_trg = f"{ML50_PATH}/ML50/clean/train.{self.active_config['data_trg_code']}-{self.active_config['data_src_code']}.{self.active_config['data_trg_code']}"

                with open(train_file_src) as f_src:
                    with open(train_file_trg) as f_trg:
                        src_lines = f_src.readlines()
                        trg_lines = f_trg.readlines()
                        for idx, (src_line, trg_line) in enumerate(zip(src_lines, trg_lines)):
                            yield {"id": idx, 
                                "translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}
                
            def gen_test():
                test_file_src = f"{ML50_PATH}/ML50/raw/test.{self.active_config['data_src_code']}-{self.active_config['data_trg_code']}.{self.active_config['data_src_code']}"
                if not os.path.exists(test_file_src):
                    test_file_src = f"{ML50_PATH}/ML50/raw/test.{self.active_config['data_trg_code']}-{self.active_config['data_src_code']}.{self.active_config['data_src_code']}"

                test_file_trg = f"{ML50_PATH}/ML50/raw/test.{self.active_config['data_src_code']}-{self.active_config['data_trg_code']}.{self.active_config['data_trg_code']}"
                if not os.path.exists(test_file_trg):
                    test_file_trg = f"{ML50_PATH}/ML50/raw/test.{self.active_config['data_trg_code']}-{self.active_config['data_src_code']}.{self.active_config['data_trg_code']}"

                with open(test_file_src) as f_src:
                    with open(test_file_trg) as f_trg:
                        src_lines = f_src.readlines()
                        trg_lines = f_trg.readlines()
                        for idx, (src_line, trg_line) in enumerate(zip(src_lines, trg_lines)):
                            yield {"id": idx,
                                "translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}            
    
            def gen_val():
                valid_file_src = f"{ML50_PATH}/ML50/raw/valid.{self.active_config['data_src_code']}-{self.active_config['data_trg_code']}.{self.active_config['data_src_code']}"
                if not os.path.exists(valid_file_src):
                    valid_file_src = f"{ML50_PATH}/ML50/raw/valid.{self.active_config['data_trg_code']}-{self.active_config['data_src_code']}.{self.active_config['data_src_code']}"

                valid_file_trg = f"{ML50_PATH}/ML50/raw/valid.{self.active_config['data_src_code']}-{self.active_config['data_trg_code']}.{self.active_config['data_trg_code']}"
                if not os.path.exists(valid_file_trg):
                    valid_file_trg = f"{ML50_PATH}/ML50/raw/valid.{self.active_config['data_trg_code']}-{self.active_config['data_src_code']}.{self.active_config['data_trg_code']}"

                with open(valid_file_src) as f_src:
                    with open(valid_file_trg) as f_trg:
                        src_lines = f_src.readlines()
                        trg_lines = f_trg.readlines()
                        for idx, (src_line, trg_line) in enumerate(zip(src_lines, trg_lines)):
                            yield {"id": idx,
                                "translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}

            if stage == 'fit':
                self.parallel_dataset["train"] = HFDataset.from_generator(gen_train)
            self.parallel_dataset["validation"] = HFDataset.from_generator(gen_val)
            self.parallel_dataset["test"] = HFDataset.from_generator(gen_test)

        elif self.active_config['dataset'] == 'mtnt':
            def gen_train():
                train_file = f"{MTNT_PATH}/train/train.{self.active_config['src']}-{self.active_config['trg']}.tsv"
                with open(train_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        comment_id, src_line, trg_line = line.split("\t")
                        yield {"translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}
                        
            def gen_val():
                val_file = f"{MTNT_PATH}/valid/valid.{self.active_config['src']}-{self.active_config['trg']}.tsv"
                with open(val_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        comment_id, src_line, trg_line = line.split("\t")
                        yield {"translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}
            def gen_test():
                test_file = f"{MTNT_PATH}/test/test.{self.active_config['src']}-{self.active_config['trg']}.tsv"
                with open(test_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        comment_id, src_line, trg_line = line.split("\t")
                        yield {"translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}
                        
            if stage == 'fit':
                self.parallel_dataset["train"] = HFDataset.from_generator(gen_train)
            self.parallel_dataset["validation"] = HFDataset.from_generator(gen_val)
            self.parallel_dataset["test"] = HFDataset.from_generator(gen_test)

        elif self.active_config["dataset"] == "robust_wmt20":
            def gen_train():
                if (self.active_config['src'] == "en" and self.active_config["trg"] == "ja") or (
                self.active_config["src"] == "ja" and self.active_config["trg"] == "en"):
                    train_file = f"{ROBUST_WMT20_PATH}/few-shot/train.{self.active_config['src']}-{self.active_config['trg']}.tsv"
                    with open(train_file) as f:
                        lines = f.readlines()
                        for line in lines:
                            comment_id, src_line, trg_line = line.split("\t")
                            yield {"translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}
                elif (self.active_config["src"] == "en" and self.active_config["trg"] == "de") or (
                    self.active_config["src"] == "de" and self.active_config["trg"] == "en"):
                    train_file_src = f"{ROBUST_WMT20_PATH}/few-shot/train.{self.active_config['src']}"
                    train_file_trg = f"{ROBUST_WMT20_PATH}/few-shot/train.{self.active_config['trg']}"
                    with open(train_file_src) as f_src:
                        with open(train_file_trg) as f_trg:
                            src_lines = f_src.readlines()
                            trg_lines = f_trg.readlines()
                            for src_line, trg_line in zip(src_lines, trg_lines):
                                yield {"translation": {
                                    f"{self.active_config['src']}": src_line,
                                    f"{self.active_config['trg']}": trg_line
                                }}

                else:
                    print("wrong lang pair for ROBUST_WMT20")
                    exit()
                    
                
            def gen_val():
                if (self.active_config['src'] == "en" and self.active_config["trg"] == "ja") or (
                self.active_config["src"] == "ja" and self.active_config["trg"] == "en"):
                    val_file = f"{ROBUST_WMT20_PATH}/few-shot/valid.{self.active_config['src']}-{self.active_config['trg']}.tsv"
                    with open(val_file) as f:
                        lines = f.readlines()
                        for line in lines:
                            comment_id, src_line, trg_line = line.split("\t")
                            yield {"translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}
                elif (self.active_config["src"] == "en" and self.active_config["trg"] == "de") or (
                    self.active_config["src"] == "de" and self.active_config["trg"] == "en"):
                    valid_file_src = f"{ROBUST_WMT20_PATH}/few-shot/valid.{self.active_config['src']}"
                    valid_file_trg = f"{ROBUST_WMT20_PATH}/few-shot/valid.{self.active_config['trg']}"
                    with open(valid_file_src) as f_src:
                        with open(valid_file_trg) as f_trg:
                            src_lines = f_src.readlines()
                            trg_lines = f_trg.readlines()
                            for src_line, trg_line in zip(src_lines, trg_lines):
                                yield {"translation": {
                                    f"{self.active_config['src']}": src_line,
                                    f"{self.active_config['trg']}": trg_line
                                }}

                else:
                    print("wrong lang pair for ROBUST_WMT20")
                    exit()

            def gen_test():
                # concat all test sets into one
                if self.active_config["src"] == "en" and (self.active_config["trg"] == "de" or 
                                                           self.active_config["trg"] == "ja"):
                    test_file_set1_src = open(f"{ROBUST_WMT20_PATH}/robustness20-3-sets/robustness20-set1-{self.active_config['src']}{self.active_config['trg']}.{self.active_config['src']}")
                    test_file_set1_trg = open(f"{ROBUST_WMT20_PATH}/robustness20-3-sets/robustness20-set1-{self.active_config['src']}{self.active_config['trg']}.{self.active_config['trg']}")
                else:
                    test_file_set1_src = None
                    test_file_set1_trg = None

                if self.active_config["src"] == "ja" or self.active_config["trg"] == "ja":
                    test_file_set2_src = open(f"{ROBUST_WMT20_PATH}/robustness20-3-sets/robustness20-set2-{self.active_config['src']}{self.active_config['trg']}.{self.active_config['src']}")
                    test_file_set2_trg = open(f"{ROBUST_WMT20_PATH}/robustness20-3-sets/robustness20-set2-{self.active_config['src']}{self.active_config['trg']}.{self.active_config['trg']}")
                else:
                    test_file_set2_src = None
                    test_file_set2_trg = None

                if {self.active_config['src']} == "de" and {self.active_config['trg']} == "en":
                    test_file_set3_src = open(f"{ROBUST_WMT20_PATH}/robustness20-3-sets/robustness20-set3-{self.active_config['src']}{self.active_config['trg']}.{self.active_config['src']}")
                    test_file_set3_trg = open(f"{ROBUST_WMT20_PATH}/robustness20-3-sets/robustness20-set3-{self.active_config['src']}{self.active_config['trg']}.{self.active_config['trg']}")
                else:
                    test_file_set3_src = None
                    test_file_set3_trg = None
                
                for f_src, f_trg in zip([test_file_set1_src, test_file_set2_src, test_file_set3_src],
                             [test_file_set1_trg, test_file_set2_trg, test_file_set3_trg]):

                    if not f_src:
                        continue
                    else:
                        src_lines = f_src.readlines()
                        trg_lines = f_trg.readlines()
                        for src_line, trg_line in zip(src_lines, trg_lines):
                            yield {"translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}
            
            if stage == 'fit':
                self.parallel_dataset["train"] = HFDataset.from_generator(gen_train)
            self.parallel_dataset["validation"] = HFDataset.from_generator(gen_val)
            self.parallel_dataset["test"] = HFDataset.from_generator(gen_test)
        
        elif self.active_config["dataset"] == "feedbackmt_highres":
            def gen_train():
                train_file = f"{FEEDBACKMT_DATA_PATH}/sft/wmt20-train-{self.active_config['src']}-{self.active_config['trg']}-nllb.json"
                with open(train_file) as f:
                    lines = f.readlines()
                    for idx, line in enumerate(lines):
                        translation_item = json.loads(line)
                        src_line = translation_item["translation"]["src_sent"]
                        trg_line = translation_item["translation"]["tgt_sent"]
                        yield {"id": idx,
                               "translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}
            
            def gen_val():
                valid_file_src = f"{FEEDBACKMT_DATA_PATH}/raw/wmt21.{self.active_config['src']}-{self.active_config['trg']}.{self.active_config['src']}"
                valid_file_trg = f"{FEEDBACKMT_DATA_PATH}/raw/wmt21.{self.active_config['src']}-{self.active_config['trg']}.{self.active_config['trg']}"
                with open(valid_file_src) as f_src:
                    with open(valid_file_trg) as f_trg:
                        src_lines = f_src.readlines()
                        trg_lines = f_trg.readlines()
                        for idx, (src_line, trg_line) in enumerate(zip(src_lines, trg_lines)):
                            yield {"id": idx,
                                "translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}
            
            def gen_test():
                test_file_src = f"{FEEDBACKMT_DATA_PATH}/raw/wmt22.{self.active_config['src']}-{self.active_config['trg']}.{self.active_config['src']}"
                test_file_trg = f"{FEEDBACKMT_DATA_PATH}/raw/wmt22.{self.active_config['src']}-{self.active_config['trg']}.{self.active_config['trg']}"
                with open(test_file_src) as f_src:
                    with open(test_file_trg) as f_trg:
                        src_lines = f_src.readlines()
                        trg_lines = f_trg.readlines()
                        for idx, (src_line, trg_line) in enumerate(zip(src_lines, trg_lines)):
                            yield {"id": idx,
                                "translation": {
                                f"{self.active_config['src']}": src_line,
                                f"{self.active_config['trg']}": trg_line
                            }}

            if stage == 'fit':
                self.parallel_dataset["train"] = HFDataset.from_generator(gen_train)
            self.parallel_dataset["validation"] = HFDataset.from_generator(gen_val)
            self.parallel_dataset["test"] = HFDataset.from_generator(gen_test)

        return self.parallel_dataset
    
    def load_monolingual_dataset_splits(self, stage):
        self.mono_dataset = DatasetDict()

        if self.config['mono_dataset_path'] == 'bookcorpus':
            self.mono_dataset["train"] = load_dataset(self.config['mono_dataset_path'], split="train")
        
        elif self.config['mono_dataset_path'] == 'newscrawl':
            lines = []
            if self.active_config['src'] == "en":
                train_file = f"{NEWSCRAWL_DATA_PATH}/monolingual.40M.en"
                with open(train_file) as f:
                    lines = f.readlines()

            def gen_train():
                for idx in range(len(lines)-2000):
                    yield {"id": idx, 
                           "translation": {
                            f"{self.active_config['src']}": lines[idx]
                        }}
            
            def gen_val():
                for idx in range(len(lines)-2000, len(lines)-1000):
                    yield {"id": idx, 
                           "translation": {
                            f"{self.active_config['src']}": lines[idx]
                        }}

            def gen_test():
                for idx in range(len(lines)-1000, len(lines)):
                    yield {"id": idx, 
                           "translation": {
                            f"{self.active_config['src']}": lines[idx]
                        }}
            
            if stage == 'fit':
                self.mono_dataset["train"] = HFDataset.from_generator(gen_train)
            self.mono_dataset["validation"] = HFDataset.from_generator(gen_val)
            self.mono_dataset["test"] = HFDataset.from_generator(gen_test)

        elif self.config['mono_dataset_path'].startswith('domain_data'):
            domains = ["it", "koran", "law", "medical"]
            if len(self.config['mono_dataset_path']) > len('domain_data'):  
                target_domain = self.config['mono_dataset_path'].removeprefix('domain_data_')
                if target_domain not in domains:
                    print("target domain not valid")
                    exit() 
                domains = [target_domain]
            
            def gen_train():
                # concat all domains into one
                if self.active_config['src'] == "en":
                    for domain in domains:
                        train_file = f"{DOMAIN_DATA_PATH}/{domain}/train.en"
                        with open(train_file) as f:
                            lines = f.readlines()
                            for line in lines:
                                yield {"translation": {
                                    f"{self.active_config['src']}": line
                                }, "domain": {domain}}

            def gen_val():
                if self.active_config['src'] == "en":
                    for domain in domains:
                        train_file = f"{DOMAIN_DATA_PATH}/{domain}/dev.en"
                        with open(train_file) as f:
                            lines = f.readlines()
                            for line in lines:
                                yield {"translation": {
                                    f"{self.active_config['src']}": line
                                }, "domain": {domain}}

            def gen_test():
                # concat all test sets into one
                if self.active_config['src'] == "en":
                    for domain in domains:
                        train_file = f"{DOMAIN_DATA_PATH}/{domain}/test.en"
                        with open(train_file) as f:
                            lines = f.readlines()
                            for line in lines:
                                yield {"translation": {
                                    f"{self.active_config['src']}": line
                                }, "domain": {domain}}

            if stage == 'fit':
                self.mono_dataset["train"] = HFDataset.from_generator(gen_train)
            self.mono_dataset["validation"] = HFDataset.from_generator(gen_val)
            self.mono_dataset["test"] = HFDataset.from_generator(gen_test)  

        elif self.config['mono_dataset_path'] == "feedbackmt_mono_highres":
            def gen_train():
                train_file = f"{FEEDBACKMT_DATA_PATH}/raft/cc-28k.{self.active_config['src']}-{self.active_config['trg']}.t2t.json"
                with open(train_file) as f:
                    obj = json.load(f)
                    for idx, item in enumerate(obj["instances"]):
                        yield {"id": len(self.parallel_dataset["train"])+idx, # TODO: better way?
                               "translation": {
                                    f"{self.active_config['src']}": item["input"]
                                }}
            if stage == 'fit':
                self.mono_dataset["train"] = HFDataset.from_generator(gen_train)

        return self.mono_dataset

    def control_parallel_dataset_size(self, stage):
        # control size
        if stage == "fit":
            if self.config['separate_monolingual_dataset']:  
                if self.config["label_train_size"] != 0:
                    if len(self.parallel_dataset["train"]) > self.config["label_train_size"]:
                        self.parallel_dataset["train"] = self.parallel_dataset["train"].select(range(self.config["label_train_size"]))
            else:
                if self.config["train_size"] != 0:
                    if len(self.parallel_dataset["train"]) > self.config["train_size"]:
                        self.parallel_dataset["train"] = self.parallel_dataset["train"].select(range(self.config["train_size"]))

        if self.config["val_size"] != 0:
            if len(self.parallel_dataset["validation"]) > self.config["val_size"]:
                self.parallel_dataset["validation"] = self.parallel_dataset["validation"].select(range(self.config["val_size"]))
        if self.config["test_size"] != 0:
            if len(self.parallel_dataset["test"]) > self.config["test_size"]:
                self.parallel_dataset["test"] = self.parallel_dataset["test"].select(range(self.config["test_size"]))
        
        return self.parallel_dataset 

    def control_monolingual_dataset_size(self, stage):
        # control size
        if stage == "fit":
            if self.config["unlabel_train_size"] != 0:
                if len(self.mono_dataset["train"]) > self.config["unlabel_train_size"]:
                    self.mono_dataset["train"] = self.mono_dataset["train"].select(range(self.config["unlabel_train_size"]))

        if "validation" in self.mono_dataset.keys() and self.config["unlabel_val_size"] != 0:
            if len(self.mono_dataset["validation"]) > self.config["unlabel_val_size"]:
                self.mono_dataset["validation"] = self.mono_dataset["validation"].select(range(self.config["unlabel_val_size"]))
        
        if "test" in self.mono_dataset.keys() and self.config["unlabel_test_size"] != 0:
            if len(self.mono_dataset["test"]) > self.config["unlabel_test_size"]:
                self.mono_dataset["test"] = self.mono_dataset["test"].select(range(self.config["unlabel_test_size"]))
        
        return self.mono_dataset

    def prepare_data(self):
        # download and cache dataset on disk
        # load parallel datasets
        if self.active_config["dataset"] == "opus100":
            load_dataset(self.active_config["dataset"], self.lang_pair)
        elif self.active_config["dataset"] == "iwslt2017":
            load_dataset(self.active_config["dataset"], self.active_config["dataset"]+"-"+self.lang_pair)
        elif self.active_config["dataset"] == "wmt19":
            load_dataset(self.active_config["dataset"], self.active_config["trg"] + "-" + self.active_config["src"])
            # test set already downloaded
        elif self.active_config["dataset"] == "wmt16":
            load_dataset(self.active_config["dataset"], self.active_config["trg"] + "-" + self.active_config["src"])
        elif self.active_config["dataset"] == "facebook/flores":
            load_dataset(self.active_config["dataset"], self.lang_pair)
            print("loading ccmatrix for training when using flores as test data")
            load_dataset("yhavinga/ccmatrix", self.active_config["src_code"].split("_")[0] + "-" + self.active_config["trg_code"].split("_")[0])

        # load monolingual datasets
        if self.config['separate_monolingual_dataset']:
            if self.config['mono_dataset_path'] == 'bookcorpus':
                load_dataset(self.config['mono_dataset_path'])

    def setup(self, stage:str):
        # set up parallel dataset
        print("set up parallel dataset")
        # 1) define splits
        self.parallel_dataset: DatasetDict = self.load_parallel_data_splits(stage)

        # 2) shuffle train set
        if stage == 'fit':
            self.parallel_dataset["train"] = self.parallel_dataset["train"].shuffle(self.config['seed'])

        # 3) control size 
        self.parallel_dataset = self.control_parallel_dataset_size(stage)

        # 4) filter by length
        parallel_dataset_splits = ['test', 'validation'] if stage == 'test' else self.parallel_dataset.keys()
        if not self.config['truncation']:
            for split in parallel_dataset_splits:
                original_size = len(self.parallel_dataset[split])
                self.parallel_dataset[split] = self.parallel_dataset[split].filter(
                    lambda example: self.filter_by_length(split, example),
                    batched=True
                )

                print(f"data left in {split} after length filter: ", len(self.parallel_dataset[split])/original_size)
        
        # 5) convert to features
        for split in parallel_dataset_splits:
            self.parallel_dataset[split] = self.parallel_dataset[split].map(
                    lambda example: self.convert_to_features(split, example),
                    batched=True,
                    load_from_cache_file=False
            )

        # 6) select numerical columns
        tmp_data: DatasetDict = self.parallel_dataset.copy()
        for split in parallel_dataset_splits:
            self.parallel_dataset[split] = tmp_data[split].select_columns(self.number_columns_keep)
            self.parallel_dataset[split].set_format(type="torch", output_all_columns=True)

        # set up monolingual dataset
        if self.config['separate_monolingual_dataset']:
            print("set up monolingual dataset")
            # 1) define splits
            self.mono_dataset: DatasetDict = self.load_monolingual_dataset_splits(stage)
            
            if stage == 'fit':
                # 2) shuffle train set
                self.mono_dataset["train"] = self.mono_dataset["train"].shuffle(self.config['seed'])
            # 3) control size
            self.mono_dataset = self.control_monolingual_dataset_size(stage)
            # 4) filter by length
            mono_dataset_splits = self.mono_dataset.keys() 
            if not self.config['truncation']:
                for split in mono_dataset_splits:
                    original_size = len(self.mono_dataset[split])
                    self.mono_dataset[split] = self.mono_dataset[split].filter(
                        lambda example: self.filter_by_length(split, example),
                        batched=True
                    )
                    print(f"data left in {split} after length filter", len(self.mono_dataset[split])/original_size)
            # 5) convert to features
            for split in mono_dataset_splits:
                self.mono_dataset[split] = self.mono_dataset[split].map(
                    self.convert_to_features_mono,
                    batched=True,
                    load_from_cache_file=False
                )
            # 6) select numerical columns
            tmp_data: DatasetDict = self.mono_dataset.copy()
            for split in mono_dataset_splits:
                number_cols_keep = set(tmp_data[split].column_names).intersection(self.number_columns_keep)
                self.mono_dataset[split] = tmp_data[split].select_columns(number_cols_keep)
                self.mono_dataset[split].set_format(type="torch", output_all_columns=True)

        # final dataset
        print("make final dataset")
        self.dataset = DatasetDict()
        if self.config['separate_monolingual_dataset'] and "test" in self.mono_dataset.keys():
            self.dataset["validation"] = self.parallel_dataset["validation"]
            self.dataset["mono_validation"] = self.mono_dataset["validation"]
            
            self.dataset["test"] = self.parallel_dataset["test"]
            self.dataset["mono_test"] = self.mono_dataset["test"]
        else:
            self.dataset["validation"] = self.parallel_dataset["validation"]
            self.dataset["test"] = self.parallel_dataset["test"]

        # set label_train, unlabel_train splits
        if stage == 'fit':
            if self.config['filter_by_alignment_quality']:
                print("filter parallel dataset by alignment quality")
                # filter parallel dataset by alignment quality
                threshold = 0.7

                def calculate_comet_kiwi(model: UnifiedMetric, src_sents: List[str], candidate_trans: List[str]):
                    data = []
                    for i in range(len(src_sents)):
                        item = {
                            "src": src_sents[i],
                            "mt": candidate_trans[i]
                        }
                        data.append(item)
                    model_output = model.predict(data, batch_size=16, gpus=1)
                    return model_output.scores

                def filter_by_alignment_quality(example_batch):
                    # return list of boolean
                    inputs = self.comet_kiwi_tokenizer.batch_decode(example_batch['input_ids'])
                    targets = self.comet_kiwi_tokenizer.batch_decode(example_batch['labels'])
            
                    scores = calculate_comet_kiwi(self.comet_kiwi_model, inputs, targets)

                    if self.config['filter_by_alignment_quality_low']:
                        return [s < threshold for s in scores]
                    else:
                        return [s >= threshold for s in scores]

                train_size_before_quality_filter = len(self.parallel_dataset["train"])

                start_time = time.time()
                self.parallel_dataset["train"] = self.parallel_dataset["train"].filter(filter_by_alignment_quality, 
                                                                            batched=True, batch_size=1000)
                end_time = time.time()
                duration = end_time - start_time
                if self.config['timing_run']:
                    with open(f"timing/{self.run_id}/filter_quality_duration.txt", 'w') as f:
                        f.write(str(duration))

                print(f"after filtering by alignment quality (threshold: {threshold}): {len(self.parallel_dataset['train'])/train_size_before_quality_filter}")

                #label_num = int(train_size_before_quality_filter * self.config["label_keep"])
                
                #self.parallel_dataset["train"] = self.parallel_dataset["train"].select(range(min(label_num, len(self.parallel_dataset["train"]))))

            if self.config['separate_monolingual_dataset']:
                # concat parallel and monolingual data
                self.dataset["label_train"] = self.parallel_dataset["train"]
                self.dataset["unlabel_train"] = concatenate_datasets([self.mono_dataset["train"],
                                                                      self.parallel_dataset["train"].remove_columns(["labels"])])
                self.dataset["unlabel_train"] = self.dataset["unlabel_train"].shuffle(self.config['seed'])

            else:
                # separate parallel data into labeled and unlabeled
                # by random shuffling indices then cutting
                train_size = len(self.parallel_dataset["train"])
                label_num = int(train_size * self.config["label_keep"])
                full_idxs = list(range(train_size))

                random.seed(self.config["seed"])
                random.shuffle(full_idxs)

                labeled_idxs = full_idxs[:label_num]
            
                self.dataset["label_train"] = self.parallel_dataset["train"].select(labeled_idxs)
                
                if self.config["no_unlabeled"]:
                    if self.config["sup_unsup_batch_equal"]:
                        self.dataset["unlabel_train"] = self.dataset["label_train"]
                    else: 
                        self.dataset["unlabel_train"] = self.dataset["label_train"].shuffle(seed=self.config['seed']-1)
                else:
                    self.dataset["unlabel_train"] = self.parallel_dataset["train"]
                    # include labeled data in unlabeled train
            
                self.dataset["unlabel_train"] = self.dataset["unlabel_train"].remove_columns(['labels'])
        
            del self.comet_kiwi_model

        '''
        self.dataset_id2entry = {}
        self.dataset_id2entry["unlabel_train"] = {item["id"].item(): item for item in self.dataset["unlabel_train"]}
        print(self.dataset_id2entry["unlabel_train"].keys())
        '''

        for split in self.dataset.keys():
            print("split: ", split)
            print("split size: ", len(self.dataset[split]))
            sample = self.dataset[split][0]
            print("sample: ", sample)

    def unshuffled_labeled_trainloader(self):
        if self.config['load_by_effective_batch_size']:
            batch_size = self.config["sup_batch_size"] * self.config['accumulate_grad_batches']
        else:
            batch_size = self.config["sup_batch_size"]
        return DataLoader(self.dataset["label_train"], collate_fn=self.padding_data_collator, 
                          batch_size=batch_size,
                          shuffle=False)

    def labeled_trainloader(self):
        if self.config['load_by_effective_batch_size']:
            batch_size = self.config["sup_batch_size"] * self.config['accumulate_grad_batches']
        else:
            batch_size = self.config["sup_batch_size"]
        if self.data_collator:
            return DataLoader(self.dataset["label_train"], collate_fn = self.data_collator, 
                              batch_size=batch_size, 
                              shuffle=(not self.config['sup_unsup_batch_equal']))

    def unshuffled_unlabeled_trainloader(self):
        """
        unsuffled means the order of samples stays the same between epochs
        """

        if self.config['load_by_effective_batch_size']:
            batch_size = self.config["unsup_batch_size"] * self.config['accumulate_grad_batches']
        else:
            batch_size = self.config["unsup_batch_size"]
        return DataLoader(self.dataset["unlabel_train"], collate_fn = self.padding_data_collator, 
                          batch_size=batch_size,
                          shuffle=False)

    def unlabeled_trainloader(self):
        if self.config['load_by_effective_batch_size']:
            batch_size = self.config["unsup_batch_size"] * self.config['accumulate_grad_batches']
        else:
            batch_size = self.config["unsup_batch_size"]
        if self.data_collator:
            return DataLoader(self.dataset["unlabel_train"], collate_fn = self.data_collator, 
                              batch_size=batch_size,
                              shuffle=(not self.config['sup_unsup_batch_equal']))

    def val_dataloader(self):
        if self.data_collator:
            return DataLoader(self.dataset["validation"], collate_fn = self.data_collator, batch_size=self.config["eval_batch_size"])

    def test_dataloader(self):
        if self.data_collator:
            return DataLoader(self.dataset["test"], collate_fn = self.data_collator, batch_size=self.config["eval_batch_size"])

    def mono_val_dataloader(self):
        if self.data_collator:
            if "mono_validation" in self.dataset.keys():
                return DataLoader(self.dataset["mono_validation"], collate_fn = self.data_collator, batch_size=self.config["eval_batch_size"])
            else:
                return None
            
    def mono_test_dataloader(self):
        if self.data_collator:
            if "mono_test" in self.dataset.keys():
                return DataLoader(self.dataset["mono_test"], collate_fn = self.data_collator, batch_size=self.config["eval_batch_size"])
            else:
                return None

    def get_src_text_from_parallel(self, split, example):
        if self.active_config["dataset"] == "facebook/flores":
            if split in ["test", "validation"]:
                return example["sentence_" + self.active_config["src"]]
            else:
                # cc matrix
                return example[self.active_config["src_code"].split("_")[0]]
        else:
            return example[self.active_config["src"]]

    def get_trg_text_from_parallel(self, split, example):
        if self.active_config["dataset"] == "facebook/flores":
            if split in ["test", "validation"]:
                return example["sentence_" + self.active_config["trg"]]
            else:
                # cc matrix
                return example[self.active_config["trg_code"].split("_")[0]]
        else:
            return example[self.active_config["trg"]]

    def filter_by_length(self, split, example_batch):
        if "translation" in example_batch.keys():
            example_batch = example_batch["translation"] # nested
        # example_batch: [{'en': "english sentence"}, {'bn': "bengali sentence"}]
        # return list of boolean
        inputs = [self.get_src_text_from_parallel(split, example) for example in example_batch]
        label_in_batch = self.active_config['trg'] in example_batch[0].keys()
        if label_in_batch:
            print("filter by length: label in batch")
            targets = [self.get_trg_text_from_parallel(split, example) for example in example_batch]
            model_inputs = self.tokenizer(inputs, text_target=targets, padding=True, return_tensors="pt")
            trg_attention_mask = model_inputs.labels.not_equal(self.tokenizer.pad_token_id)
            label_length: Tensor = trg_attention_mask.sum(dim=1) # sum by row
        else:
            print("filter by length: label not in batch")
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        
        src_length: Tensor = model_inputs.attention_mask.sum(dim=1) # sum by row

        is_right_length = src_length.le(self.config['max_length'])
        if label_in_batch:
            is_right_length = is_right_length.logical_and(label_length.le(self.config['max_length']))
        is_right_length = is_right_length.tolist()
        return is_right_length

    def convert_to_features(self, split, example_batch):
        if "translation" in example_batch.keys():
            example_batch = example_batch["translation"] # nested
        inputs = [self.get_src_text_from_parallel(split, example) for example in example_batch]
        targets = [self.get_trg_text_from_parallel(split, example) for example in example_batch]
        
        if self.config['truncation']:
            model_inputs = self.tokenizer(
            inputs, text_target=targets, padding=True, truncation=True, max_length=self.config['max_length']
            )
        else:
            model_inputs = self.tokenizer(
            inputs, text_target=targets, padding=True, truncation=False
            )
        
        model_inputs["src"] = inputs
        model_inputs["trg"] = targets

        return model_inputs

    def convert_to_features_mono(self, example_batch, indices=None):
        if self.config['mono_dataset_path'] == 'bookcorpus':
            src_sents = [example["text"] for example in example_batch]
        else:
            if "translation" in example_batch.keys():
                example_batch = example_batch["translation"] # nested
                src_sents = [example[self.active_config['src']] for example in example_batch]
            
        if self.config['truncation']:
            model_inputs = self.tokenizer(src_sents, padding=True, truncation=True, max_length=self.config['max_length'])
        else:
            model_inputs = self.tokenizer(src_sents, padding=True, truncation=False)
        
        model_inputs["src"] = src_sents
        return model_inputs

def make_data(active_config, config):
    print("Making data")

    print("Downloading dataset")

    opus_datamodule = HuggingfaceDataModule(active_config, config)  
    opus_datamodule.prepare_data()

    if config['function'] == 'test':
        opus_datamodule.setup("test")
    else:
        opus_datamodule.setup("fit")
    
    return opus_datamodule


###### Tests ######
class TestCustomDataset:
    def __init__(self, active_config, config):
        self.hf_data_module: HuggingfaceDataModule = make_data(active_config, config)

def load_configs(config_path: str):
    # load the config file
    with open(config_path, "r") as fp:
        args: dict = yaml.safe_load(fp)

    args["active_config"] = configs[args["active_config"]]
    print(args["active_config"])

    return args

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config = load_configs(args.config_path)
    tester = TestCustomDataset(config["active_config"], config)
