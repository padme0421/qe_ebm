import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer
from datasets import load_dataset
from config import configs
from main import make_data_opus

#translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
#translation_model = translation_model.cuda(0)
tokenizer: MBart50TokenizerFast = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50",
                                                                       src_lang = "en_XX", tgt_lang = "de_DE")
tokenizer.lang_code_to_id
# iwslt17
dataset = load_dataset("iwslt2017", "iwslt2017"+"-"+"en-de")

def measure_length(example_batch, indices=None):
    inputs = [example["en"] for example in example_batch["translation"]]
    targets = [example["de"] for example in example_batch["translation"]]

    model_inputs = tokenizer(
            inputs, text_target=targets, padding=True, truncation=False, return_tensors="pt"
        )
    
    trg_attention_mask = model_inputs.labels.not_equal(tokenizer.pad_token_id)
    
    lengths = {
        "src_length": model_inputs.attention_mask.sum(dim=1), # sum by row
        "label_length": trg_attention_mask.sum(dim=1)
    }

    model_inputs["src_length"] = lengths["src_length"]
    model_inputs["label_length"] = lengths["label_length"]
    
    return model_inputs

print(dataset["train"])
dataset["train"] = dataset["train"].map(measure_length, batched=True, load_from_cache_file=False)

print("first sample: ", dataset["train"][0])

max_src_length = max(dataset["train"]["src_length"])
print("max src length: ", max_src_length)

max_label_length = max(dataset["train"]["label_length"])
print("max label length: ", max_label_length)

long_dataset = dataset["train"].filter(lambda example: example["label_length"] > 50 or example["src_length"] > 50)
print("percentage of long sentences: ", len(long_dataset)/len(dataset["train"]))

# get dataset
'''
active_config = configs["iwslt17_en_de_mbart50_config"]
config = {
    "model": "mbart",
    "tokenizer": "",
    "separate_monolingual_dataset": False,
    "mono_dataset_path": "",
    "ml50_path": "",
    "seed": 231,
    "label_train_size": 1, # arbitrary
    "unlabel_train_size": 1, # arbitrary
    "train_size": 0,
    "val_size": 0,
    "test_size": 0,
    "label_keep": 0.2, # check
    "batch_size": 16,
    "max_length": 200,
    "dir_name": "en_de",
    "function": "supervised_train"
    }

datamodule = make_data_opus(active_config, config, torch.device("cuda"))
datamodule.setup_datacollator(translation_model)
'''