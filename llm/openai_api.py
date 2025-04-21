# adapted from AttetionX openapi_api.py

import os
import openai
#import tiktoken
import dotenv
from config import configs
import argparse
import yaml
from custom_dataset import make_data
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import re
import uuid
import glob

openai.api_key = dotenv.get_key(".env", 'OPENAI_KEY')

lang_dict = {"en": "English", "de": "German", "zh": "Chinese", "bn": "Bengali", "mr": "Marathi"}

class OPENAI_Translation_API:
    def __init__(self, model="gpt-4", src_lang="", trg_lang=""):
        self.model = model
        self.system_message = f"You are a translator."
        self.prompt_prefix = f"""Translate each of the following source sentences from {src_lang} to {trg_lang}, 
                                and print one translation per line in the following format:
                                "<source sentence number>": <translation> \n
                                Do not include more than one translation per source sentence.\n
                                There must be exactly the same number of translations as the number of source sentences.\n
                                For example, given three source sentences, the translations should look like the following.\n
                                "1": <translation for source sentence 1>\n
                                "2": <translation for source sentence 2>\n
                                "3": <translation for source sentence 3>\n
                                """

    def translate(self, src_sents, verbose = False):
        messages = [{"role": "system", "content": self.system_message}]
        batch = ""
        for sent in src_sents:
            batch += "source: " + sent + "\n"

        messages.append(
            {"role": "user", "content": self.prompt_prefix + batch}
        )

        if verbose:
            print(messages)
        
        try:
            response = openai.ChatCompletion.create(model=self.model, messages=messages, temperature=0.8).choices[0].message.content
            response = response + "\n"
        except Exception:
            print("Error in LLM translation")
            response = ""
            for i in range(len(src_sents)):
                response += f"""{{"target": "ERROR"}}\n"""

        return response

def main(config):
    active_config = config['active_config']

    datamodule = make_data(active_config, config)

    model = AutoModel.from_pretrained("facebook/mbart-large-50")
    datamodule.setup_datacollator(model)

    llm_translator = OPENAI_Translation_API(model=config['llm'], 
                                            src_lang=lang_dict[active_config['src']],
                                            trg_lang=lang_dict[active_config['trg']])
    
    filename_temp = f"{active_config['dataset']}_{active_config['src']}_{active_config['trg']}_{config['llm']}_{config['pretranslate_split']}-{active_config['trg']}.jsonl"
    
    if os.path.isfile(filename_temp):
        root, ext = os.path.splitext(filename_temp)
        translation_filename = root + str(uuid.uuid1()) + ext
    else:
        translation_filename = filename_temp

    translation_file = open(translation_filename, mode='w')
    
    if config['pretranslate_split'] == 'unlabel_train':
        dataloader = datamodule.unshuffled_unlabeled_trainloader()
    elif config['pretranslate_split'] == 'label_train':
        dataloader = datamodule.unshuffled_labeled_trainloader()
    elif config['pretranslate_split'] == 'validation':
        dataloader = datamodule.val_dataloader()
    elif config['pretranslate_split'] == 'test':
        dataloader = datamodule.test_dataloader()

    for idx, batch in enumerate(dataloader):
        sent_idx = batch['id'].tolist()
        print(sent_idx)
        src_sents = datamodule.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        translations = llm_translator.translate(src_sents, idx==0)
        if idx % 20 == 0:
            print(translations)
        translations = re.findall(r"""\"[\d]*\":(.*)""", translations)
        for id, translation in zip(sent_idx, translations):
            translation_file.write(f"""{{"{id}": "{translation}"}}\n""")
    
    translation_file.close()

    

def load_configs(config_path):
    # load the config file
    with open(config_path, "r") as fp:
        args: dict = yaml.safe_load(fp)

    args["active_config"] = configs[args["active_config"]]
    print(args["active_config"])

    return args

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    exp_args = load_configs(args.config_path)
    main(exp_args)