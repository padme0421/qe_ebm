import os 
import uuid 
import re 

from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import AutoModel, AutoModelForSeq2SeqLM
import torch

from pl_module.utils import load_pretranslations
from custom_dataset import HuggingfaceDataModule, make_data
from model.custom_evaluate import calculate_corpus_bleu, calculate_sentence_bleu
from llm.openai_api import OPENAI_Translation_API, lang_dict


def llm_translate(config, wandb_logger: WandbLogger):
    active_config = config['active_config']

    device = torch.device('cuda')
    datamodule = make_data(active_config, config)

    if config['llm'] == 'gpt-4-turbo':
        gpt_translator = OPENAI_Translation_API(model=config['llm'], 
                                            src_lang=lang_dict[active_config['src']],
                                            trg_lang=lang_dict[active_config['trg']])
        model = AutoModel.from_pretrained("facebook/mbart-large-50")
        datamodule.setup_datacollator(model)

    elif config['llm'] == 'nllb':
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        model = model.to(device)
        datamodule.setup_datacollator(model)
        
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
        
        # translate
        if config['llm'] == 'gpt-4-turbo':
            translations = gpt_translator.translate(src_sents, idx==0)
            translations = re.findall(r"""\"[\d]*\":(.*)""", translations)

        elif config['llm'] == 'nllb':
            batch.pop('id')
            # move batch to gpu
            batch['input_ids'] = batch['input_ids'].to(device)

            output = model.generate(batch['input_ids'],
                            forced_bos_token_id=datamodule.tokenizer.lang_code_to_id[active_config['model_trg_code']],
                            num_beams=5, num_return_sequences=1,
                            max_new_tokens = 30,
                            use_cache=False, 
                            return_dict_in_generate=False)
            
            translations = datamodule.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        if idx % 20 == 0:
            print(translations)
        
        for id, translation in zip(sent_idx, translations):
            translation_file.write(f"""{{"{id}": "{translation}"}}\n""")
            
    translation_file.close()
    
    return translation_filename
    

def eval_llm_translation(config, file, wandb_logger: WandbLogger):
    huggingface_datamodule: HuggingfaceDataModule = make_data(config["active_config"], config)
    translations = load_pretranslations(file)
    
    parallel_pairs_src = []
    parallel_pairs_tgt = []
    parallel_pairs_ref = []

    for translation in translations.items():
        id, target = translation
        if config['pretranslate_split'] == 'unlabel_train' or config['pretranslate_split'] == 'label_train':
            split = "train"
        elif config['pretranslate_split'] == 'validation':
            split = "validation"
        elif config['pretranslate_split'] == 'test':
            split = "test"
        item = huggingface_datamodule.parallel_dataset_text[split].filter(lambda x: x["id"] == id)[0]
        parallel_pairs_src.append(item["src"])
        parallel_pairs_tgt.append(target.strip())
        parallel_pairs_ref.append(item["trg"])
        print(calculate_sentence_bleu(parallel_pairs_tgt[-1], [parallel_pairs_ref[-1]], config["active_config"]['trg']))
    
    eval_result = {}
    eval_result["bleu"] = calculate_corpus_bleu(parallel_pairs_tgt, [[x] for x in parallel_pairs_ref], config["active_config"]['trg'])
    wandb_logger.log_metrics({"bleu": eval_result['bleu']})
    
    return eval_result
    
