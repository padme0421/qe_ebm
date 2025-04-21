import argparse 
import yaml
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from llm.llm_translate import llm_translate, eval_llm_translation

from config import configs

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--llm_translation_file", type=str)
    args = parser.parse_args()
    return args

def load_configs(config_path):
    # load the config file
    with open(config_path, "r") as fp:
        args: dict = yaml.safe_load(fp)

    args["active_config"] = configs[args["active_config"]]
    print(args["active_config"])

    return args

if __name__ == "__main__":
    run = wandb.init(project="reinforce-final")
    args = parse_arguments()
    exp_args = load_configs(args.config_path)

    wandb_logger = WandbLogger(project="reinforce-final", log_model='all')
    exp_args["dir_name"] = run.id
    print("save dir: ", exp_args["dir_name"])

    if exp_args['function'] == 'llm_translation_eval':
        eval_result = eval_llm_translation(exp_args, args.llm_translation_file, wandb_logger)    
    elif exp_args['function'] == 'llm_translate':
        translation_file = llm_translate(exp_args, wandb_logger)       
        eval_result = eval_llm_translation(exp_args, translation_file, wandb_logger)
    
