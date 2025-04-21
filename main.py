import argparse
import os
import regex as re
import glob
import yaml
import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from custom_dataset import HuggingfaceDataModule, make_data
from config import configs
from train_strategy import train_utils
from eval_strategy import test_utils
from pl_module import (
    mbart_pl, mbart_trl_pl, nllb_pl, mt5_pl, bart_pl, cmlm_pl, nambart_pl,
    mbart_comet_ebm_pl, mbart_awesome_align_ebm_pl, nambart_comet_ebm_pl, nambart_awesome_align_ebm_pl
)
import dotenv

SEEDS = [231,42,6]
WORKDIR = dotenv.get_key(".env", "WORKDIR")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_configs(config_path):
    # load the config file
    with open(config_path, "r") as fp:
        args: dict = yaml.safe_load(fp)

    args["active_config"] = configs[args["active_config"]]
    print(args["active_config"])

    return args

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--multi_seed", action="store_true")
    args = parser.parse_args()
    return args

def main(config, wandb_logger):
    print("Start of main program")
    
    active_config = config['active_config']
    
    # global seed
    pl.seed_everything(config['seed'])
    
    # check device status
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda device count", torch.cuda.device_count())
    print("NCCL available", torch.distributed.is_nccl_available())

    '''
        use fixed seed for data loading
    '''

    run_id = config['dir_name'].split('/')[-1]
    if config['timing_run']:
        os.mkdir(f"timing/{run_id}")
    if config['grad_analysis_run']:
        os.mkdir(f"grad_analysis/{run_id}")
    huggingface_datamodule: HuggingfaceDataModule = make_data(active_config, config)

    print_config = {
            "DEVICE": DEVICE.type
        }

    print_config.update(config)
    print(print_config)

    # empty cache before training
    torch.cuda.empty_cache()

    # run supervised or semisupervised training
    # pl_registry[function][model] = pl_module

    pl_registry = {
        "supervised_train": {
            "mbart": mbart_pl.MBARTSupPL,
            "nllb": nllb_pl.NLLBSupPL,
            "mt5": mt5_pl.MT5SupPL,
            "bart": bart_pl.BARTSupPL,
            "cmlm": cmlm_pl.CMLMSupPL,
            "nambart": nambart_pl.NAMBARTSupPL
        },

        "semisupervised_train": {
            "mbart": mbart_pl.MBARTSslPL,
            "nllb": nllb_pl.NLLBSslPL,
            "mt5": mt5_pl.MT5SslPL,
            "bart": bart_pl.BARTSslPL,
            "cmlm": cmlm_pl.CMLMSslPL,
            "nambart": nambart_pl.NAMBARTSslPL
        },

        "semisupervised_train_ppo_trl": {
            "mbart": mbart_trl_pl.MBART_TRL_PpoPL
        },

        "semisupervised_train_ebm": {
            "mbart": mbart_comet_ebm_pl.MBARTSsl_COMETEBMPL,
            "nambart": nambart_comet_ebm_pl.NAMBARTSsl_COMETEBMPL,
            "bart": bart_pl.BARTSsl_EBMPL
        },

        "semisupervised_train_align_ebm": {
            "mbart": mbart_awesome_align_ebm_pl.MBARTSsl_AlignEBMPL,
            "nambart": nambart_awesome_align_ebm_pl.NAMBARTSsl_AwesomeAlignEBMPL,
        },

        "test": { # same as supervised training
            "mbart": mbart_pl.MBARTSupPL,
            "nllb": nllb_pl.NLLBSupPL,
            "mt5": mt5_pl.MT5SupPL,
            "bart": bart_pl.BARTSupPL,
            "cmlm": cmlm_pl.CMLMSupPL,
            "nambart": nambart_pl.NAMBARTSupPL
        }

    }
    
    pl_module_class = pl_registry[config['function']][config['model']]
    if config['function'] == 'test':
       test_utils.test(active_config, config, DEVICE, huggingface_datamodule, wandb_logger, pl_module_class) 
    else:
        if config['offline_generation']:
            train_utils.offline_train(active_config, config, DEVICE, huggingface_datamodule, wandb_logger, pl_module_class)
        else:
            train_utils.train(active_config, config, DEVICE, huggingface_datamodule, wandb_logger, pl_module_class)
        
if __name__ == '__main__':
    args = parse_arguments()

    if args.run_id:
        run = wandb.init(project="reinforce-final", id=args.run_id, resume="must")
    else:
        run = wandb.init(project="reinforce-final")

    exp_args = load_configs(args.config_path)

    if args.multi_seed:
        for seed in SEEDS:
            # new wandb run per seed
            wandb_logger = WandbLogger(project="reinforce-final", log_model='all', id=run.id, resume="must")
            exp_args["dir_name"] = f"{WORKDIR}/{run.id}"
            print("save dir: ", exp_args["dir_name"])
            main(exp_args, wandb_logger)

    else:
        wandb_logger = WandbLogger(project="reinforce-final", log_model='all', id=run.id, resume="must")
        exp_args["dir_name"] = f"{WORKDIR}/{run.id}"
        print("save dir: ", exp_args["dir_name"])
        main(exp_args, wandb_logger)


