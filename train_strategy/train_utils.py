import os
import shutil
import math
import glob
import torch

import s3fs
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

from adapters.training import setup_adapter_training, AdapterArguments

from pl_module.bleu_callback import BLEUCallback
from train_strategy.offline_trainer import OfflineTrainer
from pl_module.mbart_pl import MBARTSupPL
from pl_module.mbart_ebm_pl import MBARTSsl_EBMPL
from pl_module.mbart_comet_ebm_pl import MBARTSsl_COMETEBMPL
from pl_module.mbart_awesome_align_ebm_pl import MBARTSsl_AlignEBMPL

def get_root_dir(active_config, config):
    if config["score"] == "ensemble":
        score_dir = f"{config['score']}"
        for score in config['score_list']:
            score_dir += f'_{score}'
    else:
        score_dir = f"{config['score']}"

    root_dir = f"s3://reinforce-final-logs/{config['function']}/{config['model']}/{active_config['src']}-{active_config['trg']}/{score_dir}/{config['dir_name']}"
    
    return root_dir

def build_trainer(config, datamodule, wandb_logger):
    
    bleu_callback = BLEUCallback(dirpath=f"{config['dir_name']}/callback", tokenizer=datamodule.tokenizer)

    best_checkpoint = ModelCheckpoint(
        dirpath=f"{config['dir_name']}/checkpoint",  #f"{root_dir}/checkpoint",
        filename="{epoch}-{step}-best",
        monitor='val_comet_kiwi',
        mode='max',
        save_top_k=1
    )

    epoch_end_checkpoint = ModelCheckpoint(
        dirpath=f"{config['dir_name']}/checkpoint",  #f"{root_dir}/checkpoint",
        save_on_train_epoch_end=True
    )

    if config['early_stop_metric'] in 'val_bleu':
        monitor = 'val_bleu'
        mode = 'max'
    elif config['early_stop_metric'] in 'val_comet_kiwi':
        monitor = 'val_comet_kiwi'
        mode = 'max'
    elif config['early_stop_metric'] == 'val_loss':
        monitor = 'val_loss'
        mode = 'min'

    callbacks = [
            best_checkpoint,
            epoch_end_checkpoint,
            bleu_callback,
            EarlyStopping(monitor=monitor, mode=mode, patience=10),
            LearningRateMonitor(logging_interval='step')
        ]

    train_batch_num = math.ceil(len(datamodule.dataset["label_train"]) / (config['gpu_num'] * config['sup_batch_size'] * config['accumulate_grad_batches']))

    # TODO: unify all to manual optimization?
    if config['function'] == 'semisupervised_train_ebm' or 'semisupervised_train_ppo_trl':
        # manual optimization -> accumulate grad batches & gradient clipping not specified here 
        trainer = pl.Trainer(
            accelerator = "gpu",
            num_nodes = 1,
            devices = config["gpu_num"],
            min_epochs = config['min_epoch'] + config['last_epoch'],
            #min_steps=100, # bigger than val_check_interval
            max_epochs = config['max_epoch'] + config['last_epoch'], 
            val_check_interval = min(100, train_batch_num),
            logger = wandb_logger,
            strategy = config['dist_strategy'],
            callbacks = callbacks,
            precision = 16,
            log_every_n_steps = 10
        )
    elif config['function'] == 'semisupervised_train_align_ebm':
        # manual optimization -> accumulate grad batches & gradient clipping not specified here 
        trainer = pl.Trainer(
            accelerator = "gpu",
            num_nodes = 1,
            devices=config["gpu_num"],
            min_epochs = config['min_epoch'] + config['last_epoch'],
            #min_steps=100, # bigger than val_check_interval
            max_epochs = config['max_epoch'] + config['last_epoch'], 
            val_check_interval = min(100, train_batch_num),
            logger = wandb_logger,
            strategy = config['dist_strategy'],
            callbacks = callbacks,
            precision = 'bf16',
            log_every_n_steps = 10,
            detect_anomaly = True
        )
    else:
        # automatic optimization
        trainer = pl.Trainer(
            accelerator = "gpu",
            num_nodes = 1,
            devices = config['gpu_num'],
            min_epochs = config['min_epoch'] + config['last_epoch'],
            min_steps = 100, # bigger than val_check_interval
            max_epochs = config['max_epoch'] + config['last_epoch'],
            val_check_interval = min(100, train_batch_num),
            logger = wandb_logger, 
            strategy = config['dist_strategy'],
            accumulate_grad_batches = config['accumulate_grad_batches'],
            gradient_clip_val = 0.5,
            callbacks = callbacks,
            precision = 16,
            log_every_n_steps = 10
        )

    return trainer

def move_checkpoint_to_s3(config, root_dir):
    s3_dir = s3fs.S3FileSystem()
    best_epoch_step = os.listdir(f"{config['dir_name']}/checkpoint")[0]
    checkpoint_url = f"{root_dir}/checkpoint/{best_epoch_step}"
    s3_dir.put(f"{config['dir_name']}/checkpoint/{best_epoch_step}", f"{root_dir}/checkpoint", recursive=True)
    return checkpoint_url

def clean_intermediate_files(active_config, config):
    # clean up intermediate files
    shutil.rmtree(f"{config['dir_name']}")
    if config['score'] == 'fast_align':
        shutil.rmtree(f"fast_align/{active_config['src']}-{active_config['trg']}")
        shutil.rmtree(f"fast_align/{config['dir_name']}")

def train(
        active_config,
        config, device,
        datamodule, wandb_logger, pl_module_class: LightningModule
        ):
    
    root_dir = get_root_dir(active_config, config)
    
    pl_module: LightningModule = pl_module_class(active_config, config, device, datamodule.tokenizer, datamodule)

    if config['from_local_finetuned']:
        try:
            os.makedirs(config['dir_name'], exist_ok=True)
            output_path = f"{config['dir_name']}/lightning_model.pt"
            convert_zero_checkpoint_to_fp32_state_dict(config['checkpoint'], output_path)
            pl_module = pl_module_class.load_from_checkpoint(output_path, datamodule=datamodule) #local checkpoint
        except Exception as error:
            print(error)
            print("could not load checkpoint from local")
            print("trying to download from s3 bucket")
            # download from checkpoint
            s3_dir = s3fs.S3FileSystem()
            s3_dir.get(config['checkpoint'], config['dir_name'], recursive=True) 

            output_path = f"{config['dir_name']}/lightning_model.pt"
            convert_zero_checkpoint_to_fp32_state_dict(config['dir_name'], output_path)
            pl_module = pl_module_class.load_from_checkpoint(output_path, datamodule=datamodule)

    datamodule.setup_datacollator(pl_module.model)

    trainer = build_trainer(config, datamodule, wandb_logger)

    ## Train ##
    if config['function'] == 'supervised_train':
        trainer.fit(pl_module, datamodule.labeled_trainloader(), datamodule.val_dataloader(), 
                    ckpt_path = config['checkpoint'])
        
    else: # semisupervised_train, semisupervised_train_ppo, semisupervised_train_ebm
        comb_trainloader = CombinedLoader(iterables=
                                      {'label': datamodule.labeled_trainloader(), 
                                       'unlabel': datamodule.unlabeled_trainloader()}, mode="min_size")
        
        print("labeled trainloader size: ", len(datamodule.labeled_trainloader()))
        print("unlabeled trainloader size: ", len(datamodule.unlabeled_trainloader()))
        print("combined trainloader size: ", len(iter(comb_trainloader)))

        if datamodule.mono_val_dataloader() is not None:
            val_dataloaders = [datamodule.val_dataloader(), datamodule.mono_val_dataloader()]
        else:
            val_dataloaders = [datamodule.val_dataloader()]
        trainer.fit(pl_module, comb_trainloader, val_dataloaders, ckpt_path=config['checkpoint']) 
    
    checkpoint_url = move_checkpoint_to_s3(config, root_dir)

    ## Test ##
    if datamodule.mono_test_dataloader() is not None:
        test_dataloaders = [datamodule.test_dataloader(), datamodule.mono_test_dataloader()]
    else:
        test_dataloaders = [datamodule.test_dataloader()] 

    # best checkpoint
    ckpt_path = glob.glob(f"{config['dir_name']}/checkpoint/epoch=*-step=*-best.ckpt")[0]
    output_path = f"{config['dir_name']}/lightning_model.pt"
    convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, output_path)
    pl_module = pl_module_class.load_from_checkpoint(output_path, strict=False, datamodule=datamodule)

    trainer.test(pl_module, dataloaders=test_dataloaders)

    clean_intermediate_files(active_config, config)

    with open("checkpoint_url.txt", "w") as f:
        f.write(checkpoint_url)
    
    return

def offline_train(active_config, config, device,
        datamodule, wandb_logger, pl_module_class):
    
    #config['max_epoch'] = 1
    #config['min_epoch'] = 1
    root_dir = get_root_dir(active_config, config)
    
    # currently only supports mbartssl_ebm module
    pl_module: LightningModule = MBARTSsl_EBMPL(active_config, config, device, datamodule.tokenizer, by_steps=True, warmup=True)
    
    datamodule.setup_datacollator(pl_module.model)

    trainer = build_trainer(config, datamodule, wandb_logger)

    offline_trainer = OfflineTrainer(active_config, config, pl_module, datamodule, trainer)
    offline_trainer.run()
    
    checkpoint_url = move_checkpoint_to_s3(config, root_dir)

    trainer.test(dataloaders=datamodule.test_dataloader())

    clean_intermediate_files(active_config, config)

    with open("checkpoint_url.txt", "w") as f:
        f.write(checkpoint_url)
    
    return