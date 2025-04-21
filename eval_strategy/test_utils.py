import torch
import s3fs
import shutil

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

import adapters
from adapters.training import setup_adapter_training, AdapterArguments

from pl_module.bleu_callback import BLEUCallback

def get_root_dir(active_config, config):
    if config["score"] == "ensemble":
        score_dir = f"{config['score']}"
        for score in config['score_list']:
            score_dir += f'_{score}'
    else:
        score_dir = f"{config['score']}"

    root_dir = f"s3://reinforce-final-logs/{config['function']}/{config['model']}/{active_config['src']}-{active_config['trg']}/{score_dir}/{config['dir_name']}"
    
    return root_dir

def clean_intermediate_files(active_config, config):
    # clean up intermediate files
    shutil.rmtree(f"{config['dir_name']}")
    if config['score'] == 'fast_align':
        shutil.rmtree(f"fast_align/{active_config['src']}-{active_config['trg']}")
        shutil.rmtree(f"fast_align/{config['dir_name']}")

def build_trainer(active_config, config, pl_module: LightningModule, datamodule, wandb_logger):

    if config['from_local_finetuned']:
        # download from checkpoint
        s3_dir = s3fs.S3FileSystem()
        s3_dir.get(config['checkpoint'], config['dir_name'], recursive=True) 

        output_path = f"{config['dir_name']}/lightning_model.pt"
        convert_zero_checkpoint_to_fp32_state_dict(config['dir_name'], output_path)

        if config['adapter']:
            # add adapter to match finetuned model structure
            adapters.init(pl_module.model)
            adapter_args = AdapterArguments(train_adapter=True, adapter_config="pfeiffer+inv")
            setup_adapter_training(pl_module.model, adapter_args, f"{active_config['src']}-{active_config['trg']}_adapter")

        checkpoint = torch.load(output_path)
    
        wandb_logger.log_text("model info", ["epoch", "global_step", "hyperparams"],
                          [[checkpoint["epoch"], checkpoint["global_step"], checkpoint["hyper_parameters"]['config']]])
        
        model_state_dict = {}
        for key in checkpoint['state_dict']:
            if 'energy_model' in key or 'loss_weights' in key: 
                continue
            else:
                model_state_dict[key] = checkpoint['state_dict'][key]
        pl_module.load_state_dict(model_state_dict)
    
    bleu_callback = BLEUCallback(dirpath=f"{config['dir_name']}/callback", tokenizer=datamodule.tokenizer)
    callbacks = [
        bleu_callback
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=1,
        devices=config["gpu_num"],
        logger = wandb_logger, 
        strategy=config['dist_strategy'],
        callbacks=callbacks,
        precision=16
    )

    return trainer 


def test(
        active_config,
        config, device,
        datamodule, wandb_logger, pl_module_class
        ):
    
    root_dir = get_root_dir(active_config, config)

    pl_module: LightningModule = pl_module_class(active_config, config, device, datamodule.tokenizer, datamodule)

    datamodule.setup_datacollator(pl_module.model)

    trainer = build_trainer(active_config, config, pl_module, datamodule, wandb_logger)
    
    if datamodule.mono_val_dataloader() is not None:
        val_dataloaders = [datamodule.val_dataloader(), datamodule.mono_val_dataloader()]
    else:
        val_dataloaders = [datamodule.val_dataloader()] 

    if datamodule.mono_test_dataloader() is not None:
        test_dataloaders = [datamodule.test_dataloader(), datamodule.mono_test_dataloader()]
    else:
        test_dataloaders = [datamodule.test_dataloader()]

    val_results = trainer.validate(pl_module, val_dataloaders) # to confirm loading checkpoint worked successfully
    test_results = trainer.test(pl_module, test_dataloaders)

    clean_intermediate_files(active_config, config)

    return

