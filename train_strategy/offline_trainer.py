'''
- grow:
    - generate new translations for each source sentence in labeled data + unlabeled data
    - D_grow = D_label(x, y) + D_unlabel(x, y_hat) + D_new_label(x, y_hat_pos, y_hat_neg)
    - annotate translations with reward
- improve (each improve step)
    - update energy model with margin loss on D_label & D_new_label
    - update NMT model with
        - MLE training on D_label & D_new_label (positive)
        - energy based training on D_unlabel(x,y_hat)
'''
import math
import shutil

import torch
from torch import Tensor
import torch.nn.functional as F
from comet.models.utils import Prediction
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from custom_dataset import HuggingfaceDataModule
from pl_module.mbart_pl import MBARTSupPL
from pl_module.mbart_ebm_pl import MBARTSsl_EBMPL
from pl_module.bleu_callback import BLEUCallback

def build_trainer(config, datamodule, wandb_logger):
    
    bleu_callback = BLEUCallback(dirpath=f"{config['dir_name']}/callback",tokenizer=datamodule.tokenizer)

    model_checkpoint = ModelCheckpoint(
        dirpath=f"{config['dir_name']}/checkpoint",  #f"{root_dir}/checkpoint",
        monitor='val_comet_kiwi',
        mode='max',
        save_top_k=1,
        save_on_train_epoch_end=False
    )

    callbacks = [
            model_checkpoint,
            bleu_callback,
            EarlyStopping(monitor='val_comet_kiwi', mode='max', patience=10)
        ]

    train_batch_num = math.ceil(len(datamodule.dataset["label_train"]) / (config['batch_size'] * config['accumulate_grad_batches']))

    # TODO: unify all to manual optimization?
    if config['function'] == 'semisupervised_train_ebm':
        # manual optimization -> accumulate grad batches & gradient clipping not specified here 
        trainer = pl.Trainer(
            accelerator="gpu",
            num_nodes=1,
            devices=config["gpu_num"],
            min_epochs=config['min_epoch'],
            #min_steps=100, # bigger than val_check_interval
            max_epochs = config['max_epoch'], 
            val_check_interval=min(100, train_batch_num),
            logger = wandb_logger, 
            strategy = config['dist_strategy'],
            callbacks=callbacks,
            precision=16,
            log_every_n_steps=10
        )
    else:
        # automatic optimization
        trainer = pl.Trainer(
            accelerator="gpu",
            num_nodes=1,
            devices=config['gpu_num'],
            min_epochs=config['min_epoch'],
            min_steps=100, # bigger than val_check_interval
            max_epochs = config['max_epoch'],
            val_check_interval=min(100, train_batch_num),
            logger = wandb_logger, 
            strategy=config['dist_strategy'],
            accumulate_grad_batches=config['accumulate_grad_batches'],
            gradient_clip_val=0.5,
            callbacks=callbacks,
            precision=16,
            log_every_n_steps=10
        )

    return trainer

class OfflineTrainer:
    def __init__(self, active_config, config, pl_module: MBARTSsl_EBMPL, 
                 datamodule: HuggingfaceDataModule, trainer: Trainer):
        self.active_config = active_config
        self.config = config
        self.pl_module = pl_module
        self.datamodule = datamodule
        self.pl_trainer = trainer

    def warmup(self):
        # run for one epoch
        self.pl_trainer.fit(self.pl_module, self.datamodule.labeled_trainloader(), self.datamodule.val_dataloader())
       
    @torch.autocast("cuda")
    @torch.no_grad()
    def score(self, src: Tensor, labels: Tensor, hypotheses_num: int):
        print("score function: start")
        # get scores
        vocab_size = self.pl_module.energy_model_tokenizer.vocab_size

        repeated_src = src.repeat(hypotheses_num, 1)
        
        # move tensors to energy model device
        print("moving tensors: start")
        self.pl_module.energy_model.cuda()
        print("energy model device: ", self.pl_module.energy_model.device)
        repeated_src = repeated_src.to(self.pl_module.energy_model.device)
        labels = labels.to(self.pl_module.energy_model.device)
        print("moving tensors: end")

        print("energy model concat inputs: start")
        concat_inputs = self.pl_module.prepare_energy_model_input(repeated_src[:, 1:], # leave out lang id
                                                            F.one_hot(labels[:, 1:], vocab_size), # leave out lang id
                                                            labels[:, 1:]) # leave out lang id
        print("energy model concat inputs: end")

        print("energy model forward: start")
        prediction: Prediction = self.pl_module.energy_model_forward(
                                    concat_inputs['concat_input']['input_ids'],
                                    concat_inputs['concat_input']['attention_mask']
                                    )
        print("energy model forward: end")

        print("score function: end")
        return prediction['score']

    @torch.no_grad()
    @torch.autocast("cuda")
    def grow_step(self):
        torch.cuda.empty_cache()
        print("grow step")

        d_label = {'translation': []}
        d_label_pos = {'translation': []}
        d_label_neg = {'translation': []}
        d_unlabel = {'translation': []}

        self.pl_module.num_hypotheses = 5
        labeled_translations = self.pl_trainer.predict(self.pl_module, self.datamodule.unshuffled_labeled_trainloader())
        labeled_translations = [self.pl_module.remove_incompatible_ids(trans_batch.clone()) for trans_batch in labeled_translations]
            
        # get rewards
            
        # rank with rewards, get negative, positive samples
        for batch_idx, (batch, trans_batch) in enumerate(zip(self.datamodule.unshuffled_labeled_trainloader(), labeled_translations)):
            log_src = self.pl_module.tokenizer.batch_decode(batch['input_ids'])
            log_pred = self.pl_module.tokenizer.batch_decode(trans_batch)
            batch['labels'] = torch.where(batch['labels'] != -100, batch['labels'], self.pl_module.tokenizer.pad_token_id)
            log_label = self.pl_module.tokenizer.batch_decode(batch['labels'])
            
            self.pl_module.logger.log_text(key="grow_step", columns=["src", "prediction", "labels"],
                                           data=[[l1,l2,l3] for (l1,l2,l3) in zip(log_src, log_pred, log_label)])

            batch_size = len(batch['input_ids'])

            max_length = trans_batch.size(-1)
            trans_batch = trans_batch.view(batch_size, -1, max_length)

            '''
            scores = self.score(batch['input_ids'], trans_batch, 5)
            scores = scores.view(batch_size, -1) # batch_size, hypothesis num

            sorted_indices = torch.argsort(scores, dim=1, descending=True) # batch_size, hypothesis num (sorted per batch)

            pos_samples = [] # list of list of ints
            neg_samples = []

            max_length = trans_batch.size(-1)
            
            for i, hyps in enumerate(trans_batch):
                # hyps: [hypothesis num, seq length]
                pos_sample = hyps[sorted_indices[i][0]]
                neg_sample = hyps[sorted_indices[i][-1]]
                print("pos_sample: ", pos_sample)
                print("neg_sample: ", neg_sample)

                pos_samples.append(pos_sample.tolist())
                neg_samples.append(neg_sample.tolist())
            
            d_label_pos['translation'].extend(pos_samples)
            d_label_neg['translation'].extend(neg_samples)
            '''
            d_label['translation'].extend(trans_batch.tolist())
        
        self.datamodule.dataset["label_train"] = self.datamodule.dataset["label_train"].remove_columns(['sys']) #'sys_pos', 'sys_neg'])
        self.datamodule.dataset["label_train"] = self.datamodule.dataset["label_train"].add_column('sys', d_label['translation'])
        #self.datamodule.dataset["label_train"] = self.datamodule.dataset["label_train"].add_column('sys_pos', d_label_pos['translation'])
        #self.datamodule.dataset["label_train"] = self.datamodule.dataset["label_train"].add_column('sys_neg', d_label_neg['translation'])
        
        # list of batch tensors
        self.pl_module.num_hypotheses = 1
        unlabeled_translations = self.pl_trainer.predict(self.pl_module, self.datamodule.unshuffled_unlabeled_trainloader())
        unlabeled_translations = [self.pl_module.remove_incompatible_ids(trans_batch.clone()) for trans_batch in unlabeled_translations]

        for batch in unlabeled_translations:
            # batch.tolist(): list of list of ints
            d_unlabel['translation'].extend(batch.tolist())
        
        self.datamodule.dataset["unlabel_train"] = self.datamodule.dataset["unlabel_train"].remove_columns('sys')
        self.datamodule.dataset["unlabel_train"] = self.datamodule.dataset["unlabel_train"].add_column('sys', d_unlabel['translation'])

        #self.pl_trainer.save_checkpoint(f"{self.config['dir_name']}/plus_energy_checkpoint")
        

    def improve_step(self):
        # - update energy model with margin loss on D_label & D_new_label
        # - update NMT model with
        #   - MLE training on D_label & D_new_label (positive)
        #   - energy based training on D_unlabel(x,y_hat)
        
        print("improve step")
        torch.cuda.empty_cache()

        comb_trainloader = CombinedLoader(iterables=
                                      {'label': self.datamodule.labeled_trainloader(), 
                                       'unlabel': self.datamodule.unlabeled_trainloader()})
        
        self.pl_trainer.fit(self.pl_module, comb_trainloader, self.datamodule.val_dataloader())
            
    def run(self):
        self.pl_module.configure_optimizers() # so that initialized optimizer states are saved in first grow step

        label_train_size = len(self.datamodule.dataset["label_train"])
        unlabel_train_size = len(self.datamodule.dataset["unlabel_train"])

        self.datamodule.dataset["label_train"] = self.datamodule.dataset["label_train"].add_column('sys', [list()] * label_train_size)
        #self.datamodule.dataset["label_train"] = self.datamodule.dataset["label_train"].add_column('sys_pos', [list()] * label_train_size)
        #self.datamodule.dataset["label_train"] = self.datamodule.dataset["label_train"].add_column('sys_neg', [list()] * label_train_size)
        self.datamodule.dataset["unlabel_train"] = self.datamodule.dataset["unlabel_train"].add_column('sys', [list()] * unlabel_train_size)

        self.warmup()

        for grow_step in range(self.config['grow_steps']):
            self.grow_step()
            for improve_step in range(self.config['improve_steps']):
                self.improve_step()

        