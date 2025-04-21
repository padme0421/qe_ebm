import numpy as np

import torch
from torchmetrics.functional.regression.pearson import pearson_corrcoef

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from transformers import PreTrainedTokenizer
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.custom_evaluate import calculate_corpus_bleu, calculate_sentence_bleu, calculate_chrf_pp, calculate_tokenized_bleu, calculate_comet_kiwi

from score.score import Scorer

NUM_BEAMS = 5
        
    
class BLEUCallback(pl.Callback):
    def __init__(self,
                 dirpath, 
                 tokenizer):
        
        self.dirpath = dirpath

        # List[str]
        self.src_sents = []
        # List[List[str]]
        self.src_sents_tokenized = []

        self.src_sents_mono = []
        self.src_sents_tokenized_mono = []

        # List[str]
        self.candidate_trans = []
        # List[List[str]]
        self.candidate_trans_tokenized = []

        self.candidate_trans_mono = []
        self.candidate_trans_tokenized_mono = []      

        # List[List[str]] # each inner list contains one ref stirng
        self.ref_trans = []
        # List[List[List[str]]] # each inner list (depth=1) contains one ref string
        self.ref_trans_tokenized = []

        # per batch
        self.batch_mean_score = []
        self.batch_bleu = []
        self.batch_tokenized_bleu = []
        self.batch_chrf_pp = []

        self.batch_mean_score_mono = []

        # per sample
        self.example_raw_scores = {} 
        self.example_raw_scores_mono = {}
        
        self.example_scaled_scores = {} 
        self.example_scaled_scores_mono = {}
        
        self.example_bleu = [] 
        self.example_tokenized_bleu = [] 
        self.example_chrf_pp = []

        self.tokenizer: PreTrainedTokenizer = tokenizer
        

    def on_validation_epoch_end(self, trainer, pl_module):

        # parallel data 
        total_bleu = calculate_corpus_bleu(self.candidate_trans, self.ref_trans, pl_module.active_config['trg'])
        total_tokenized_bleu = calculate_tokenized_bleu(self.candidate_trans_tokenized, self.ref_trans_tokenized)
        total_chrf_pp = calculate_chrf_pp(self.candidate_trans, self.ref_trans)
        total_comet_kiwi = calculate_comet_kiwi(pl_module.scorer.comet_kiwi_model, self.src_sents, self.candidate_trans)

        # monolingual data
        if pl_module.config['separate_monolingual_dataset']:
            total_comet_kiwi_mono = calculate_comet_kiwi(pl_module.scorer.comet_kiwi_model, self.src_sents_mono, self.candidate_trans_mono)

        # calculate correlation between scores and metrics
        #bleu_cor = pearson_corrcoef(torch.tensor(self.example_scaled_scores['final']), torch.tensor(self.example_bleu))
        #tokenized_bleu_cor = pearson_corrcoef(torch.tensor(self.example_scaled_scores['final']), torch.tensor(self.example_tokenized_bleu))
        #chrf_cor = pearson_corrcoef(torch.tensor(self.example_scaled_scores['final']), torch.tensor(self.example_chrf_pp))
        #bleu_cor = bleu_cor.cuda()
        #tokenized_bleu_cor = tokenized_bleu_cor.cuda()
        #chrf_cor = chrf_cor.cuda()

        # calculate correlation for examples that have bleu over 0
        #bleu_cutoff_indices = [i for i, bleu in enumerate(self.example_bleu) if bleu > 0]
        #example_bleu_cutoff_bleu = [self.example_bleu[i] for i in bleu_cutoff_indices]
        #example_scores_cutoff_bleu = [self.example_scaled_scores['final'][i] for i in bleu_cutoff_indices]

        #pl_module.log('val_bleu_cutoff_rate', len(example_bleu_cutoff_bleu)/len(self.example_bleu), sync_dist=True)
        
        #bleu_cor_cutoff_bleu = pearson_corrcoef(torch.tensor(example_scores_cutoff_bleu), torch.tensor(example_bleu_cutoff_bleu))
        #bleu_cor_cutoff_bleu = bleu_cor_cutoff_bleu.cuda()

        # log metrics
        pl_module.log('val_bleu', total_bleu, sync_dist=True)
        pl_module.log('val_tokenized_bleu', total_tokenized_bleu, sync_dist=True)
        pl_module.log('val_chrf_pp', total_chrf_pp, sync_dist=True)
        pl_module.log('val_comet_kiwi', total_comet_kiwi, sync_dist=True)

        if pl_module.config['separate_monolingual_dataset']:
            pl_module.log('val_comet_kiwi_mono', total_comet_kiwi_mono, sync_dist=True)
       
        # log correlations between score and metrics
        #pl_module.log('val_score_bleu_correlation', bleu_cor, sync_dist=True)
        #pl_module.log('val_score_tokenized_bleu_correlation', tokenized_bleu_cor, sync_dist=True)
        #pl_module.log('val_score_chrf_pp_correlation', chrf_cor, sync_dist=True)
        #pl_module.log('val_score_bleu_correlation_cutoff_bleu', bleu_cor_cutoff_bleu, sync_dist=True)

        # plot correlations between score and metrics
        #bleu_score_data = [[x, y] for (x, y) in zip(self.example_bleu, self.example_scaled_scores['final'])]
        #tokenized_bleu_score_data = [[x, y] for (x, y) in zip(self.example_tokenized_bleu, self.example_scaled_scores['final'])]
        #chrf_score_data = [[x, y] for (x, y) in zip(self.example_chrf_pp, self.example_scaled_scores['final'])]

        
        wandb_logger: WandbLogger = pl_module.logger
        #wandb_logger.log_table(key="val_bleu_score_plot", columns=["bleu", "score"], data=bleu_score_data)
        #wandb_logger.log_table(key="val_tokenized_bleu_score_data", columns=["tokenized_bleu", "score"], data=tokenized_bleu_score_data)
        #wandb_logger.log_table(key="val_chrf_pp_score_plot", columns=["chrf_pp", "score"], data=chrf_score_data)

        # log each example translation, metrics, scores
        if pl_module.config['eval_example_score']:
            val_translations = [[l1,l2,l3,l4,l5,l6,l7,l8] for (l1,l2,l3,l4,l5,l6,l7,l8) in 
                                zip(self.src_sents, self.candidate_trans, self.ref_trans, self.example_bleu, self.example_tokenized_bleu, self.example_chrf_pp, 
                                    self.example_scaled_scores['final'], self.example_raw_scores['final'])]
            wandb_logger.log_text(key="val_translations", columns=["src", "sys", "ref", 
                                                                   "bleu", "tokenized_bleu", "chrf_pp", 
                                                                   "scaled_score", "raw_score"], 
                                                                   data=val_translations)
        
        else:
            val_translations = [[l1,l2,l3,l4,l5,l6] for (l1,l2,l3,l4,l5,l6) in 
                                zip(self.src_sents, self.candidate_trans, self.ref_trans, self.example_bleu, self.example_tokenized_bleu, self.example_chrf_pp)]
            wandb_logger.log_text(key="val_translations", columns=["src", "sys", "ref", 
                                                                   "bleu", "tokenized_bleu", "chrf_pp"], 
                                                                   data=val_translations)


        if pl_module.config['separate_monolingual_dataset']:
            if pl_module.config['eval_example_score']:
                val_translations_mono = [[l1,l2,l3,l4] for (l1,l2,l3,l4) in 
                                     zip(self.src_sents_mono, self.candidate_trans_mono, 
                                         self.example_scaled_scores_mono['final'], self.example_raw_scores_mono['final'])]
                wandb_logger.log_text(key="val_translations_mono", columns=["src_mono", "sys_mono", 
                                                                            "scaled_score_mono", "raw_score_mono"], 
                                                                            data=val_translations_mono)
            else:
                val_translations_mono = [[l1,l2] for (l1,l2) in 
                                     zip(self.src_sents_mono, self.candidate_trans_mono)]
                wandb_logger.log_text(key="val_translations_mono", columns=["src_mono", "sys_mono"], 
                                      data=val_translations_mono)

        # clear records at end of epoch
        self.src_sents = []
        self.src_sents_tokenized = []
        self.candidate_trans = []
        self.candidate_trans_tokenized = []
        self.ref_trans = []
        self.ref_trans_tokenized = []

        if pl_module.config['separate_monolingual_dataset']:
            self.src_sents_mono = []
            self.src_sents_tokenized_mono = []
            self.candidate_trans_mono = []
            self.candidate_trans_tokenized_mono = []

        self.batch_mean_score = []
        self.batch_bleu = []
        self.batch_tokenized_bleu = []
        self.batch_chrf_pp = []
        
        if pl_module.config['separate_monolingual_dataset']:
            self.batch_mean_score_mono = []
    
        self.example_scaled_scores = {}
        self.example_raw_scores = {}

        if pl_module.config['separate_monolingual_dataset']:
            self.example_scaled_scores_mono = {}
            self.example_raw_scores_mono = {}

        self.example_bleu = []
        self.example_tokenized_bleu = []
        self.example_chrf_pp = []

        # adjust learning rate according to comet kiwi
        if pl_module.config['lr_schedule']:
            sch = pl_module.lr_schedulers()
            if isinstance(sch, ReduceLROnPlateau):
                sch.step(total_comet_kiwi)
            else:
                sch.step()
        

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # outputs: outputs from validation_step

        if pl_module.config['model'] == "cmlm":
            src = batch["src_tokens"]
        else:
            src = batch["input_ids"]

        batch_size = len(src)
        
        if "labels" in outputs:
            trg = outputs["labels"]
        else:
            trg = None

        # decode
        # List[str]
        decoded_src_sents = self.tokenizer.batch_decode(src, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # List[List[str]]
        decoded_src_tokens = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in src]
            
        # List[str]
        decoded_preds_sents = self.tokenizer.batch_decode(outputs["preds"],skip_special_tokens=True, clean_up_tokenization_spaces=True)            
        # List[List[str]]
        decoded_preds_tokens = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in outputs["preds"]]

        if trg is not None:
            trg = trg.cpu()
            trg = np.where(trg != -100, trg, self.tokenizer.pad_token_id)
            # List[str]
            decoded_trg_sents = self.tokenizer.batch_decode(trg, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # List[List[str]]
            decoded_trg_tokens = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in trg]
            # List[List[str]] # one ref str in each inner list
            decoded_trg_sents_for_metric_calc = [[s] for s in decoded_trg_sents]
            # List[List[List[str]]] # one ref's token ids in each inner list
            decoded_trg_tokens_for_metric_calc = [[s] for s in decoded_trg_tokens]
        
            
        # store in records
        if trg is not None:
            self.src_sents.extend(decoded_src_sents)
            self.src_sents_tokenized.extend(decoded_src_tokens)

            self.candidate_trans.extend(decoded_preds_sents)
            self.candidate_trans_tokenized.extend(decoded_preds_tokens)
        else:
            self.src_sents_mono.extend(decoded_src_sents)
            self.src_sents_tokenized_mono.extend(decoded_src_tokens)

            self.candidate_trans_mono.extend(decoded_preds_sents)
            self.candidate_trans_tokenized_mono.extend(decoded_preds_tokens)

        if trg is not None:
            self.ref_trans.extend(decoded_trg_sents_for_metric_calc)
            self.ref_trans_tokenized.extend(decoded_trg_tokens_for_metric_calc)

        if batch_idx == 0:
            # TODO check whether padded to longest in batch
            #print("src: ", src)
            #print("trg: ", trg)    
            print("decoded src: ", decoded_src_sents)

            #print("output_preds: ", outputs["preds"][0])
            #print("decoded preds: ", decoded_preds[0])

            if trg is not None:    
                print("decoded trg: ", decoded_trg_sents)
            print("decoded pred sents: ", decoded_preds_sents)

        # calculate metrics
        if trg is not None:
            example_bleu = [calculate_sentence_bleu(pred, ref, pl_module.active_config['trg']) for pred, ref in zip(decoded_preds_sents, decoded_trg_sents_for_metric_calc)]
            batch_bleu = calculate_corpus_bleu(decoded_preds_sents, decoded_trg_sents_for_metric_calc, pl_module.active_config['trg'])
            
            example_tokenized_bleu = [calculate_tokenized_bleu([pred], [ref]) for pred, ref in zip(decoded_preds_tokens, decoded_trg_tokens_for_metric_calc)]
            batch_tokenized_bleu = calculate_tokenized_bleu(decoded_preds_tokens, decoded_trg_tokens_for_metric_calc)

            example_chrf_pp = [calculate_chrf_pp([pred], [ref]) for pred, ref in zip(decoded_preds_sents, decoded_trg_sents_for_metric_calc)]
            batch_chrf_pp = calculate_chrf_pp(decoded_preds_sents, decoded_trg_sents_for_metric_calc)

        if pl_module.config['eval_example_score']:
            scorer : Scorer = pl_module.scorer
            batch_scaled_scores = {}
            batch_raw_scores = {}
            
            if pl_module.config['score'] == "base":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.base_score(outputs["cross_attention"], 
                                                                                                                                self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "uniform":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.uniform_score(src)
            
            elif pl_module.config['score'] == "fast_align":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] =scorer.fast_align_alignment_score(src, outputs["preds"], outputs["cross_attention"], 
                                                                                                                                                self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "awesome_align":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.awesome_align_alignment_score(src, outputs["preds"], outputs["cross_attention"], 
                                                                                                                                                    self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "dep_parse_awesome_align":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.dependency_parse_score_awesome_align(src, outputs["preds"], outputs["cross_attention"],
                                                                                                                                                            self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "dep_parse_base_align":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.dependency_parse_score_base_align(src, outputs["labels"], outputs["cross_attention"], 
                                                                                                                                                        outputs["encoder_attention"], outputs["decoder_attention"], 
                                                                                                                                                        self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "comet_kiwi":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.comet_kiwi_score(src, outputs["preds"], 
                                                                                            outputs["cross_attention"], outputs["encoder_attention"], outputs["decoder_attention"], 
                                                                                            self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "ensemble":
                if "base" in pl_module.config["score_list"]:
                    batch_scaled_scores['base'], batch_raw_scores['base'] = scorer.base_score(
                                                                            outputs["cross_attention"], self.example_scaled_scores.get('base', []), self.example_raw_scores.get('base', [])
                                                                            )
                if "awesome_align" in pl_module.config["score_list"]:
                    batch_scaled_scores['awesome_align'], batch_raw_scores['awesome_align'] = scorer.awesome_align_alignment_score(
                                                                src, outputs["preds"], outputs["cross_attention"], self.example_scaled_scores.get('awesome_align', []), self.example_raw_scores.get('awesome_align', [])
                                                                )
                if "comet_kiwi" in pl_module.config["score_list"]:
                    batch_scaled_scores['comet_kiwi'], batch_raw_scores['comet_kiwi'] = scorer.comet_kiwi_score(src, outputs["preds"], 
                                                                                            outputs["cross_attention"], outputs["encoder_attention"], outputs["decoder_attention"], 
                                                                                            self.example_scaled_scores.get('comet_kiwi', []), self.example_raw_scores.get('comet_kiwi', []))
        
            for score_key, score_values in batch_scaled_scores.items():
                batch_scaled_scores[score_key] = score_values.float()
            for score_key, score_values in batch_raw_scores.items():
                batch_raw_scores[score_key] = score_values.float()

            batch_raw_scores['final'] = torch.mean(torch.stack(tuple(batch_raw_scores.values())), dim=0)
            batch_scaled_scores['final'] = torch.mean(torch.stack(tuple(batch_scaled_scores.values())), dim=0)
    
        if trg is not None:
            self.batch_bleu.append(batch_bleu)
            self.batch_tokenized_bleu.append(batch_tokenized_bleu)
            self.batch_chrf_pp.append(batch_chrf_pp)
            if pl_module.config['eval_example_score']:
                self.batch_mean_score.append(batch_scaled_scores['final'].mean().item())
        else:
            if pl_module.config['eval_example_score']:
                self.batch_mean_score_mono.append(batch_scaled_scores['final'].mean().item())

        if trg is not None:
            self.example_bleu.extend(example_bleu)
            self.example_tokenized_bleu.extend(example_tokenized_bleu)
            self.example_chrf_pp.extend(example_chrf_pp)
        
        if pl_module.config['eval_example_score']:
            for score_key in batch_scaled_scores.keys(): # including 'final'
                if trg is not None:
                    if score_key not in self.example_scaled_scores.keys():
                        self.example_scaled_scores[score_key] = []
                        self.example_raw_scores[score_key] = []
                    self.example_scaled_scores[score_key].extend(batch_scaled_scores[score_key].tolist())
                    self.example_raw_scores[score_key].extend(batch_raw_scores[score_key].tolist())
                else:
                    if score_key not in self.example_scaled_scores_mono.keys():
                        self.example_scaled_scores_mono[score_key] = []
                        self.example_raw_scores_mono[score_key] = []
                    self.example_scaled_scores_mono[score_key].extend(batch_scaled_scores[score_key].tolist())
                    self.example_raw_scores_mono[score_key].extend(batch_raw_scores[score_key].tolist())
            
    def on_test_epoch_end(self, trainer, pl_module):
   
        # parallel data 
        total_bleu = calculate_corpus_bleu(self.candidate_trans, self.ref_trans, pl_module.active_config['trg'])
        total_tokenized_bleu = calculate_tokenized_bleu(self.candidate_trans_tokenized, self.ref_trans_tokenized)
        total_chrf_pp = calculate_chrf_pp(self.candidate_trans, self.ref_trans)
        total_comet_kiwi = calculate_comet_kiwi(pl_module.scorer.comet_kiwi_model, self.src_sents, self.candidate_trans)

        # monolingual data
        if pl_module.config['separate_monolingual_dataset']:
            total_comet_kiwi_mono = calculate_comet_kiwi(pl_module.scorer.comet_kiwi_model, self.src_sents_mono, self.candidate_trans_mono)

        # calculate correlation between scores and metrics
        #bleu_cor = pearson_corrcoef(torch.tensor(self.example_scaled_scores['final']), torch.tensor(self.example_bleu))
        #tokenized_bleu_cor = pearson_corrcoef(torch.tensor(self.example_scaled_scores['final']), torch.tensor(self.example_tokenized_bleu))
        #chrf_cor = pearson_corrcoef(torch.tensor(self.example_scaled_scores['final']), torch.tensor(self.example_chrf_pp))
        #bleu_cor = bleu_cor.cuda()
        #tokenized_bleu_cor = tokenized_bleu_cor.cuda()
        #chrf_cor = chrf_cor.cuda()

        # calculate correlation for examples that have bleu over 0
        #bleu_cutoff_indices = [i for i, bleu in enumerate(self.example_bleu) if bleu > 0]
        #example_bleu_cutoff_bleu = [self.example_bleu[i] for i in bleu_cutoff_indices]
        #example_scores_cutoff_bleu = [self.example_scaled_scores['final'][i] for i in bleu_cutoff_indices]

        #pl_module.log('val_bleu_cutoff_rate', len(example_bleu_cutoff_bleu)/len(self.example_bleu), sync_dist=True)
        
        #bleu_cor_cutoff_bleu = pearson_corrcoef(torch.tensor(example_scores_cutoff_bleu), torch.tensor(example_bleu_cutoff_bleu))
        #bleu_cor_cutoff_bleu = bleu_cor_cutoff_bleu.cuda()

        # log metrics
        pl_module.log('test_bleu', total_bleu, sync_dist=True)
        pl_module.log('test_tokenized_bleu', total_tokenized_bleu, sync_dist=True)
        pl_module.log('test_chrf_pp', total_chrf_pp, sync_dist=True)
        pl_module.log('test_comet_kiwi', total_comet_kiwi, sync_dist=True)

        if pl_module.config['separate_monolingual_dataset']:
            pl_module.log('test_comet_kiwi_mono', total_comet_kiwi_mono, sync_dist=True)
       
        # log correlations between score and metrics
        #pl_module.log('val_score_bleu_correlation', bleu_cor, sync_dist=True)
        #pl_module.log('val_score_tokenized_bleu_correlation', tokenized_bleu_cor, sync_dist=True)
        #pl_module.log('val_score_chrf_pp_correlation', chrf_cor, sync_dist=True)
        #pl_module.log('val_score_bleu_correlation_cutoff_bleu', bleu_cor_cutoff_bleu, sync_dist=True)

        # plot correlations between score and metrics
        #bleu_score_data = [[x, y] for (x, y) in zip(self.example_bleu, self.example_scaled_scores['final'])]
        #tokenized_bleu_score_data = [[x, y] for (x, y) in zip(self.example_tokenized_bleu, self.example_scaled_scores['final'])]
        #chrf_score_data = [[x, y] for (x, y) in zip(self.example_chrf_pp, self.example_scaled_scores['final'])]

        
        wandb_logger: WandbLogger = pl_module.logger
        #wandb_logger.log_table(key="val_bleu_score_plot", columns=["bleu", "score"], data=bleu_score_data)
        #wandb_logger.log_table(key="val_tokenized_bleu_score_data", columns=["tokenized_bleu", "score"], data=tokenized_bleu_score_data)
        #wandb_logger.log_table(key="val_chrf_pp_score_plot", columns=["chrf_pp", "score"], data=chrf_score_data)

        # log each example translation, metrics, scores
        if pl_module.config['eval_example_score']:
            test_translations = [[l1,l2,l3,l4,l5,l6,l7,l8] for (l1,l2,l3,l4,l5,l6,l7,l8) in 
                                zip(self.src_sents, self.candidate_trans, self.ref_trans, self.example_bleu, self.example_tokenized_bleu, self.example_chrf_pp, 
                                    self.example_scaled_scores['final'], self.example_raw_scores['final'])]
            wandb_logger.log_text(key="test_translations", columns=["src", "sys", "ref", 
                                                                   "bleu", "tokenized_bleu", "chrf_pp", 
                                                                   "scaled_score", "raw_score"], 
                                                                   data=test_translations)
        
        else:
            test_translations = [[l1,l2,l3,l4,l5,l6] for (l1,l2,l3,l4,l5,l6) in 
                                zip(self.src_sents, self.candidate_trans, self.ref_trans, self.example_bleu, self.example_tokenized_bleu, self.example_chrf_pp)]
            wandb_logger.log_text(key="test_translations", columns=["src", "sys", "ref", 
                                                                   "bleu", "tokenized_bleu", "chrf_pp"], 
                                                                   data=test_translations)


        if pl_module.config['separate_monolingual_dataset']:
            if pl_module.config['eval_example_score']:
                test_translations_mono = [[l1,l2,l3,l4] for (l1,l2,l3,l4) in 
                                     zip(self.src_sents_mono, self.candidate_trans_mono, 
                                         self.example_scaled_scores_mono['final'], self.example_raw_scores_mono['final'])]
                wandb_logger.log_text(key="test_translations_mono", columns=["src_mono", "sys_mono", 
                                                                            "scaled_score_mono", "raw_score_mono"], 
                                                                            data=test_translations_mono)
            else:
                test_translations_mono = [[l1,l2] for (l1,l2) in 
                                     zip(self.src_sents_mono, self.candidate_trans_mono)]
                wandb_logger.log_text(key="test_translations_mono", columns=["src_mono", "sys_mono"], 
                                      data=test_translations_mono)

        # clear records at end of epoch
        self.src_sents = []
        self.src_sents_tokenized = []
        self.candidate_trans = []
        self.candidate_trans_tokenized = []
        self.ref_trans = []
        self.ref_trans_tokenized = []

        if pl_module.config['separate_monolingual_dataset']:
            self.src_sents_mono = []
            self.src_sents_tokenized_mono = []
            self.candidate_trans_mono = []
            self.candidate_trans_tokenized_mono = []

        self.batch_mean_score = []
        self.batch_bleu = []
        self.batch_tokenized_bleu = []
        self.batch_chrf_pp = []
        
        if pl_module.config['separate_monolingual_dataset']:
            self.batch_mean_score_mono = []
    
        self.example_scaled_scores = {}
        self.example_raw_scores = {}

        if pl_module.config['separate_monolingual_dataset']:
            self.example_scaled_scores_mono = {}
            self.example_raw_scores_mono = {}

        self.example_bleu = []
        self.example_tokenized_bleu = []
        self.example_chrf_pp = []

        
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):

        if pl_module.config['model'] == "cmlm":
            src = batch["src_tokens"]
        else:
            src = batch["input_ids"]

        batch_size = len(src)
        
        if "labels" in outputs:
            trg = outputs["labels"]
        else:
            trg = None

        # decode
        # List[str]
        decoded_src_sents = self.tokenizer.batch_decode(src, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # List[List[str]]
        decoded_src_tokens = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in src]
            
        # List[str]
        decoded_preds_sents = self.tokenizer.batch_decode(outputs["preds"],skip_special_tokens=True, clean_up_tokenization_spaces=True)            
        # List[List[str]]
        decoded_preds_tokens = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in outputs["preds"]]

        if trg is not None:
            trg = trg.cpu()
            trg = np.where(trg != -100, trg, self.tokenizer.pad_token_id)
            # List[str]
            decoded_trg_sents = self.tokenizer.batch_decode(trg, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # List[List[str]]
            decoded_trg_tokens = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in trg]
            # List[List[str]] # one ref str in each inner list
            decoded_trg_sents_for_metric_calc = [[s] for s in decoded_trg_sents]
            # List[List[List[str]]] # one ref's token ids in each inner list
            decoded_trg_tokens_for_metric_calc = [[s] for s in decoded_trg_tokens]
        
            
        # store in records
        if trg is not None:
            self.src_sents.extend(decoded_src_sents)
            self.src_sents_tokenized.extend(decoded_src_tokens)

            self.candidate_trans.extend(decoded_preds_sents)
            self.candidate_trans_tokenized.extend(decoded_preds_tokens)
        else:
            self.src_sents_mono.extend(decoded_src_sents)
            self.src_sents_tokenized_mono.extend(decoded_src_tokens)

            self.candidate_trans_mono.extend(decoded_preds_sents)
            self.candidate_trans_tokenized_mono.extend(decoded_preds_tokens)

        if trg is not None:
            self.ref_trans.extend(decoded_trg_sents_for_metric_calc)
            self.ref_trans_tokenized.extend(decoded_trg_tokens_for_metric_calc)

        if batch_idx == 0:
            # TODO check whether padded to longest in batch
            #print("src: ", src)
            #print("trg: ", trg)    
            print("decoded src: ", decoded_src_sents)

            #print("output_preds: ", outputs["preds"][0])
            #print("decoded preds: ", decoded_preds[0])

            if trg is not None:    
                print("decoded trg: ", decoded_trg_sents)
            print("decoded pred sents: ", decoded_preds_sents)

        # calculate metrics
        if trg is not None:
            example_bleu = [calculate_sentence_bleu(pred, ref, pl_module.active_config['trg']) for pred, ref in zip(decoded_preds_sents, decoded_trg_sents_for_metric_calc)]
            batch_bleu = calculate_corpus_bleu(decoded_preds_sents, decoded_trg_sents_for_metric_calc, pl_module.active_config['trg'])
            
            example_tokenized_bleu = [calculate_tokenized_bleu([pred], [ref]) for pred, ref in zip(decoded_preds_tokens, decoded_trg_tokens_for_metric_calc)]
            batch_tokenized_bleu = calculate_tokenized_bleu(decoded_preds_tokens, decoded_trg_tokens_for_metric_calc)

            example_chrf_pp = [calculate_chrf_pp([pred], [ref]) for pred, ref in zip(decoded_preds_sents, decoded_trg_sents_for_metric_calc)]
            batch_chrf_pp = calculate_chrf_pp(decoded_preds_sents, decoded_trg_sents_for_metric_calc)

        if pl_module.config['eval_example_score']:
            scorer : Scorer = pl_module.scorer
            batch_scaled_scores = {}
            batch_raw_scores = {}
            
            if pl_module.config['score'] == "base":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.base_score(outputs["cross_attention"], 
                                                                                                                                self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "uniform":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.uniform_score(src)
            
            elif pl_module.config['score'] == "fast_align":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] =scorer.fast_align_alignment_score(src, outputs["preds"], outputs["cross_attention"], 
                                                                                                                                                self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "awesome_align":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.awesome_align_alignment_score(src, outputs["preds"], outputs["cross_attention"], 
                                                                                                                                                    self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "dep_parse_awesome_align":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.dependency_parse_score_awesome_align(src, outputs["preds"], outputs["cross_attention"],
                                                                                                                                                            self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "dep_parse_base_align":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.dependency_parse_score_base_align(src, outputs["labels"], outputs["cross_attention"], 
                                                                                                                                                        outputs["encoder_attention"], outputs["decoder_attention"], 
                                                                                                                                                        self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "comet_kiwi":
                batch_scaled_scores[pl_module.config['score']], batch_raw_scores[pl_module.config['score']] = scorer.comet_kiwi_score(src, outputs["preds"], 
                                                                                            outputs["cross_attention"], outputs["encoder_attention"], outputs["decoder_attention"], 
                                                                                            self.example_scaled_scores.get(pl_module.config['score'], []), self.example_raw_scores.get(pl_module.config['score'], []))
            elif pl_module.config['score'] == "ensemble":
                if "base" in pl_module.config["score_list"]:
                    batch_scaled_scores['base'], batch_raw_scores['base'] = scorer.base_score(
                                                                            outputs["cross_attention"], self.example_scaled_scores.get('base', []), self.example_raw_scores.get('base', [])
                                                                            )
                if "awesome_align" in pl_module.config["score_list"]:
                    batch_scaled_scores['awesome_align'], batch_raw_scores['awesome_align'] = scorer.awesome_align_alignment_score(
                                                                src, outputs["preds"], outputs["cross_attention"], self.example_scaled_scores.get('awesome_align', []), self.example_raw_scores.get('awesome_align', [])
                                                                )
                if "comet_kiwi" in pl_module.config["score_list"]:
                    batch_scaled_scores['comet_kiwi'], batch_raw_scores['comet_kiwi'] = scorer.comet_kiwi_score(src, outputs["preds"], 
                                                                                            outputs["cross_attention"], outputs["encoder_attention"], outputs["decoder_attention"], 
                                                                                            self.example_scaled_scores.get('comet_kiwi', []), self.example_raw_scores.get('comet_kiwi', []))
        
            for score_key, score_values in batch_scaled_scores.items():
                batch_scaled_scores[score_key] = score_values.float()
            for score_key, score_values in batch_raw_scores.items():
                batch_raw_scores[score_key] = score_values.float()

            batch_raw_scores['final'] = torch.mean(torch.stack(tuple(batch_raw_scores.values())), dim=0)
            batch_scaled_scores['final'] = torch.mean(torch.stack(tuple(batch_scaled_scores.values())), dim=0)
    
        if trg is not None:
            self.batch_bleu.append(batch_bleu)
            self.batch_tokenized_bleu.append(batch_tokenized_bleu)
            self.batch_chrf_pp.append(batch_chrf_pp)
            if pl_module.config['eval_example_score']:
                self.batch_mean_score.append(batch_scaled_scores['final'].mean().item())
        else:
            if pl_module.config['eval_example_score']:
                self.batch_mean_score_mono.append(batch_scaled_scores['final'].mean().item())

        if trg is not None:
            self.example_bleu.extend(example_bleu)
            self.example_tokenized_bleu.extend(example_tokenized_bleu)
            self.example_chrf_pp.extend(example_chrf_pp)
        
        if pl_module.config['eval_example_score']:
            for score_key in batch_scaled_scores.keys(): # including 'final'
                if trg is not None:
                    if score_key not in self.example_scaled_scores.keys():
                        self.example_scaled_scores[score_key] = []
                        self.example_raw_scores[score_key] = []
                    self.example_scaled_scores[score_key].extend(batch_scaled_scores[score_key].tolist())
                    self.example_raw_scores[score_key].extend(batch_raw_scores[score_key].tolist())
                else:
                    if score_key not in self.example_scaled_scores_mono.keys():
                        self.example_scaled_scores_mono[score_key] = []
                        self.example_raw_scores_mono[score_key] = []
                    self.example_scaled_scores_mono[score_key].extend(batch_scaled_scores[score_key].tolist())
                    self.example_raw_scores_mono[score_key].extend(batch_raw_scores[score_key].tolist())
            