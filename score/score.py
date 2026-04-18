from typing import Tuple
import torch
from torch import Tensor
import torch
import subprocess
import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
import matplotlib.pyplot as plt
from comet import download_model, load_from_checkpoint
from comet.models import UnifiedMetric
import huggingface_hub
from pathlib import Path

class Scorer:
    def __init__(self, active_config, config, tokenizer: PreTrainedTokenizer = None, src_vocab = None, trg_vocab = None, device:torch.device = None):
        self.active_config = active_config
        self.config = config
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device
        # download comet kiwi model
        # log in to huggingface
        with open(f"{Path.home()}/reinforce/hf_token.txt") as f:
            token = f.readline().strip()        
        huggingface_hub.login(token)
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        self.comet_kiwi_model: UnifiedMetric = load_from_checkpoint(model_path)
        self.comet_kiwi_model.requires_grad_(False)
            
        self.comet_kiwi_tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")

    def scale_and_baseline(self, batch_scores, scaled_score_record, raw_score_record):
        if self.config['baseline_strategy'] == "none":
            return batch_scores

        def scale(reward, a, b, minim, maxim):
            # a, b: absolute bounds
            if maxim-minim == 0:
                print("minim==maxim")
                return 0
            rel = (reward - minim)/(maxim-minim)
            #assert rel <= 1 and rel >= 0
            return (b-a)*rel + a

        if len(scaled_score_record) > 0:
            if self.config['baseline_strategy'] == 'epoch_mean':
                baseline = np.mean(raw_score_record)
            elif self.config['baseline_strategy'] == 'none':
                baseline = 0
            epoch_max = np.max(raw_score_record) - baseline
            epoch_min = np.min(raw_score_record) - baseline
            scaled_scores = [scale(score-baseline, -1, 1, epoch_min, epoch_max) for score in batch_scores]
        # TODO: add prev epoch mean
        elif self.config['baseline_strategy'] == 'batch_mean':
            baseline = np.mean(batch_scores)
            batch_max = np.max(batch_scores) - baseline
            batch_min = np.min(batch_scores) - baseline
            scaled_scores = [scale(score-baseline, -1, 1, batch_min, batch_max) for score in batch_scores]
        elif len(scaled_score_record) <= 0:
            # first batch
            print("batch scaling")
            baseline = 0
            batch_max = np.max(batch_scores) - baseline
            batch_min = np.min(batch_scores) - baseline
            scaled_scores = [scale(score-baseline, -1, 1, batch_min, batch_max) for score in batch_scores]

        return scaled_scores

    
    def uniform_score(self, src):
        return torch.ones(src.shape[0]) * 0.1, torch.ones(src.shape[0]) * 0.1


    def base_score(self, attention: torch.Tensor, scaled_score_record = [], raw_score_record = []) -> Tuple[Tensor]:
        # attention [batch_size, n_heads, trg_len, src_len]

        max_vals = torch.max(attention, dim=-1).values # [batch_size, n_heads, trg_len]
        raw_scores = (max_vals).mean(dim=(1,2)) #torch.log(max_vals).sum(dim=(1,2))
        assert list(raw_scores.shape) == [attention.shape[0]] # batch_size
        
        raw_scores = raw_scores.detach().cpu().numpy()
        raw_score_record.extend(raw_scores)
        scaled_scores = self.scale_and_baseline(raw_scores, scaled_score_record, raw_score_record)

        print("raw scores: ", raw_scores)
        print("scaled scores: ", scaled_scores)

        return torch.tensor(scaled_scores), torch.tensor(raw_scores)

    
    def read_alignments_from_file(self, input_file, alignment_file, alignment_prob_file = None):
            print("function: read alignments")
            with open(input_file, 'r') as f:
                corpus = f.readlines()

            with open(alignment_file, 'r') as f:
                id_alignment_lines = f.readlines()

            if alignment_prob_file:
                with open(alignment_prob_file, 'r') as f:
                    alignment_prob_lines = f.readlines()

            id_alignments = []
            token_alignments = []    
            prob_list = []

            for i in range(len(corpus)):
                #print("sent_pair: ", sent_pair)
                #print("id_alignment_line: ", id_alignment_line)
                sent_pair = corpus[i]
                id_alignment_line = id_alignment_lines[i]
                if alignment_prob_file:
                    alignment_prob_line = alignment_prob_lines[i]

                src_tokens, trg_tokens = sent_pair.split("|||")[0].split(), sent_pair.split("|||")[1].split()
                id_pairs = id_alignment_line.split()
                if alignment_prob_file:
                    probs = alignment_prob_line.split()
                    if len(probs) == 0:
                        prob = 0
                    else:
                        prob = sum([float(prob) for prob in probs])/len(probs)
                else:
                    prob = 0
                prob_list.append(prob)
                id_alignment = []
                token_alignment = []
                for id_pair in id_pairs:
                    #print(id_pair)
                    src_id, trg_id = int(id_pair.split("-")[0]), int(id_pair.split("-")[1])

                    id_pair = (src_id, trg_id) # (1,2)
                    if src_id < len(src_tokens) and trg_id < len(trg_tokens):
                        src_token = src_tokens[src_id] 
                        trg_token = trg_tokens[trg_id]
                        token_pair = (src_token, trg_token)
                    
                        id_alignment.append(id_pair) # [(), ()]
                        token_alignment.append(token_pair)
                
                id_alignments.append(id_alignment) # [[(), ()], [(), ()]]
                token_alignments.append(token_alignment)

            return id_alignments, token_alignments, prob_list


    def visualize_attention(self, src, labels, cross_attention=None, encoder_attention=None, decoder_attention=None):
        # code reference: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#a-simple-categorical-heatmap

        # src: [batch_size, source seq length]
        # labels: [batch_size, target seq length]
        # attention: [batch_size, n_heads, trg_len, src_len]

        print("viz for first in batch")
        cross_att = torch.mean(cross_attention, dim=1)[0].cpu() # [trg_len, src_len]
        enc_self_att = torch.mean(encoder_attention, dim=1)[0].cpu() # [src_len, src_len]
        dec_self_att = torch.mean(decoder_attention, dim=1)[0].cpu() # [trg_len, trg_len]

        fig, ax = plt.subplots()
        ax.imshow(cross_att)

        src_tokens = self.tokenizer.convert_ids_to_tokens(src[0])
        trg_tokens = self.tokenizer.convert_ids_to_tokens(labels[0])

        ax.set_xticks(np.arange(len(trg_tokens)), labels=trg_tokens)
        ax.set_yticks(np.arange(len(src_tokens)), labels=src_tokens)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(trg_tokens)):
            for j in range(len(src_tokens)):
                text = ax.text(i, j, cross_att[i][j],
                       ha="center", va="center", color="w")
                
        ax.set_title("Cross Attention, pooled between heads")
        fig.tight_layout()

        plt.savefig("cross_attention_fig.jpg")

        plt.clf()
        
        fig, ax = plt.subplots()
        ax.imshow(enc_self_att)

        ax.set_xticks(np.arange(len(src_tokens)), labels=src_tokens)
        ax.set_yticks(np.arange(len(src_tokens)), labels=src_tokens)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(src_tokens)):
            for j in range(len(src_tokens)):
                text = ax.text(i, j, enc_self_att[i][j],
                       ha="center", va="center", color="w")
                
        ax.set_title("Encoder Self Attention, pooled between heads")
        fig.tight_layout()
        plt.savefig("enc_attention_fig.jpg")

        plt.clf()

        fig, ax = plt.subplots()
        ax.imshow(dec_self_att)

        ax.set_xticks(np.arange(len(trg_tokens)), labels=trg_tokens)
        ax.set_yticks(np.arange(len(trg_tokens)), labels=trg_tokens)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(trg_tokens)):
            for j in range(len(trg_tokens)):
                text = ax.text(i, j, dec_self_att[i][j],
                       ha="center", va="center", color="w")
                
        ax.set_title("Decoder Self Attention, pooled between heads")
        fig.tight_layout()
        plt.savefig("dec_attention_fig.jpg")

    def dependency_parse_score_base_align(self, src, labels, cross_attention=None, encoder_attention=None, decoder_attention=None, scaled_score_record=[], raw_score_record=[]):
        # src: [batch_size, src seq length] does not include src lang code
        # labels: [batch_size, trg seq length] does not include trg lang code
        # attention: [batch_size, n_heads, trg_len, src_len] does not include src lang code, target lang code

        src_tokens_list = [self.tokenizer.convert_ids_to_tokens(src_sent) for src_sent in src]
        output_tokens_list = [self.tokenizer.convert_ids_to_tokens(label_sent) for label_sent in labels]

        head_pooled_cross_attention = torch.mean(cross_attention, dim=1) # [batch_size, trg_len, src_len]
        head_pooled_enc_attention = torch.mean(encoder_attention, dim=1) # [batch_size, src_len, src_len]
        head_pooled_dec_attention = torch.mean(decoder_attention, dim=1) # [batch_size, trg_len, trg_len]
        
        src_eos_indices = torch.nonzero(src == self.tokenizer.eos_token_id) # [batch_size, 2] # [a,b] = example a, eos token in index b
        trg_eos_indices = torch.nonzero(labels == self.tokenizer.eos_token_id) # [batch_size, 2] 
        
        for src_eos_index in src_eos_indices:
            print("src_eos_index", src_eos_index)
            # TODO fix (considering shift-att, how to deal with this?)
            head_pooled_cross_attention[src_eos_index[0], :, src_eos_index[1]:] = 0.0 # just in case, set attention to zero for every token after eos
            
            # don't consider attention for same token
            src_len = head_pooled_enc_attention[src_eos_index[0]].shape[0]
            # TODO fix (considering shift-att, how to deal with this?)
            for i in range(src_len):
                head_pooled_enc_attention[src_eos_index[0], i, i] = 0.0

            # don't consider attention for tokens after eos
            head_pooled_enc_attention[src_eos_index[0], :, src_eos_index[1]:] = 0.0
        
        for trg_eos_index in trg_eos_indices:
            print("trg_eos_index", trg_eos_index)
            head_pooled_dec_attention[trg_eos_index[0], :, trg_eos_index[1]:] = 0.0

            # don't consider attention for same token
            trg_len = head_pooled_dec_attention[trg_eos_index[0]].shape[0]
            for i in range(trg_len):
                head_pooled_dec_attention[trg_eos_index[0], i, i] = 0.0

        # get trg-to-src alignments using cross attention    
        src_id_aligned_to_trg_id = torch.argmax(head_pooled_cross_attention, dim=-1).tolist() # [batch_size, trg_len]
        # shift-att
        for ex in range(len(src_id_aligned_to_trg_id)):
            for i in range(1,len(src_id_aligned_to_trg_id[ex])):
                src_id_aligned_to_trg_id[ex][i-1] = src_id_aligned_to_trg_id[ex][i]
        # don't calculate attention scores after eos in trg
        for trg_eos_index in trg_eos_indices:
            src_id_aligned_to_trg_id[trg_eos_index[0]] = src_id_aligned_to_trg_id[trg_eos_index[0]][:trg_eos_index[1]]
    
        # get src-to-src alignments using encoder attention
        src_id_aligned_to_src_id = torch.argmax(head_pooled_enc_attention, dim=-1).tolist() # [batch_size, src_len]
        # shift-att
        for ex in range(len(src_id_aligned_to_src_id)):
            for i in range(1,len(src_id_aligned_to_src_id[ex])):
                src_id_aligned_to_src_id[ex][i-1] = src_id_aligned_to_src_id[ex][i]
        # don't calculate attention scores after eos in src
        for src_eos_index in src_eos_indices:
            src_id_aligned_to_src_id[src_eos_index[0]] = src_id_aligned_to_src_id[src_eos_index[0]][:src_eos_index[1]]
            
        # get trg-to-trg alignments using decoder attention
        trg_id_aligned_to_trg_id = torch.argmax(head_pooled_dec_attention, dim=-1).tolist() # [batch_size, trg_len]
        # shift-att
        for ex in range(len(trg_id_aligned_to_trg_id)):
            for i in range(1, len(trg_id_aligned_to_trg_id[ex])):
                trg_id_aligned_to_trg_id[ex][i-1] = trg_id_aligned_to_trg_id[ex][i]
        # don't calculate attention scroes after eos in trg
        for trg_eos_index in trg_eos_indices:
            trg_id_aligned_to_trg_id[trg_eos_index[0]] = trg_id_aligned_to_trg_id[trg_eos_index[0]][:trg_eos_index[1]]

        raw_scores = []

        for i in range(len(src)): 
            print(f"Processing sent pair {i}")
            print("src:", src_tokens_list[i])
            print("output:", output_tokens_list[i])

            print("head pooled cross attention")
            print(head_pooled_cross_attention[i])
            print("head pooled cross attention shape: ", head_pooled_cross_attention[i].shape)
            
            print("src len: ", len(src[i]))
            print("trg len: ", len(labels[i]))
            
            assert head_pooled_cross_attention[i].shape[1] == len(src[i])
            assert head_pooled_cross_attention[i].shape[0] == len(labels[i])

            #self.visualize_attention(src, labels, cross_attention, encoder_attention, decoder_attention)
            
            id_cross_alignment = {}
            for trg_id, src_id in enumerate(src_id_aligned_to_trg_id[i]):
                id_cross_alignment[trg_id] = src_id
            
            id_src_alignment = {}
            for src_id1, src_id2 in enumerate(src_id_aligned_to_src_id[i]):
                id_src_alignment[src_id1] = src_id2

            id_trg_alignment = {}
            for trg_id1, trg_id2 in enumerate(trg_id_aligned_to_trg_id[i]):
                id_trg_alignment[trg_id1] = trg_id2

            # log (intuitively) which src token is aligned to which target token 
            _scores = []
            for trg_id, src_id in id_cross_alignment.items():
                src_vocab_id = src[i][src_id].item()
                trg_vocab_id = labels[i][trg_id].item()
                src_token = self.tokenizer.convert_ids_to_tokens(src_vocab_id)
                trg_token = self.tokenizer.convert_ids_to_tokens(trg_vocab_id)
                print("trg-src alignment: (trg token, src token)", trg_token, src_token)

                # using self attention, get the most relevant word for src token in src sentence
                src_id2 = id_src_alignment[src_id]
                src_vocab_id2 = src[i][src_id2].item()
                src_token2 = self.tokenizer.convert_ids_to_tokens(src_vocab_id2)
                print("src-src alignment: (src token1, src token2)", src_token, src_token2)

                # using self attention, get the most relevant word for trg token in trg sentence
                trg_id2 = id_trg_alignment[trg_id]
                trg_vocab_id2 = labels[i][trg_id2].item()
                trg_token2 = self.tokenizer.convert_ids_to_tokens(trg_vocab_id2)
                print("trg-trg alignment: (trg token1, trg token2)", trg_token, trg_token2)

                # use the relevant word pair's cross attention value as score
                _score = head_pooled_cross_attention[i, trg_id2, src_id2].item()
                _scores.append(_score)
            if len(_scores) > 0:
                score = sum(_scores)/len(_scores)
            else:
                score = 0.0
            raw_scores.append(score)

        raw_score_record.extend(raw_scores)
        scaled_scores = self.scale_and_baseline(raw_scores, scaled_score_record, raw_score_record)
        
        print("raw scores: ", raw_scores)
        print("scaled scores: ", scaled_scores)

        return torch.tensor(scaled_scores), torch.tensor(raw_scores)
    

    def remove_incompatible_ids(self, token_ids: Tensor):
        # turn lang ids to pad
        incompatible_ids = self.tokenizer.additional_special_tokens_ids
        incompatible_ids.append(-100)
        incompatible_ids = torch.tensor(incompatible_ids, device=token_ids.device)
        mask_tensor =  torch.isin(token_ids, incompatible_ids)
        token_ids = token_ids.masked_fill_(mask_tensor, self.tokenizer.pad_token_id)
        return token_ids


    def comet_kiwi_score(self, src, labels, cross_attention=None, encoder_attention=None, decoder_attention=None, scaled_score_record=[], raw_score_record=[]):
        
        data = [
                {
                    "src": self.tokenizer.decode(src_sent, skip_special_tokens=True, clean_up_tokenization_spaces=True), 
                    "mt": self.tokenizer.decode(label_sent, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                } 
                for src_sent, label_sent in zip(src, labels)
            ]
            
        model_output = self.comet_kiwi_model.predict(data, batch_size=len(data), gpus=self.config['gpu_num']) 
            
        raw_scores = model_output.scores
        
        print("raw scores: ", raw_scores)
        
        raw_score_record.extend(raw_scores)
        scaled_scores = self.scale_and_baseline(raw_scores, scaled_score_record, raw_score_record)
            
        print("scaled scores: ", scaled_scores)

        return torch.tensor(scaled_scores), torch.tensor(raw_scores)

    def test_score(self):
        """
        @me
        """
        print("Testing score function")
        attention = torch.tensor([[0.5,0.2,0.3],
                              [0.3,0.6,0.1]])
        attention = attention.unsqueeze(0).unsqueeze(1)
        answer = torch.tensor([(torch.tensor(0.5) + torch.tensor(0.6))/2])
        result = self.base_score(None, None, attention)
        if result != answer:
            print("Wrong score implementation")
            print(f"Expected {answer} but got {result}")
            return False
        else:
            print("Test passed")
            return True
    