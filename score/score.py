from typing import Tuple
import torch
from torch import Tensor
import torch
import subprocess
import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer
from fast_align.build.force_align import run as force_align_run
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
import matplotlib.pyplot as plt
from comet import download_model, load_from_checkpoint
from comet.models import UnifiedMetric
import huggingface_hub
from pathlib import Path
from score.awesome_align_module import AwesomeAligner

class Scorer:
    def __init__(self, active_config, config, tokenizer: PreTrainedTokenizer = None, src_vocab = None, trg_vocab = None, device:torch.device = None):
        self.active_config = active_config
        self.config = config
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device

        if self.config["score"] == "dep_parse_awesome_align" or self.config["score"] == "const_parse":
            self.spacy_src = spacy.load(active_config["spacy_src"])
            self.spacy_trg = spacy.load(active_config["spacy_trg"])

        # download comet kiwi model
        # log in to huggingface
        with open(f"{Path.home()}/reinforce/hf_token.txt") as f:
            token = f.readline().strip()        
        huggingface_hub.login(token)
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        self.comet_kiwi_model: UnifiedMetric = load_from_checkpoint(model_path)
        self.comet_kiwi_model.requires_grad_(False)
            
        self.comet_kiwi_tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")

        if self.config["score"] == "awesome_align" or ((self.config["score"] == "ensemble") and ("awesome_align" in self.config["score_list"])):
            self.awesome_align_model = AwesomeAligner(model_name_or_path="bert-base-multilingual-cased", device=self.device)


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

    def fast_align_alignment_score(self, src, labels, attention: torch.Tensor = None, scaled_score_record=None, raw_score_record = None):
        # src: [batch size, (src len - 1)] -> without <sos>
        # labels: [batch size, (trg len - 1)]
    
        batch_size = attention.shape[0]
    
        src_sents = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in src]
        output_sents = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in labels]

        fast_align_input = ""
        for i in range(batch_size):
            fast_align_input += " ".join(src_sents[i]) + " ||| " + " ".join(output_sents[i]) +"\n"     
        #print(fast_align_input)

        # run fast align
        print("running fast align to calculate score")
        #result = subprocess.run(["/bin/bash", "test.sh", 
        #                         #"--fp", "fwd_params", "--fe", "fwd_err", 
        #                        #"--rp", "rev_params", "--re", "rev_err"
        #                         ],
        #                        cwd="fast_align",
        #                        input=fast_align_input,
        #                        capture_output=True, text=True)
        #print("current working directory: ", os.getcwd())
    
        #result = subprocess.run(["build/force_align.py", "fwd_params", "fwd_err", "rev_params", "rev_err",
        #                          "grow-diag-final-and"],
        #                          cwd="fast_align",
        #                          input=fast_align_input,
        #                          capture_output=True, text=True
        #                         )

        output = force_align_run(fast_align_input, 
                             f"fast_align/{self.config['dir_name']}/fwd_params", 
                             f"fast_align/{self.config['dir_name']}/fwd_err", 
                             f"fast_align/{self.config['dir_name']}/rev_params", 
                             f"fast_align/{self.config['dir_name']}/rev_err", 
                             "grow-diag-final-and")
    

        # process alignments
        #print(output)
        raw_scores = []
        out_lines = output.rstrip("\n").split("\n")
        for out_line in out_lines:
            alignment = out_line.split("|")[0]
            prob = out_line.split("|")[1]
            prob = float(prob)
            score = prob # log likelihood -> higher is better
            raw_scores.append(score)
        
        print("raw scores: ", raw_scores)
        raw_score_record.extend(raw_scores)
        scaled_scores = self.scale_and_baseline(raw_scores, scaled_score_record, raw_score_record)
        print("scaled scores: ", scaled_scores)
        
        return torch.tensor(scaled_scores), torch.tensor(raw_scores)

    
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

    
    def awesome_align_alignment_score(self, src, labels, attention = None, scaled_score_record=[],  raw_score_record=[]):
        src_sents = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in src]
        output_sents = [self.tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) + [self.tokenizer.eos_token] for s in labels]

        # make input for awesome-align
        awesome_align_input = []
        for i in range(len(src_sents)):
            try:
                src_tokens = " ".join(src_sents[i])
                trg_tokens = " ".join(output_sents[i])
            except:
                print("Error in [awesome_align_alignment_score]")
                src_tokens = ""
                trg_tokens = ""
                
            awesome_align_input.append(f"{src_tokens} ||| {trg_tokens}")
        
        #all_devices = find_usable_cuda_devices(2)
        #if len(all_devices) >= 2:
        #    device_id = all_devices[1] # out of available device, use the second one
        #else:
        #    device_id = all_devices[0]
        
        # align words
        raw_scores = []
        for pair in awesome_align_input:
            word_aligns = self.awesome_align_model.align(pair, output_prob=True)
            raw_score = np.mean([val.cpu() for val in word_aligns.values()])
            raw_scores.append(raw_score)
            
        '''
        output = subprocess.run(["awesome-align",
            "--output_file", "awesome_align_output.txt", # f"{self.config['dir_name']}
            "--output_prob_file", "awesome_align_probs.txt", # f"{self.config['dir_name']}
            "--model_name_or_path", "bert-base-multilingual-cased",
            "--data_file", "awesome_align_input.txt", #f"{self.config['dir_name']}
            "--extraction", "softmax",
            "--batch_size", "32"], capture_output=True)
        print(output)
        
        id_alignments, token_alignments, raw_scores = self.read_alignments_from_file("awesome_align_input.txt", #f"{self.config['dir_name']}
                                                                      "awesome_align_output.txt","awesome_align_probs.txt")
        '''

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
    

    def dependency_parse_score_awesome_align(self, src, labels, attention = None, scaled_score_record=[], raw_score_record=[]):
        src_sents = self.tokenizer.batch_decode(src, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_sents = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        src_tokens_list = []
        src_docs = []
        for src_sent in src_sents:
            doc_src = self.spacy_src(src_sent)
            src_docs.append(doc_src)
            #print(doc_src.text)
            src_tokens = []
            for token in doc_src:
                src_tokens.append(token.text)
                #print(token.text, token.pos_, token.dep_)
            src_tokens_list.append(src_tokens)

                        
        trg_docs = []
        trg_tokens_list = []
        for output_sent in output_sents:
            doc_trg = self.spacy_trg(output_sent)
            trg_docs.append(doc_trg)
            #print(doc_trg.text)
            trg_tokens = []
            for token in doc_trg:
                trg_tokens.append(token.text)
                #print(token.text, token.pos_, token.dep_)
            trg_tokens_list.append(trg_tokens)
        
        # make input file for awesome-align
        with open("awesome_align_input.txt", "w") as f:
            for i in range(len(src_sents)):
                src_tokens = " ".join(src_tokens_list[i])
                trg_tokens = " ".join(trg_tokens_list[i])
                f.write(f"{src_tokens} ||| {trg_tokens}")
                # avoid writing blank line at the end
                if i != len(src_sents)-1:
                    f.write("\n")

        # align words (as heads)
        subprocess.run(["awesome-align", # "CUDA_VISIBLE_DEVICES=0", 
            "--output_file", "awesome_align_output.txt",
            "--model_name_or_path", "bert-base-multilingual-cased",
            "--data_file", "awesome_align_input.txt",
            "--extraction", "softmax",
            "--batch_size", "32"])


        id_alignments, token_alignments, probs = self.read_alignments_from_file("awesome_align_input.txt", "awesome_align_output.txt")
        #pprint.pprint(token_alignments)

        # for each aligned head pair, calculate the alignment scores of their heads and sum them up
        raw_scores = []
        for i in range(len(src_sents)):
            print(f"Processing sent pair {i}")
            _scores = []
            # tokens
            src_tokens, trg_tokens = src_tokens_list[i], trg_tokens_list[i]

            # parse trees
            src_doc: Doc = src_docs[i]
            #print("src_doc:", src_doc)
            trg_doc: Doc = trg_docs[i]
            #print("trg_doc:", trg_doc)
            # alignment
            id_alignment = id_alignments[i] # [(1,1), (2,4)]

            # each pair
            print("alignments:")
            id_alignment_dict = {}
            for id_pair in id_alignment:
                id_alignment_dict[id_pair[0]] = id_pair[1]

            for id_pair in id_alignment:
                #print(id_pair)
                src_id, trg_id = id_pair
                if src_id < len(src_doc) and trg_id < len(trg_doc):
                    src_node: Token = src_doc[src_id]
                    trg_node: Token = trg_doc[trg_id]
                    #print(src_node.text, trg_node.text)
                    #_scores.append(src_node.head.similarity(trg_node.head))
                    _scores.append(int(id_alignment_dict.get(src_node.head.i,-1) == trg_node.head.i))
            if len(_scores) == 0:
                score = 0
                print("len(_scores) == 0")
            else:
                score = sum(_scores)/len(_scores)
            
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
    