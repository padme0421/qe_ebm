from datasets import load_dataset, Dataset
from typing import List
import torch
import os
import subprocess
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token

lang_pair = "en-de"

dataset: Dataset = load_dataset("RicardoRei/wmt-da-human-evaluation")["train"].filter(lambda example: example["lp"] == lang_pair).select(range(0,1200))
print("dataset size: ", len(dataset))

os.makedirs("wmt/awesome_align", exist_ok=True)
os.makedirs("wmt/dep_parse", exist_ok=True)

awesome_align_dir = "wmt/awesome_align"
dep_parse_dir = "wmt/dep_parse"

spacy_src = spacy.load("en_core_web_md")
spacy_trg = spacy.load("de_core_news_md")

def scale(reward, a, b, minim, maxim):
    if maxim-minim == 0:
        return 0
    return (((b-a)*(reward - minim))/(maxim-minim)) + a

def awesome_align_alignment_score(src_sents: List[str], output_sents: List[str]):

        # make input file for awesome-align
        with open(f"{awesome_align_dir}/awesome_align_input.txt", "w") as f:
            for i in range(len(src_sents)):
                try:
                    src_tokens = " ".join(src_sents[i])
                    trg_tokens = " ".join(output_sents[i])
                except:
                    print("Error in [awesome_align_alignment_score]")
                    src_tokens = ""
                    trg_tokens = ""
                
                f.write(f"{src_tokens} ||| {trg_tokens}")
                # avoid writing blank line at the end
                if i != len(src_sents)-1:
                    f.write("\n")
        
        
        # align words
        subprocess.run(["awesome-align", # "CUDA_VISIBLE_DEVICES=0", 
            "--output_file", f"{awesome_align_dir}/awesome_align_output.txt",
            "--output_prob_file", f"{awesome_align_dir}/awesome_align_probs.txt",
            "--model_name_or_path", "bert-base-multilingual-cased",
            "--data_file", f"{awesome_align_dir}/awesome_align_input.txt",
            "--extraction", "softmax",
            "--batch_size", "64"])
        
        id_alignments, token_alignments, probs = read_alignments(f"{awesome_align_dir}/awesome_align_input.txt", 
                                                                      f"{awesome_align_dir}/awesome_align_output.txt",
                                                                      f"{awesome_align_dir}/awesome_align_probs.txt")
        
        batch_max = max(probs)
        batch_min = min(probs)
        probs = [scale(score, -1, 1, batch_min, batch_max) for score in probs]
        print(probs)
        return probs


def dependency_parse_score(src_sents, output_sents):

        src_tokens_list = []
        src_docs = []
        for src_sent in src_sents:
            doc_src = spacy_src(src_sent)
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
            try:
              doc_trg = spacy_trg(output_sent)
              trg_docs.append(doc_trg)
              #print(doc_trg.text)
              trg_tokens = []
              for token in doc_trg:
                  trg_tokens.append(token.text)
                  #print(token.text, token.pos_, token.dep_)
              trg_tokens_list.append(trg_tokens)
            except:
              trg_docs.append(None)
              trg_tokens_list.append([])
        
        # make input file for awesome-align
        with open(f"{dep_parse_dir}/awesome_align_input.txt", "w") as f:
            for i in range(len(src_sents)):
              try:
                src_tokens = " ".join(src_tokens_list[i])
                trg_tokens = " ".join(trg_tokens_list[i])
              except:
                src_tokens = ""
                trg_tokens = ""
              f.write(f"{src_tokens} ||| {trg_tokens}")
              # avoid writing blank line at the end
              if i != len(src_sents)-1:
                  f.write("\n")

        # align words (as heads)
        subprocess.run(["awesome-align", # "CUDA_VISIBLE_DEVICES=0", 
            "--output_file", f"{dep_parse_dir}/awesome_align_output.txt",
            "--model_name_or_path", "bert-base-multilingual-cased",
            "--data_file", f"{dep_parse_dir}/awesome_align_input.txt",
            "--extraction", "softmax",
            "--batch_size", "64"])


        id_alignments, token_alignments, probs = read_alignments(f"{dep_parse_dir}/awesome_align_input.txt", f"{dep_parse_dir}/awesome_align_output.txt")
        #pprint.pprint(token_alignments)

        # for each aligned head pair, calculate the alignment scores of their heads and sum them up
        scores = []
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
            
            scores.append(score)

        batch_max = max(scores)
        batch_min = min(scores)
        scores = [scale(score, -1, 1, batch_min, batch_max) for score in scores]
        print(scores)
        return scores

def read_alignments(input_file, alignment_file, alignment_prob_file = None):
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

def score_dataset(batch):
    dep_parse_score = dependency_parse_score(batch["src"], batch["mt"])
    awesome_align_score = awesome_align_alignment_score(batch["src"], batch["mt"])
    return {"dep_parse": dep_parse_score, "awesome_align": awesome_align_score}
    
dataset = dataset.map(score_dataset, batched=True, batch_size=1200)
df = dataset.to_pandas()
print(df)
score_df = df[["score", "dep_parse", "awesome_align"]]
corr = score_df.corr(method="spearman", numeric_only=True)
print(corr)