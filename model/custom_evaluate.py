import pandas as pd
from typing import List
from evaluate import load
from utils import compute_bleu
from sacrebleu.metrics import BLEU
from bleurt import score
from pathlib import Path
from comet.models import UnifiedMetric

huggingface_bleu = load("bleu")
huggingface_chrf_pp = load("chrf")

def save_translations(src_sents: List[List[str]], candidate_trans_sents: List[List[str]], ref_trans_sents: List[List[List[str]]], tag: str):
    assert len(src_sents) == len(candidate_trans_sents) and len(src_sents) == len(ref_trans_sents)
    data = pd.DataFrame()
    data['source'] = src_sents
    data['candidate_translation'] = candidate_trans_sents
    data['reference_translation'] = ref_trans_sents
    #data.to_csv(f'{dir_name}/{tag}-translations.csv')


def calculate_corpus_bleu(candidate_trans: List[str], ref_trans: List[List[str]], trg_lang: str = ''):
    ref_trans = [[ref[0] for ref in ref_trans]]
    bleu = BLEU(trg_lang=trg_lang)
    score = bleu.corpus_score(candidate_trans, ref_trans)
    return score.score

def calculate_sentence_bleu(candidate_trans: str, ref_trans: List[str], trg_lang: str = ''):
    bleu = BLEU(effective_order=True, trg_lang=trg_lang)
    sent_bleu = bleu.sentence_score(candidate_trans, ref_trans)
    return sent_bleu.score

def calculate_tokenized_bleu(candidate_trans: List[List[str]], ref_trans: List[List[List[str]]]):
    bleu = compute_bleu(ref_trans, candidate_trans) # revision: include <eos>
    return bleu[0]

def calculate_chrf_pp(candidate_trans: List[str], ref_trans: List[List[str]]):
    return huggingface_chrf_pp.compute(predictions=candidate_trans, references=ref_trans, word_order=2)["score"]

def calculate_comet_kiwi(model: UnifiedMetric, src_sents: List[str], candidate_trans: List[str]):
    data = []
    for i in range(len(src_sents)):
        item = {
                "src": src_sents[i],
                "mt": candidate_trans[i]
            }
        data.append(item)
    model_output = model.predict(data, batch_size=16, gpus=1)
    return model_output.system_score
    

def calculate_bleurt(candidate_trans: List[str], ref_trans: List[List[str]]):
    ref_trans = [ref[0] for ref in ref_trans]

    checkpoint = f"{Path.home()}/bleurt/bleurt/test_checkpoint"

    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=ref_trans, candidates=candidate_trans)

    return 0.5