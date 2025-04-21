import wandb
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

import sys
import ipdb
import argparse

from GEMBA.gemba.cache import Cache
from GEMBA.gemba.gpt_api import GptApi
from GEMBA.gemba.CREDENTIALS import credentials
from GEMBA.gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer

from absl import app, flags

def main(eval_run, run_id):
    # download test translations
    src_lang = input("src_lang: ")
    trg_lang = input("trg_lang: ")
    run_id = input("run id:")
    try:
        artifact_dir = eval_run.use_artifact(f"reinforce-final/run-{run_id}-test_translations:v0", type='run_table').download()
    except:
        artifact_dir = f"artifacts/run-{run_id}-test_translations:v0"

    # change to correct format
    source = []
    hypothesis = []
    with open(f"{artifact_dir}/test_translations.table.json") as f:
        raw_data = json.load(f)
        for raw_item in raw_data["data"]:
            source.append(raw_item[0])
            hypothesis.append(raw_item[1])

    print(source)
    print(hypothesis)
    assert len(source) == len(hypothesis), "Source and hypothesis files must have the same number of lines."

    df = pd.DataFrame({'source_seg': source, 'target_seg': hypothesis})
    df['source_lang'] = src_lang
    df['target_lang'] = trg_lang

    df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)

    model = "gpt-3.5-turbo" #gpt-4
    gptapi = GptApi(credentials, verbose=True)
    cache = Cache(f"{model}_GEMBA-MQM.jsonl")

    answers = gptapi.bulk_request(df, model, lambda x: parse_mqm_answer(x, list_mqm_errors=False, full_desc=True), cache=cache, max_tokens=500)
    scores = []
    for answer in answers:
        scores.append(float(answer['answer']))
    system_score = sum(scores)/len(scores)

    os.makedirs(f"gemba/run-{run_id}/", exist_ok=True)
    with open(f"gemba/run-{run_id}/test_translation_results.txt", "w") as f:
        f.write(str(system_score))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, nargs='+')
    args = parser.parse_args()

    eval_run = wandb.init(project="reinforce-final")

    for run_id in args.run:
        main(eval_run, run_id)
    