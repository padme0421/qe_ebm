import wandb
import json
from bleurt.score import BleurtScorer
import os
from pathlib import Path
import numpy as np
import argparse
import evaluate

def main(run_id):
    # download test translations
    run = wandb.init(project="reinforce-final")

    try:
        artifact_dir = run.use_artifact(f"reinforce-final/run-{run_id}-test_translations:v0", type='run_table').download()
    except:
        artifact_dir = f"artifacts/run-{run_id}-test_translations:v0"

    checkpoint = f"{Path.home()}/bleurt/bleurt/BLEURT-20"
    #scorer = BleurtScorer(checkpoint)
    scorer = evaluate.load("bleurt", "BLEURT-20", module_type="metric")

    refs = []
    candidates = []
    with open(f"{artifact_dir}/test_translations.table.json") as f:
        raw_data = json.load(f)
        for raw_item in raw_data["data"]:
            refs.append(raw_item[2][0])
            candidates.append(raw_item[1])

    scores = scorer.compute(predictions=candidates,references=refs)
    system_score = np.mean(scores)
    print(system_score)

    os.makedirs(f"bleurt/run-{run_id}/", exist_ok=True)
    with open(f"bleurt/run-{run_id}/test_translation_results.txt", "w") as f:
        f.write(str(system_score))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str)
    args = parser.parse_args()
    main(args.run)