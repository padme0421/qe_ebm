import wandb
from comet import download_model, load_from_checkpoint
import json
import huggingface_hub
import os
import argparse

def main(run_id):
    # download test translations
    run = wandb.init(project="reinforce-final")
    try:
        artifact_dir = run.use_artifact(f"reinforce-final/run-{run_id}-test_translations:v0", type='run_table').download()
    except:
        artifact_dir = f"artifacts/run-{run_id}-test_translations:v0"

    data = []
    with open(f"{artifact_dir}/test_translations.table.json") as f:
        raw_data = json.load(f)
        for raw_item in raw_data["data"]:
            item = {
                "src": raw_item[0],
                "mt": raw_item[1]
            }
            data.append(item)

    # download comet kiwi model
    # log in to huggingface
    with open("hf_token.txt") as f:
        token = f.readline().strip()
    huggingface_hub.login(token)

    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)

    model_output = model.predict(data, batch_size=16, gpus=1)
    print(model_output.system_score)
    os.makedirs(f"comet_kiwi/run-{run_id}/", exist_ok=True)
    with open(f"comet_kiwi/run-{run_id}/test_translation_results.txt", "w") as f:
        f.write(str(model_output.system_score))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str)
    args = parser.parse_args()
    main(args.run)
    