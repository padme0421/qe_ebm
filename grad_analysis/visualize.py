import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import rcParams
import pickle
import json
import numpy as np
import ast
import os

def draw(run_id, algorithm, sample_id):

    grads_file = f"{run_id}/{algorithm}_grads_id{sample_id}.pkl"
    updated_graph_file = f"{run_id}/{algorithm}_grad_analysis_id{sample_id}.png"
    strings_file = f"{run_id}/{algorithm}_strings_id{sample_id}.json"
    tokens_file = f"{run_id}/{algorithm}_tokens_id{sample_id}.txt"
    gloss_file = f"{run_id}/{algorithm}_token_gloss_id{sample_id}.txt"

    with open(grads_file, 'rb') as f:
        filtered_grads = pickle.load(f)

    with open(tokens_file) as f:
        unsup_labels_tokens = json.load(f) # TODO: unsup_labels misnomer -> change to prediction(?)

    with open(strings_file) as f:
        text_dict = json.load(f)
        unsup_input_str = text_dict["input"]
        label_str = text_dict["labels"]
        unsup_labels_str = text_dict["output"]

    with open(gloss_file) as f:
        glosses = f.readline()
        glosses = ast.literal_eval(glosses)
    
    if len(glosses) != len(unsup_labels_tokens):
        print("ERROR")
        return
        
    valid_indices = [i for i, token in enumerate(unsup_labels_tokens) if token != "<pad>"]
    filtered_tokens = [unsup_labels_tokens[i] for i in valid_indices]
    filtered_glosses = [glosses[i] for i in valid_indices]

    # unsup_labels: [batch_size * num_hypotheses_nmt, seq_length]
    # grad of loss wrt param
    rcParams['font.family'] = 'FreeSerif'

    colormap = plt.cm.viridis 
    colors = colormap(filtered_grads)

    fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
    fig.suptitle(f"Source: {unsup_input_str}\nGold Translation: {label_str}\nPrediction: {unsup_labels_str}")
    ax: Axes = ax

    ax.set_xticks(range(len(filtered_tokens)))
    token_and_gloss = [f"{t} ({g})" for t,g in zip(filtered_tokens, filtered_glosses)]
    ax.set_xticklabels(token_and_gloss, rotation=90)
    ax.set_yticks([])
    im = ax.imshow(colors[np.newaxis, :, :], aspect='auto')
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.2)

    fig.savefig(updated_graph_file)
    plt.close()

if __name__ == "__main__":
    run_id = "kr7xnh32"
    algorithm = "ebm"
    tgt_lang = "Bengali"

    for sample_id in range(216):
        grads_file = f"{run_id}/{algorithm}_grads_id{sample_id}.pkl"

        if not os.path.isfile(grads_file):
            continue

        draw(run_id, algorithm, sample_id)

