import json
import wandb

MAIN_LANGS = ['bn', 'mr', 'de', 'zh']
ADD_LANGS = ['az', 'mn', 'ka']

run_dict = {
    'bn': {
        'supervised': 'jcdd9jys',
        'reinforce': '7pvxs4ch',
        'ppo': 'az7pigqk',
        'qe_static': '59mefoo9',
        'qe_dynamic': 'y0e2853t'
    },
    'mr': {
        'supervised': '398ft75k',
        'reinforce': 'o4t86krj',
        'ppo': '7n23f65c',
        'qe_static': '6vxm0crq',
        'qe_dynamic': 'vhf8xjzs'
    },
    'de': {
        'supervised': 'cvyyzegv',
        'reinforce': '92k5uro2',
        'ppo': 'utrcwka3',
        'qe_static': 'd9pt1wg5',
        'qe_dynamic': '802i0qdb'
    },
    'zh': {
        'supervised': 'w5jsx4ad',
        'reinforce': 'vr4y1csi',
        'ppo': '92hhmbg4',
        'qe_static': 'dh6g79rw',
        'qe_dynamic': '981bxny2'
    },
    'az': {
        'supervised': 'zljgph1j',
        'qe_static': '3s7k7fgw',
        'qe_dynamic': 'e29j4fyy'
    },
    'mn': {
        'supervised': 'hs1akeis',
        'qe_static': '7heogcqh',
        'qe_dynamic': '00hw6hkz'
    },
    'ka': {
        'supervised': 'zctrc476',
        'qe_static': 'i9eqp9e6',
        'qe_dynamic': 'v8hyd6mw'
    }
}

def get_data_from_run_id(current_run, run_id):
    try:
        artifact_dir = current_run.use_artifact(f"reinforce-final/run-{run_id}-test_translations:v0", type='run_table').download()
    except:
        artifact_dir = f"artifacts/run-{run_id}-test_translations:v0"

    data = []
    with open(f"{artifact_dir}/test_translations.table.json") as f:
        raw_data = json.load(f)
        for raw_item in raw_data["data"]:
            item = {
                "src": raw_item[0],
                "mt": raw_item[1],
                "ref": raw_item[2][0],
                "bleu": raw_item[3]
            }
            data.append(item)

    return data

for lang in MAIN_LANGS:
    sup_run_id = run_dict[lang]['supervised']
    reinforce_run_id = run_dict[lang]['reinforce']
    ppo_run_id = run_dict[lang]['ppo']
    qe_static_run_id = run_dict[lang]['qe_static']
    qe_dynamic_run_id = run_dict[lang]['qe_dynamic']

    run = wandb.init(project="reinforce-final")
    sup_run_data = get_data_from_run_id(run, sup_run_id)
    reinforce_run_data = get_data_from_run_id(run, reinforce_run_id)
    ppo_run_data = get_data_from_run_id(run, ppo_run_id)
    qe_static_run_data = get_data_from_run_id(run, qe_static_run_id)
    qe_dynamic_run_data = get_data_from_run_id(run, qe_dynamic_run_id)
    
    # find the samples that satisfy sup < (reinforce, ppo) < (qe_static, qe_dynamic)
    best_samples = []
    for i in range(len(sup_run_data)):
        sup_rl_comp = (sup_run_data[i]['bleu'] < reinforce_run_data[i]['bleu']
                       ) and (sup_run_data[i]['bleu'] < ppo_run_data[i]['bleu'])
        
        rl_qe_comp = (reinforce_run_data[i]['bleu'] < qe_static_run_data[i]['bleu']
                        ) and (reinforce_run_data[i]['bleu'] < qe_dynamic_run_data[i]['bleu']
                        ) and (ppo_run_data[i]['bleu'] < qe_static_run_data[i]['bleu']
                        ) and (ppo_run_data[i]['bleu'] < qe_dynamic_run_data[i]['bleu']
                        )
        
        if sup_rl_comp and rl_qe_comp:
            best_samples.append({
                "supervised": sup_run_data[i],
                "reinforce": reinforce_run_data[i],
                "ppo": ppo_run_data[i],
                "qe_static": qe_static_run_data[i],
                "qe_dynamic": qe_dynamic_run_data[i]
                })
            
    best_samples.sort(key=lambda item:item['qe_dynamic']['bleu'], reverse=True)
    with open(f"best_samples_main_{lang}.json", 'w') as f:
        for sample in best_samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")

for lang in ADD_LANGS:
    sup_run_id = run_dict[lang]['supervised']
    qe_static_run_id = run_dict[lang]['qe_static']
    qe_dynamic_run_id = run_dict[lang]['qe_dynamic']

    run = wandb.init(project="reinforce-final")
    sup_run_data = get_data_from_run_id(run, sup_run_id)
    qe_static_run_data = get_data_from_run_id(run, qe_static_run_id)
    qe_dynamic_run_data = get_data_from_run_id(run, qe_dynamic_run_id)
    
    # find the samples that satisfy sup < (reinforce, ppo) < (qe_static, qe_dynamic)
    best_samples = []
    for i in range(len(sup_run_data)):
        sup_qe_comp = (sup_run_data[i]['bleu'] < qe_static_run_data[i]['bleu']
                        ) and (sup_run_data[i]['bleu'] < qe_dynamic_run_data[i]['bleu'])
        
        if sup_qe_comp:
            best_samples.append({
                "supervised": sup_run_data[i],
                "qe_static": qe_static_run_data[i],
                "qe_dynamic": qe_dynamic_run_data[i]
                })
    
    best_samples.sort(key=lambda item:item['qe_dynamic']['bleu'], reverse=True)
    with open(f"best_samples_add_{lang}.json", 'w') as f:
        for sample in best_samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")