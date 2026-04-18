# ⚡ Improving NMT Models by Retrofitting Quality Estimators into Trainable Energy Loss
https://aclanthology.org/2025.coling-main.545/

## 🌒 Environment Setup

### 1. Run basic dependency installation

```bash
conda create -n qe_ebm
bash scripts/setup.sh
```

### 2. WMT19 Test data

`scripts/wmt19_test_data.sh`

```bash
sacrebleu -t wmt19 -l gu-en --echo all > data/wmt19.gu-en.test
sacrebleu -t wmt19 -l kk-en --echo all > data/wmt19.kk-en.test
sacrebleu -t wmt19 -l zh-en --echo all > data/wmt19.zh-en.test
sacrebleu -t wmt19 -l de-en --echo all > data/wmt19.de-en.test
sacrebleu -t wmt19 -l lt-en --echo all > data/wmt19.lt-en.test
```

### 3. ML50 data

a. Install `fairseq`

b. Set env variables 
```
WORKDIR_ROOT=[{repo dir}/ML50]
SPM_PATH=[SPM PATH]
```

c. Run the following 

```bash
cd fairseq/examples/multilingual/data_scripts
pip install -r requirement.txt
bash download_ML50_v1.sh
bash preprocess_ML50_v1.sh
```

[How to find SPM path]

```bash
pip show sentencepiece
# SPM path = result location + '/sentencepiece'
```

### 4. Store your Huggingface token in hf_token.txt

### 5. Configure .env
```
WORKDIR=/example/path
ML50_PATH=/example/path
WMT19_TEST_DATA_PATH=/example/path
```

## 🌓 Experiments
You can run different experiments by adding a config file to `exp_configs`, and passing it to the main entrypoint `main.py`.

`python main.py --config_path exp_configs/{config_example.yml}`

The following shows configs used for experiments in the paper.

| Algorithm  | Mono  | Config         |
| ---------- | ----- | -------------- |
| Supervised |   -   | sup_config.yml |
| REINFORCE  | +Mono | semi_config.yml |
|            | -Mono | semi_config_nomono.yml |
| PPO        | +Mono | semi_ppo_trl_config.yml |
|            | -Mono | semi_ppo_trl_config_nomono.yml |
| QE-Static  | +Mono | ebm_static_config.yml |
|            | -Mono | ebm_static_config_nomono.yml |
| QE-Dynamic | +Mono | ebm_config.yml |
|            | -Mono | ebm_config_nomono.yml |
| Test | - | test_config.yml |

## 🌕 Inspecting Results
- 📉 Gradient analysis code for analyzing influence of individual tokens are in `grad_analysis/`.
- 🍒 Cherry-picked samples that satisfy `sup < (reinforce, ppo) < (qe_static, qe_dynamic)` are in `samples/`. 


