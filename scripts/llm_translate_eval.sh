#! /bin/bash
#SBATCH --job-name=qe_ebm_seed
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n01
#SBATCH --time=48:00:00

srun python llm_translate_script.py --config_path exp_configs/llm_translation_eval.yml \
                            --llm_translation_file ML50_en_mr_gpt-4-turbo_test-mr.jsonl