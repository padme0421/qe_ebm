#! /bin/bash
#SBATCH --job-name=pretranslate
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=master
#SBATCH --time=48:00:00

srun python3 llm_translate_script.py --config_path exp_configs/llm_translation.yml