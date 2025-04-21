#! /bin/bash
#SBATCH --job-name=qe_token_reparam
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n01
#SBATCH --time=48:00:00

srun python main.py --config_path exp_configs/ebm_static_config.yml