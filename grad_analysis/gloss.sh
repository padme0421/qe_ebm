#! /bin/bash
#SBATCH --job-name=qe_ebm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n02
#SBATCH --time=48:00:00

srun python gloss.py
