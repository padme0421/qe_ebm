#! /bin/bash
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n02
#SBATCH --time=48:00:00

srun python best_samples.py
