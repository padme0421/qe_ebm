#! /bin/bash
#SBATCH --job-name=semi_reinforce_pl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=40000MB 
#SBATCH --cpus-per-task=8

srun bash scripts/sup_semi_pl_mbart_multilang.sh