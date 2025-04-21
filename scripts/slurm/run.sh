#! /bin/bash
#SBATCH --job-name=reinforce
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB 
#SBATCH --cpus-per-task=8

source ~/anaconda/etc/profile.d/conda.sh
conda activate align_reinforce_adapter

srun python main.py --train 10000 --test 1000 --val 1000 --epochs 10 --label_keep 0.2 --load_vocab 1