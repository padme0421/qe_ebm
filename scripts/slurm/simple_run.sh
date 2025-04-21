#! /bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB 
#SBATCH --cpus-per-task=8
#SBATCH --signal=SIGUSR1@90

source ~/anaconda/etc/profile.d/conda.sh
conda activate align_reinforce_adapter

srun python main.py --function test --model mbart --train_size 10 --test_size 10 --val_size 10 --batch_size 16 --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 1 --score fast_align
#fast_align
#base