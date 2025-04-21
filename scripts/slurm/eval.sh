#! /bin/bash
#SBATCH --job-name=eval_zeroshot
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB 
#SBATCH --cpus-per-task=8
#SBATCH --signal=SIGUSR1@90

source ~/anaconda/etc/profile.d/conda.sh
conda activate align_reinforce_adapter

srun python main.py --active_config flores_en_si_mbart50_config --function test --model mbart --train_size 0 --test_size 0 --val_size 0 \
    --batch_size 8 --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 1 \
    --score base --unsup_wt 0.5
#fast_align
#base