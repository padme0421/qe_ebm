#! /bin/bash
#SBATCH --job-name=semi_reinforce_pl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB 
#SBATCH --cpus-per-task=8
#SBATCH --signal=SIGUSR1@90

source ~/anaconda/etc/profile.d/conda.sh
conda activate align_reinforce_adapter

srun python main.py --active_config wmt19_en_kk_mt5_config --function semisupervised_train --model mt5 --train_size 0 --test_size 0 --val_size 0 \
    --epochs 30 --load_vocab --score base --selfsup_strategy greedy --batch_size 8 \
    --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 4 --dir_name en-kk-fix-wt-update-mmt --unsup_wt 0.2
#score: fast_align/base
#selfsup_strategy: greedy/random