#! /bin/bash
#SBATCH --job-name=sup_pl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB 
#SBATCH --cpus-per-task=8
#SBATCH --signal=SIGUSR1@90

source ~/anaconda/etc/profile.d/conda.sh
conda activate align_reinforce_adapter

srun python main.py --active_config wmt19_en_kk_mt5_config --function supervised_train --model mt5 \
    --train_size 0 --test_size 0 --val_size 0 --epochs 200 --load_vocab --score base --selfsup_strategy greedy \
    --batch_size 16 --accumulate_grad_batches 1 --dir_name en-kk-sup
#fast_align
#base
#--dist_strategy deepspeed_stage_2