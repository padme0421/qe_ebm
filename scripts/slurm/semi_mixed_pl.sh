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

srun python main.py --active_config wmt19_en_kk_mt5_config --function semisupervised_train_mixed --model mt5 --train_size 100 --test_size 100 --val_size 100 \
    --epochs 30 --load_vocab --score awesome_align --selfsup_strategy beam --batch_size 4 \
    --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 4 --dir_name en-kk-fix-wt-update-mmt --unsup_wt 0.2
#score: fast_align/base
#selfsup_strategy: greedy/random