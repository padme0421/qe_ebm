#! /bin/bash
#SBATCH --job-name=eval_finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB 
#SBATCH --cpus-per-task=8
#SBATCH --signal=SIGUSR1@90

source ~/anaconda/etc/profile.d/conda.sh
conda activate align_reinforce_adapter

srun python main.py --active_config iwslt17_en_ko_mbart50_config --function test --model mbart --train_size 0 --test_size 0 --val_size 0 \
 --batch_size 16 --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 1 \
 --score fast_align --from_local_finetuned --checkpoint en-kov0/epoch=4-step=4800-v1.ckpt/checkpoint/mp_rank_00_model_states.pt

#fast_align
#base