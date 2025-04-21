#! /bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n02
#SBATCH --time=48:00:00

srun python metrics/xcomet.py --run ku2pdgn4 ha06550a m1jberrv 44kf8vgd nu3updls qioh2lvg