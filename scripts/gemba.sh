#! /bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n02
#SBATCH --time=48:00:00

srun python metrics/gemba_calc.py --run y0e2853t