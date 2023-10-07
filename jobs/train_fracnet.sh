#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=40:00:00
#SBATCH --output=train_unet.out
#SBATCH --job-name=train_unet

# Execute program located in $HOME
source activate ribfrac

srun python main.py