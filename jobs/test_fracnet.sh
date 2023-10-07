#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=0:10:00
#SBATCH --output=test_unet.out
#SBATCH --job-name=test_unet

# Execute program located in $HOME
source activate ribfrac

srun python predict.py