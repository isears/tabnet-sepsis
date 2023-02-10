#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output ./logs/debug.log


export PYTHONUNBUFFERED=TRUE

#conda activate ecmo-interpretability

echo "Establishing connection back to $SLURM_SUBMIT_HOST:63321"
~/anaconda3/envs/ecmo-interpretability/bin/python -m debugpy --connect $SLURM_SUBMIT_HOST:63321 --wait-for-client $1