#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --time=24:00:00
#SBATCH --output ./logs/captum.log

nvidia-smi

export PYTHONUNBUFFERED=TRUE
source /gpfs/home/isears1/anaconda3/bin/activate /users/isears1/anaconda/lsepsis

python src/tabsep/modeling/featureImportanceTst.py
