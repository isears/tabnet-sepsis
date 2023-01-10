#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output ./logs/skorchEnsembleTst.log

module load cuda/11.3.1
module load cudnn/8.2.0
nvidia-smi

export PYTHONUNBUFFERED=TRUE


python --version

python src/tabsep/modeling/skorchEnsembleTst.py
