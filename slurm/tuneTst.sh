#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output ./logs/tuneTst.log

module load cuda/11.3.1
module load cudnn/8.2.0
nvidia-smi

export PYTHONUNBUFFERED=TRUE


python --version

python src/tabsep/modeling/tstTuning.py