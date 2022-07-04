#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output ./logs/tscv-%j.log

module load cuda/11.3.1
module load cudnn/8.2.0
nvidia-smi

export PYTHONUNBUFFERED=TRUE

python3 src/tabsep/modeling/timeseriesCV.py tst