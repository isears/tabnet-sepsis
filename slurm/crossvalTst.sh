#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output ./logs/cvTst.log

module load cuda/11.3.1
module load cudnn/8.2.0
nvidia-smi

export PYTHONUNBUFFERED=TRUE

python3 src/tabsep/modeling/timeseriesCV.py TST