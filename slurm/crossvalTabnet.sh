#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output ./logs/cvTabnet.log

export PYTHONUNBUFFERED=TRUE

python3 src/tabsep/modeling/timeseriesCV.py "TABNET"