#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output ./logs/mimic2ts-%j.log

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH=./:$PYTHONPATH

cd ~/Repos/tabnet-sepsis/

python -m mimic2ts --src mimiciv --dst mimicts
