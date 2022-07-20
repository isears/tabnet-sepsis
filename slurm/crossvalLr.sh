#!/bin/bash
#SBATCH -n 1
#SBATCH -p bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=4:00:00
#SBATCH --output ./logs/cvLr-%j.log

export PYTHONUNBUFFERED=TRUE

python3 src/tabsep/modeling/timeseriesCV.py LR