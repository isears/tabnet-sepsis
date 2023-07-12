#!/bin/bash
#SBATCH -n 1
#SBATCH -p bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=752G
#SBATCH --time=1:00:00
#SBATCH --output ./logs/captum.log

nvidia-smi

export PYTHONUNBUFFERED=TRUE

python src/tabsep/modeling/skorchTST.py captum