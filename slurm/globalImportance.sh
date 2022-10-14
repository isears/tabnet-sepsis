#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=190G
#SBATCH --time=00:15:00
#SBATCH --output ./logs/globalImportance.log


export PYTHONUNBUFFERED=TRUE
source /gpfs/home/isears1/anaconda3/bin/activate /users/isears1/anaconda/lsepsis

python src/tabsep/reporting/globalImportance.py