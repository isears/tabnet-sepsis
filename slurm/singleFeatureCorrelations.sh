#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output ./logs/singleFeatureCorrelations-%j.log


export PYTHONUNBUFFERED=TRUE
source /gpfs/home/isears1/anaconda3/bin/activate /users/isears1/anaconda/lsepsis

python src/tabsep/reporting/singleFeatureCorrelations.py