#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output ./logs/debug-%j.log


module load python/3.9.0
export PYTHONUNBUFFERED=TRUE
# source /gpfs/runtime/opt/anaconda/3-5.2.0/bin/activate /users/isears1/anaconda/rnaseq
source ~/.virtualenvs/tabsep/bin/activate
echo $1

echo "Establishing connection back to $SLURM_SUBMIT_HOST:51277"
python -m debugpy --connect $SLURM_SUBMIT_HOST:51277 --wait-for-client $1