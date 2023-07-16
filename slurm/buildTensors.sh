#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=5:00:00
#SBATCH --output ./logs/buildTensors.log

export PYTHONUNBUFFERED=TRUE
for i in 3 6 9 12 15 18 21 24
do
    python src/tabsep/dataProcessing/derived2tensor.py $i
done

# wait $(jobs -p)