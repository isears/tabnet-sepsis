#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output ./logs/unifiedPostProcessing-%j.log

export PYTHONUNBUFFERED=TRUE

echo "[*] Precomputing meta..."
python3 src/tabsep/dataProcessing/precomputeMeta.py
echo "[*] Sepsis aggregation..."
python3 src/tabsep/dataProcessing/sepsisAggregate.py
echo "[*] Post processing..."
python3 src/tabsep/dataProcessing/postProcessTst.py