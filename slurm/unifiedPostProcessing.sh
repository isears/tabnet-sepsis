#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output ./logs/unifiedPostProcessing.log

export PYTHONUNBUFFERED=TRUE

echo "[*] Generating scaler params..."
python3 src/tabsep/dataProcessing/computeScalerParams.py
echo "[*] Generating included features..."
python3 src/tabsep/dataProcessing/featuresIC.py
echo "[*] Inclusion criteria..."
python3 src/tabsep/dataProcessing/inclusionCriteria.py
echo "[*] Precomputing meta..."
python3 src/tabsep/dataProcessing/precomputeMeta.py
echo "[*] Sepsis aggregation..."
python3 src/tabsep/dataProcessing/sepsisAggregate.py
echo "[*] Post processing..."
python3 src/tabsep/dataProcessing/generateExamples.py

echo "[+] Done"