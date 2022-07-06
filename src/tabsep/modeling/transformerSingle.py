"""
Train single transformer model for downstream analysis
"""

import torch
from sklearn.model_selection import train_test_split
from tabsep.modeling.timeseriesCV import TstWrapper, load_to_mem
import pandas as pd
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
import os
import datetime

CORES_AVAILABLE = len(os.sched_getaffinity(0))


if __name__ == "__main__":
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    print("[*] Loading data to memory")
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    cut_sample = cut_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    ds = FileBasedDataset(processed_mimic_path="./cache/mimicts", cut_sample=cut_sample)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=256,
        num_workers=CORES_AVAILABLE,
        pin_memory=True,
    )

    X, y = load_to_mem(dl)

    print("[+] Data loaded, training...")
    tst = TstWrapper()
    tst.fit(X, y)

    save_path = f"cache/models/singleTst_{start_time_str}"
    print(f"[+] Training complete, saving to {save_path}")
    torch.save(tst.model.state_dict(), save_path)
