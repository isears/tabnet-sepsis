"""
Train single transformer model for downstream analysis
"""

import datetime
import os

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling.tstEstimator import TstWrapper

if __name__ == "__main__":
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    sids_train, sids_test = train_test_split(
        cut_sample["stay_id"].to_list(), test_size=0.1, random_state=42
    )

    save_path = f"cache/models/singleTst_{start_time_str}"
    os.mkdir(save_path)

    pd.DataFrame(data={"stay_id": sids_train}).to_csv(
        f"{save_path}/train_stayids.csv", index=False
    )

    pd.DataFrame(data={"stay_id": sids_test}).to_csv(
        f"{save_path}/test_stayids.csv", index=False
    )

    print("[+] Data loaded, training...")
    tst = TstWrapper()
    tst.fit(sids_train)

    print(f"[+] Training complete, saving to {save_path}")

    torch.save(tst.model.state_dict(), f"{save_path}/model.pt")

    preds = tst.decision_function(sids_test)
    torch.save(preds, f"{save_path}/preds.pt")
    score = roc_auc_score(FileBasedDataset(sids_test).get_labels(), preds)
    print(f"Validation score: {score}")

    with open(f"{save_path}/roc_auc_score.txt", "w") as f:
        f.write(str(score))
        f.write("\n")
