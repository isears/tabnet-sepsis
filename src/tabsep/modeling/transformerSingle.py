"""
Train single transformer model for downstream analysis
"""

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabsep.modeling.timeseriesCV import TstWrapper, load_to_mem
import pandas as pd
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
import os
import datetime


if __name__ == "__main__":
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    cut_sample = cut_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    ds = FileBasedDataset(processed_mimic_path="./mimicts", cut_sample=cut_sample)
    validation_size = int(0.2 * len(ds))
    # TODO: may have to manually seed the generator if it doesn't happen automatically
    train_ds, valid_ds = torch.utils.data.random_split(
        ds, [len(ds) - validation_size, validation_size]
    )

    print("[+] Data loaded, training...")
    tst = TstWrapper(
        max_epochs=100, max_len=ds.max_len
    )  # To allow early stopper to do its thing
    tst._fitdl(train_ds, use_es=True, valid_ds=valid_ds)

    save_path = f"cache/models/singleTst_{start_time_str}"
    print(f"[+] Training complete, saving to {save_path}")

    os.mkdir(save_path)
    torch.save(tst.model.state_dict(), f"{save_path}/model.pt")

    # torch.save(X_test, f"{save_path}/X_test.pt")
    # torch.save(y_test, f"{save_path}/y_test.pt")

    # preds = tst.decision_function(X_test)
    # torch.save(preds, f"{save_path}/preds.pt")
    # score = roc_auc_score(y_test, preds)
    # print(f"Validation score: {score}")

    # with open(f"{save_path}/roc_auc_score.txt", "w") as f:
    #     f.write(str(score))
    #     f.write("\n")
