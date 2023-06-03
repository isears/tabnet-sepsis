import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from tabsep import config
from tabsep.dataProcessing.derivedDataset import DerivedDataset


def load_to_mem(train_ds):
    all_X, all_y = torch.tensor([]), torch.tensor([])

    memory_loader = DataLoader(
        train_ds,
        batch_size=16,  # Batch size only important for tuning # workers to load to mem
        num_workers=config.cores_available,
        collate_fn=train_ds.last_available_collate,
        drop_last=False,
    )

    for batch_X, batch_y in memory_loader:
        all_X = torch.cat((all_X, batch_X))
        all_y = torch.cat((all_y, batch_y))

    return all_X, all_y


if __name__ == "__main__":
    stay_ids = pd.read_csv("cache/included_stay_ids.csv").squeeze("columns").to_list()
    train_sids, test_sids = train_test_split(stay_ids, test_size=0.1, random_state=42)
    train_ds = DerivedDataset(train_sids)
    train_X, train_y = load_to_mem(train_ds)
    print("[*] Training data loaded data to memory")

    lr = LogisticRegression(max_iter=10000)
    lr.fit(train_X, train_y)

    test_ds = DerivedDataset(test_sids)
    test_X, test_y = load_to_mem(test_ds)
    print("[*] Testing data loaded to memory")

    final_auroc = roc_auc_score(test_y, lr.predict_proba(test_X)[:, 1])
    final_auprc = average_precision_score(test_y, lr.predict_proba(test_X)[:, 1])
    final_f1 = f1_score(test_y, lr.predict_proba(test_X)[:, 1].round())

    print("Saving model and feature importances")
    os.makedirs(config.lr_path, exist_ok=True)

    with open(f"{config.lr_path}/whole_model.pkl", "wb") as f:
        pickle.dump(lr, f)

    odds_ratios = np.exp(lr.coef_)
    coefficients = lr.coef_
    odds_ratios_df = pd.DataFrame(
        data={
            "Variable": train_ds.features,
            "Odds Ratios": odds_ratios.squeeze(),
            "Coefficients": coefficients.squeeze(),
        }
    )

    odds_ratios_df.to_csv(f"{config.lr_path}/odds_ratios.csv", index=False)

    print("Final score:")
    print(f"\tAUROC: {final_auroc}")
    print(f"\tAverage precision: {final_auprc}")
    print(f"\tF1: {final_f1}")
