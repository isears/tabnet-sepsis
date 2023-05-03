import numpy as np
import pandas as pd
import torch
from scipy.stats import sem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling.cvUtil import cv_generator


def load_to_mem(train_ds):
    all_X, all_y = torch.tensor([]), torch.tensor([])

    memory_loader = DataLoader(
        train_ds,
        batch_size=16,  # Batch size only important for tuning # workers to load to mem
        num_workers=config.cores_available,
        collate_fn=train_ds.last_nonzero_collate,
        drop_last=False,
    )

    for batch_X, batch_y in memory_loader:
        all_X = torch.cat((all_X, batch_X))
        all_y = torch.cat((all_y, batch_y))

    return all_X, all_y


if __name__ == "__main__":
    all_scores = {"AUROC": list(), "Average precision": list(), "F1": list()}

    for fold_idx, (train_ds, test_ds) in enumerate(cv_generator(n_splits=10)):
        print(f"[*] Starting fold {fold_idx}")

        train_X, train_y = load_to_mem(train_ds)
        print("[*] Training data loaded data to memory")

        lr = LogisticRegression(max_iter=10000)
        lr.fit(train_X, train_y)

        test_X, test_y = load_to_mem(test_ds)
        print("[*] Testing data loaded to memory")

        preds = lr.predict_proba(test_X)[:, 1]

        all_scores["AUROC"].append(roc_auc_score(test_y, preds))
        all_scores["Average precision"].append(average_precision_score(test_y, preds))
        all_scores["F1"].append(f1_score(test_y, preds.round()))

        print("[+] Fold complete")
        for key, val in all_scores.items():
            print(f"\t{key}: {val[-1]}")

    print("[+] All folds complete")
    for key, val in all_scores.items():
        score_mean = np.mean(val)
        standard_error = sem(val)
        print(f"{key}: {score_mean:.5f} +/- {standard_error:.5f}")
