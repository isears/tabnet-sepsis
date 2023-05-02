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
from tabsep.modeling.skorchTst import TUNING_PARAMS, TSTConfig, skorch_tst_factory

if __name__ == "__main__":
    all_scores = {"AUROC": list(), "Average precision": list(), "F1": list()}

    for fold_idx, (train_ds, test_ds) in enumerate(cv_generator(n_splits=5)):
        print(f"[*] Starting fold {fold_idx}")

        # Need to make sure max_len is the same so that the shapes don't change
        actual_max_len = max(train_ds.max_len, test_ds.max_len)
        train_ds.max_len = actual_max_len
        test_ds.max_len = actual_max_len

        tst_config = TSTConfig(save_path="cache/models/skorchCvTst", **TUNING_PARAMS)

        tst = skorch_tst_factory(tst_config, train_ds, pretrained_encoder=False)
        tst.fit(train_ds, y=None)

        test_y = test_ds.examples["label"].to_numpy()
        preds = tst.predict_proba(test_ds)[:, 1]

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
