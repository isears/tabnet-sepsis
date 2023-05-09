"""
Find optimal hyperparameters for TST
"""
from dataclasses import fields

import numpy as np
import optuna
import pandas as pd
import torch
from mvtst.optimizers import AdamW, PlainRAdam, RAdam
from optuna.integration.skorch import SkorchPruningCallback
from scipy.stats import sem
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from tabsep import config
from tabsep.dataProcessing.ensembleDataset import EnsembleDataset
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import TSTConfig
from tabsep.modeling.cvUtil import cv_generator
from tabsep.modeling.skorchEnsembleTst import ensemble_tst_factory
from tabsep.modeling.skorchTst import skorch_tst_factory


def objective(trial: optuna.Trial) -> float:
    # Parameters to tune:
    trial.suggest_float("lr", 1e-8, 0.1, log=True)
    trial.suggest_float("dropout", 0.1, 0.7)
    trial.suggest_categorical("d_model_multiplier", [1, 2, 4, 8, 16, 32, 64])
    trial.suggest_int("num_layers", 1, 15)
    trial.suggest_categorical("n_heads", [4, 8, 16, 32, 64])
    trial.suggest_int("dim_feedforward", 128, 512)
    trial.suggest_int("batch_size", 8, 256)
    trial.suggest_categorical("pos_encoding", ["fixed", "learnable"])
    trial.suggest_categorical("activation", ["gelu", "relu"])
    trial.suggest_categorical("norm", ["BatchNorm", "LayerNorm"])
    trial.suggest_categorical("optimizer_name", ["AdamW", "PlainRAdam", "RAdam"])
    trial.suggest_categorical("weight_decay", [1e-3, 1e-2, 1e-1, 0])

    all_scores = {"AUROC": list(), "Average precision": list(), "F1": list()}

    for fold_idx, (train_ds, test_ds) in enumerate(cv_generator(n_splits=3)):
        print(f"[*] Starting fold {fold_idx}")

        # Need to make sure max_len is the same so that the shapes don't change
        actual_max_len = max(train_ds.max_len, test_ds.max_len)
        train_ds.max_len = actual_max_len
        test_ds.max_len = actual_max_len

        tst_config = TSTConfig(save_path="cache/models/skorchCvTst", **trial.params)

        tst = skorch_tst_factory(tst_config, train_ds, pretrained_encoder=False)

        try:
            tst.fit(train_ds, y=None)
        except RuntimeError as e:
            print(f"Warning, assumed runtime error: {e}")
            del tst
            torch.cuda.empty_cache()
            # return float("nan")
            return 0.0

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

    return np.mean(all_scores["AUROC"])


if __name__ == "__main__":
    pruner = None
    # pruner = optuna.pruners.PercentilePruner(25.0)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=10000)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
