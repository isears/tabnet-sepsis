"""
Find optimal hyperparameters for TST
"""
import os
from dataclasses import fields

import numpy as np
import optuna
import pandas as pd
import skorch
import torch
from optuna.integration import SkorchPruningCallback
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import (
    TSTCombinedConfig,
    TSTModelConfig,
    TSTRunConfig,
    my_auprc,
    my_auroc,
    tst_skorch_factory,
)


def optuna_params_to_config(optuna_params: dict) -> TSTCombinedConfig:
    """
    Utility function to convert optuna trial params to model factory configs
    """
    model_config = TSTModelConfig(
        **{k: v for k, v in optuna_params.items() if k in fields(TSTModelConfig)}
    )
    run_config = TSTRunConfig(
        **{k: v for k, v in optuna_params.items() if k in fields(TSTRunConfig)}
    )
    combined_config = TSTCombinedConfig(
        save_path="cache/models/optunatst",
        model_config=model_config,
        run_config=run_config,
    )

    return combined_config


class Objective:
    def __init__(self, trainvalid_sids):
        self.trainvalid_sids = trainvalid_sids

    def __call__(self, trial: optuna.Trial):
        # Parameters to tune:
        trial.suggest_float("lr", 1e-10, 0.1, log=True)
        trial.suggest_float("dropout", 0.01, 0.7)
        trial.suggest_categorical("d_model_multiplier", [1, 2, 4, 8, 16, 32, 64])
        trial.suggest_int("num_layers", 1, 15)
        trial.suggest_categorical("n_heads", [4, 8, 16, 32, 64])
        trial.suggest_int("dim_feedforward", 128, 512)
        trial.suggest_int("batch_size", 8, 256)
        trial.suggest_categorical("pos_encoding", ["fixed", "learnable"])
        trial.suggest_categorical("activation", ["gelu", "relu"])
        trial.suggest_categorical("norm", ["BatchNorm", "LayerNorm"])

        train_ds = FileBasedDataset(self.trainvalid_sids)

        tst = tst_skorch_factory(
            optuna_params_to_config(trial.params),
            train_ds,
            # pruner=SkorchPruningCallback(trial=trial, monitor="valid_loss"),
            pruner=None,
        )

        try:
            tst.fit(train_ds, train_ds.get_labels())
        except RuntimeError as e:
            print(f"Warning, assumed runtime error: {e}")
            del tst
            torch.cuda.empty_cache()
            return 0

        epoch_scoring_callbacks = [c for c in tst.callbacks if type(c) == EpochScoring]
        best_auprc = next(
            filter(lambda c: c.name == "auprc", epoch_scoring_callbacks)
        ).best_score_

        return best_auprc


def split_data_consistently():
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    sids_train, sids_test = train_test_split(
        cut_sample["stay_id"].to_list(), test_size=0.1, random_state=42
    )

    return sids_train, sids_test


if __name__ == "__main__":
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    sids_train, sids_test = split_data_consistently()

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=None)
    study.optimize(Objective(sids_train), n_trials=500)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Retrain w/hyperparams then test on hold-out dataset
    print("Re-training with hyperparams and testing on hold-out dataset")
    train_ds = FileBasedDataset(sids_train)
    test_ds = FileBasedDataset(sids_test)

    tuned_tst = tst_skorch_factory(optuna_params_to_config(trial.params), train_ds)

    tuned_tst.fit(train_ds, train_ds.get_labels())

    final_auroc = roc_auc_score(
        test_ds.get_labels(), tuned_tst.predict_proba(test_ds)[:, 1]
    )

    final_auprc = average_precision_score(
        test_ds.get_labels(), tuned_tst.predict_proba(test_ds)
    )

    print("Final score:")
    print(f"\tAUROC: {final_auroc}")
    print(f"\tAverage precision: {final_auprc}")
