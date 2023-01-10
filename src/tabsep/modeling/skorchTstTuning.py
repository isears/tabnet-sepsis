"""
Find optimal hyperparameters for TST
"""
from dataclasses import fields

import optuna
import pandas as pd
import torch
from mvtst.optimizers import AdamW, PlainRAdam, RAdam
from optuna.integration.skorch import SkorchPruningCallback
from sklearn.metrics import average_precision_score, roc_auc_score
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
from tabsep.modeling.skorchEnsembleTst import ensemble_tst_factory
from tabsep.modeling.skorchTst import skorch_tst_factory


def objective(trial: optuna.Trial) -> float:
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
    trial.suggest_categorical("optimizer_name", ["AdamW", "PlainRAdam", "RAdam"])
    trial.suggest_categorical("weight_decay", [1e-3, 1e-2, 1e-1, 0])

    pretraining_ds = EnsembleDataset("cache/train_examples.csv")
    tst_config = TSTConfig(save_path="cache/models/tstTuning", **trial.params)

    tst = ensemble_tst_factory(
        tst_config,
        pretraining_ds,
        # pruner=SkorchPruningCallback(trial=trial, monitor="auprc"),
    )

    try:
        tst.fit(pretraining_ds, y=None)
    except RuntimeError as e:
        print(f"Warning, assumed runtime error: {e}")
        del tst
        torch.cuda.empty_cache()
        # return float("nan")
        return 0.0

    # TODO: there has to be a better way to do this
    # this doesn't necessarily return the auprc of the checkpoint-ed model
    # Checkpoints are based on loss
    epoch_scoring_callbacks = [c for c in tst.callbacks if type(c) == EpochScoring]
    best_auprc = next(
        filter(lambda c: c.name == "auprc", epoch_scoring_callbacks)
    ).best_score_
    return best_auprc


if __name__ == "__main__":
    pruner = None
    # pruner = optuna.pruners.PercentilePruner(25.0)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=500)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
