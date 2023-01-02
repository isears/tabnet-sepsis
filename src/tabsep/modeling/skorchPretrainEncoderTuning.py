from dataclasses import fields

import optuna
import pandas as pd
import torch
from mvtst.optimizers import AdamW, PlainRAdam, RAdam
from optuna.integration import SkorchPruningCallback
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from tabsep import config
from tabsep.dataProcessing.fileBasedImputationDataset import FileBasedImputationDataset
from tabsep.modeling import TSTConfig
from tabsep.modeling.skorchPretrainingTst import skorch_pretraining_encoder_factory


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
    trial.suggest_categorical("optimizer_cls", [AdamW, PlainRAdam, RAdam])
    trial.suggest_categorical("weight_decay", [1e-3, 1e-2, 1e-1, None])

    pretraining_ds = FileBasedImputationDataset("cache/pretrain_examples.csv")
    tst_config = TSTConfig(save_path="cache/models/optunaPretraining", **trial.params)

    pretraining_encoder = skorch_pretraining_encoder_factory(
        tst_config,
        pretraining_ds,
        pruner=SkorchPruningCallback(trial=trial, monitor="valid_loss"),
    )

    try:
        pretraining_encoder.fit(pretraining_ds)
    except RuntimeError as e:
        print(f"Warning, assumed runtime error: {e}")
        del pretraining_encoder
        torch.cuda.empty_cache()
        return 0

    return max(pretraining_encoder.history[:, "valid_loss"])


if __name__ == "__main__":
    pruner = optuna.pruners.PercentilePruner(25.0)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=500)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

