"""
Find optimal hyperparameters for TST
"""
import os

import numpy as np
import optuna
import pandas as pd
import skorch
import torch
from optuna.integration import SkorchPruningCallback
from sklearn.metrics import roc_auc_score
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
from tabsep.modeling import my_auc
from tabsep.modeling.tstImpl import AdamW, TSTransformerEncoderClassiregressor


def tunable_tst_factory(
    tuning_params: dict,
    ds: FileBasedDataset,
    save_path: str = "cache/models/optuna",
    pruner=None,
):
    os.makedirs(save_path, exist_ok=True)

    # Need to make sure d_model is divisible by n_heads, so some cleanup needs to be done here
    tuning_params["module__d_model"] = (
        tuning_params["module__n_heads"] * tuning_params["d_model_multiplier"]
    )

    del tuning_params["d_model_multiplier"]

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
    ]

    if pruner:
        tst_callbacks.append(pruner)

    tst = NeuralNetBinaryClassifier(
        TSTransformerEncoderClassiregressor,
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=AdamW,
        iterator_train__collate_fn=ds.maxlen_padmask_collate,
        iterator_valid__collate_fn=ds.maxlen_padmask_collate,
        iterator_train__num_workers=config.cores_available,
        iterator_train__pin_memory=True,
        device="cuda",
        callbacks=tst_callbacks,
        train_split=None,
        # TST params
        module__feat_dim=ds.get_num_features(),
        module__max_len=ds.max_len,
        module__num_classes=1,
        **tuning_params,
    )

    return tst


class Objective:
    def __init__(self, training_sids):
        self.training_sids = training_sids

    def __call__(self, trial: optuna.Trial):
        # Parameters to tune:
        trial.suggest_float("optimizer__lr", 1e-10, 0.1, log=True)
        trial.suggest_float("module__dropout", 0.01, 0.9)
        trial.suggest_categorical("d_model_multiplier", [1, 2, 4, 8, 16, 32, 64])
        trial.suggest_int("module__num_layers", 1, 15)
        trial.suggest_categorical("module__n_heads", [4, 8, 16, 32, 64])
        trial.suggest_int("module__dim_feedforward", 128, 1024)
        trial.suggest_int("iterator_train__batch_size", 8, 512)
        trial.suggest_int("max_epochs", 1, 2)  # TODO: three things

        # Have to do manual CV b/c sklearn cross_val_score is broken with pruning callback
        kfolds = KFold(n_splits=2, shuffle=False, random_state=None)
        scores = list()
        for train_indices, valid_indices in kfolds.split(self.training_sids):
            np.array(self.training_sids)[train_indices].tolist()
            train_ds = FileBasedDataset(
                np.array(self.training_sids)[train_indices].tolist()
            )
            valid_ds = FileBasedDataset(
                np.array(self.training_sids)[valid_indices].tolist()
            )

            tst = tunable_tst_factory(
                trial.params,
                train_ds,
                pruner=SkorchPruningCallback(trial=trial, monitor="train_loss"),
            )

            tst.fit(train_ds, train_ds.get_labels())

            scores.append(
                roc_auc_score(valid_ds.get_labels(), tst.predict_proba(valid_ds)[:, 1])
            )

        return sum(scores) / len(scores)


if __name__ == "__main__":
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    sids_train, sids_test = train_test_split(
        cut_sample["stay_id"].to_list(), test_size=0.1, random_state=42
    )

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(Objective(sids_train), n_trials=2)

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

    tuned_tst = tunable_tst_factory(trial.params, train_ds)
    tuned_tst.fit(train_ds, train_ds.get_labels())

    final_score = roc_auc_score(
        test_ds.get_labels(), tuned_tst.predict_proba(test_ds)[:, 1]
    )

    print(f"Final score: {final_score}")
