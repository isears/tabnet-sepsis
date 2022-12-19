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
from tabsep.modeling import my_auprc, my_auroc
from tabsep.modeling.tstImpl import AdamW, TSTransformerEncoderClassiregressor


def tunable_tst_factory(
    tuning_params: dict,
    ds: FileBasedDataset,
    save_path: str = "cache/models/optunatst",
    pruner=None,
):
    os.makedirs(save_path, exist_ok=True)

    # Need to make sure d_model is divisible by n_heads, so some cleanup needs to be done here
    tuning_params["module__d_model"] = (
        tuning_params["module__n_heads"] * tuning_params["d_model_multiplier"]
    )

    del tuning_params["d_model_multiplier"]

    # Set valid batch size to be same as train batch size
    tuning_params["iterator_valid__batch_size"] = tuning_params[
        "iterator_train__batch_size"
    ]

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=3),
        Checkpoint(
            load_best=True,
            fn_prefix=f"{save_path}/",
            f_pickle="whole_model.pkl",
        ),
        EpochScoring(my_auroc, name="auroc", lower_is_better=False),
        EpochScoring(my_auprc, name="auprc", lower_is_better=False),
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
        iterator_valid__num_workers=config.cores_available,
        iterator_train__pin_memory=True,
        iterator_valid__pin_memory=True,
        device="cuda",
        callbacks=tst_callbacks,
        train_split=skorch.dataset.ValidSplit(0.1),
        # TST params
        module__feat_dim=ds.get_num_features(),
        module__max_len=ds.max_len,
        module__num_classes=1,
        max_epochs=15,
        **tuning_params,
    )

    return tst


class Objective:
    def __init__(self, trainvalid_sids):
        self.trainvalid_sids = trainvalid_sids

    def __call__(self, trial: optuna.Trial):
        # Parameters to tune:
        trial.suggest_float("optimizer__lr", 1e-10, 0.1, log=True)
        trial.suggest_float("module__dropout", 0.01, 0.7)
        trial.suggest_categorical("d_model_multiplier", [1, 2, 4, 8, 16, 32, 64])
        trial.suggest_int("module__num_layers", 1, 15)
        trial.suggest_categorical("module__n_heads", [4, 8, 16, 32, 64])
        trial.suggest_int("module__dim_feedforward", 128, 512)
        trial.suggest_int("iterator_train__batch_size", 8, 256)

        train_ds = FileBasedDataset(self.trainvalid_sids)

        tst = tunable_tst_factory(
            trial.params,
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
    idx_train, idx_test = train_test_split(
        cut_sample.index.to_list(), test_size=0.1, random_state=42
    )

    return idx_train, idx_test


if __name__ == "__main__":
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    sids_train, sids_test = split_data_consistently()

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=None)
    study.optimize(Objective(sids_train), n_trials=1000)

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

    final_auroc = roc_auc_score(
        test_ds.get_labels(), tuned_tst.predict_proba(test_ds)[:, 1]
    )

    final_auprc = average_precision_score(
        test_ds.get_labels(), tuned_tst.predict_proba(test_ds)
    )

    print("Final score:")
    print(f"\tAUROC: {final_auroc}")
    print(f"\tAverage precision: {final_auprc}")
