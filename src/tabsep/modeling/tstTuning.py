"""
Find optimal hyperparameters for TST
"""
import os

import optuna
import pandas as pd
import skorch
import torch
from optuna.integration import SkorchPruningCallback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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


class Objective:
    def __init__(self, trainvalid_sids, test_sids):
        self.trainvalid_sids = trainvalid_sids
        self.test_sids = test_sids

    def __call__(self, trial: optuna.Trial):
        # Parameters to tune:
        learning_rate = trial.suggest_float("learning_rate", 1e-10, 0.1, log=True)
        dropout = trial.suggest_float("dropout", 0.01, 0.7)
        d_model_multiplier = trial.suggest_categorical("d_model", [1, 2, 4, 8, 16, 32])
        num_layers = trial.suggest_int("num_layers", 1, 5)
        n_heads = trial.suggest_categorical("n_heads", [4, 8, 16, 32, 64])
        dim_feedforward = trial.suggest_int("dim_feedforward", 128, 512)
        batch_size = trial.suggest_int("batch_size", 8, 128)

        # Embed_dim must be divisible by n_heads
        d_model = n_heads * d_model_multiplier

        save_path = f"cache/models/optuna"
        os.makedirs(save_path, exist_ok=True)
        train_ds = FileBasedDataset(self.trainvalid_sids)

        tst = NeuralNetBinaryClassifier(
            TSTransformerEncoderClassiregressor,
            criterion=torch.nn.BCEWithLogitsLoss,
            optimizer=AdamW,
            optimizer__lr=learning_rate,
            iterator_train__batch_size=batch_size,
            iterator_valid__batch_size=batch_size,
            iterator_train__collate_fn=train_ds.maxlen_padmask_collate,
            iterator_valid__collate_fn=train_ds.maxlen_padmask_collate,
            iterator_train__num_workers=config.cores_available,
            iterator_valid__num_workers=config.cores_available,
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            device="cuda",
            max_epochs=15,  # Large for early stopping
            callbacks=[
                EarlyStopping(patience=3),
                Checkpoint(
                    load_best=True,
                    fn_prefix=f"{save_path}/",
                    f_pickle="whole_model.pkl",
                ),
                EpochScoring(my_auc, name="auc", lower_is_better=False),
                GradientNormClipping(gradient_clip_value=4.0),
                SkorchPruningCallback(trial, monitor="auc"),
            ],
            train_split=skorch.dataset.ValidSplit(0.1),
            # TST params
            module__feat_dim=train_ds.get_num_features(),
            module__d_model=d_model,
            module__dropout=dropout,
            module__dim_feedforward=dim_feedforward,
            module__max_len=train_ds.max_len,
            module__n_heads=n_heads,
            module__num_classes=1,
            module__num_layers=num_layers,
        )

        tst.fit(train_ds, y=train_ds.get_labels())

        test_ds = FileBasedDataset(self.test_sids)

        return roc_auc_score(test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1])


if __name__ == "__main__":
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    sids_train, sids_test = train_test_split(
        cut_sample["stay_id"].to_list(), test_size=0.1, random_state=42
    )

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(Objective(sids_train, sids_test), n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
