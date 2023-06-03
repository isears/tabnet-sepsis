"""
Define a simple 3-layer feedforward neural network classifier
"""

import pandas as pd
import skorch
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from tabsep import config
from tabsep.dataProcessing.derivedDataset import DerivedDataset
from tabsep.modeling import TSTConfig, my_auprc, my_auroc, my_f1
from tabsep.modeling.singleLR import load_to_mem


def skorch_ffnn_factory(ds, batch_size: int):
    m = NeuralNetBinaryClassifier(
        SimpleFFNN,
        module__n_features=len(ds.features),
        module__hidden_dim=int(len(ds.features) / 2),
        iterator_train__collate_fn=ds.last_available_collate,
        iterator_valid__collate_fn=ds.last_available_collate,
        iterator_train__num_workers=config.cores_available,
        iterator_valid__num_workers=config.cores_available,
        iterator_train__pin_memory=True,
        iterator_valid__pin_memory=True,
        device="cuda",
        max_epochs=100,
        lr=0.1,
        batch_size=batch_size,
        train_split=skorch.dataset.ValidSplit(0.1),
        callbacks=[
            EarlyStopping(patience=3),
            GradientNormClipping(gradient_clip_value=4.0),
            Checkpoint(
                load_best=True,
                fn_prefix=f"cache/ffnn/",
                f_pickle="whole_model.pkl",
            ),
            EpochScoring(my_auroc, name="auroc", lower_is_better=False),
            EpochScoring(my_auprc, name="auprc", lower_is_better=False),
            EpochScoring(my_f1, name="f1", lower_is_better=False),
        ],
    )

    return m


class SimpleFFNN(nn.Module):
    def __init__(self, n_features, hidden_dim=32) -> None:
        super().__init__()

        self.ffnn = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        y_hat = self.ffnn(x)
        return torch.sigmoid(y_hat)


if __name__ == "__main__":
    stay_ids = pd.read_csv("cache/included_stay_ids.csv").squeeze("columns").to_list()
    train_sids, test_sids = train_test_split(stay_ids, test_size=0.1, random_state=42)
    train_ds = DerivedDataset(stay_ids=stay_ids)
    ffnn = skorch_ffnn_factory(train_ds, batch_size=64)

    ffnn.fit(train_ds, y=None)

    test_ds = DerivedDataset(test_sids)
    X, y = load_to_mem(test_ds)
    y_pred = ffnn.predict_proba(test_ds)[:, 1]

    final_auroc = roc_auc_score(y, y_pred)
    final_auprc = average_precision_score(y, y_pred)
    final_f1 = f1_score(y, y_pred.round())

    print("Final score:")
    print(f"\tAUROC: {final_auroc}")
    print(f"\tAverage precision: {final_auprc}")
    print(f"\tF1: {final_f1}")
