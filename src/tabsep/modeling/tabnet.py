import tempfile

import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skorch import NeuralNet, NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
    LRScheduler,
)

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import (
    BaseModelRunner,
    TabsepModelFactory,
    my_auprc,
    my_auroc,
    my_f1,
)
from tabsep.modeling.commonCV import cv_runner

BEST_PARAMS = {
    "n_d": 64,
    "n_a": 44,
    "n_steps": 9,
    "gamma": 1.472521619271299,
    "n_independent": 1,
    "momentum": 0.05692333567726392,
    "mask_type": "sparsemax",
    "optimizer": {"lr": 0.004280287778344044},
}


class CompatibleTabnet(TabNetClassifier):
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.numpy()
        y = y.numpy()

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

        super().fit(
            X_train,
            y_train,
            patience=10,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["logloss", "auc"],
        )


def do_cv():
    d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
    X = d.get_snapshot()
    y = d.get_labels()

    cv_runner(lambda: CompatibleTabnet(**BEST_PARAMS), X, y)


if __name__ == "__main__":
    do_cv()
