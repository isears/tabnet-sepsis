import os
import pickle
import sys
from dataclasses import asdict, dataclass, fields
from typing import Callable, Protocol, Tuple

import numpy as np
import scipy.stats as st
import torch
from mvtst.optimizers import AdamW, PlainRAdam, RAdam
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    roc_auc_score,
    roc_curve,
)

from tabsep import config


@dataclass
class TSTModelConfig:
    """
    A config for the TST model with "sensible" params
    for a classifier recommended in the paper as default values
    """

    d_model_multiplier: int = None
    d_model: int = 128
    n_heads: int = 16
    num_layers: int = 3
    dim_feedforward: int = 256
    num_classes: int = 1
    dropout: float = 0.1
    pos_encoding: str = "fixed"
    activation: str = "gelu"
    norm: str = "BatchNorm"
    freeze: bool = False

    def __post_init__(self):
        # Prefer to use d_model_multiplier, but drop down to d_model if not available
        if self.d_model_multiplier is not None:
            self.d_model = self.n_heads * self.d_model_multiplier
        elif self.d_model is not None:
            assert self.d_model % self.n_heads == 0
            assert self.d_model > self.n_heads
            self.d_model_multiplier = int(self.d_model / self.n_heads)
        else:
            raise ValueError(f"One of d_model_multiplier and d_model must be sepcified")

    def generate_params(self):
        # Drop d_model_multiplier
        ret = {
            param_name: param_val
            for param_name, param_val in self.__dict__.items()
            if param_name != "d_model_multiplier"
        }

        return ret

    def generate_skorch_params(self):
        ret = self.generate_params()
        # Add prefix
        ret = {
            f"module__{param_name}": param_val for param_name, param_val in ret.items()
        }
        return ret


@dataclass
class TSTConfig:
    # Non-model Params
    save_path: str
    optimizer_name: str = "AdamW"
    weight_decay: float = 0
    batch_size: int = 128
    lr: float = 1e-4

    # Model Params
    d_model_multiplier: int = None
    d_model: int = 128
    n_heads: int = 16
    num_layers: int = 3
    dim_feedforward: int = 256
    num_classes: int = 1
    dropout: float = 0.1
    pos_encoding: str = "fixed"
    activation: str = "gelu"
    norm: str = "BatchNorm"
    freeze: bool = False

    def __post_init__(self) -> None:
        # Gather params and distribute to appropriate sub-config
        model_config_params = {
            f.name: asdict(self)[f.name] for f in fields(TSTModelConfig)
        }

        self.model_config = TSTModelConfig(**model_config_params)

    def get_optimizer_cls(self):
        if self.optimizer_name == "AdamW":
            return AdamW
        elif self.optimizer_name == "PlainRAdam":
            return PlainRAdam
        elif self.optimizer_name == "RAdam":
            return RAdam
        else:
            raise ValueError("Optimizer must be one of: (AdamW | PlainRAdam | RAdam)")

    def generate_optimizer(self, model_params):
        optimizer_cls = self.get_optimizer_cls()
        return optimizer_cls(model_params, lr=self.lr)

    def generate_skorch_full_params(self) -> dict:
        """
        Params passable to skorch NeuralNet for full training
        """
        return dict(
            **self.model_config.generate_skorch_params(),
            optimizer=self.get_optimizer_cls(),
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            iterator_train__batch_size=self.batch_size,
            iterator_valid__batch_size=self.batch_size,
        )

    def generate_skorch_pretraining_params(self) -> dict:
        """
        Params passable to skorch NeuralNet for encoder pretraining
        """
        non_passable_params = ["module__num_classes"]
        ret = self.generate_skorch_full_params()
        ret = {k: v for k, v in ret.items() if k not in non_passable_params}
        return ret

    def generate_model_params(self) -> dict:
        return self.model_config.generate_params()

    def generate_optimizer_params(self) -> dict:
        return dict(lr=self.lr)


@dataclass
class SingleCVResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    avg_precision: float


class CVResults:
    """
    Standardized storage of CV Results + util functions
    """

    @classmethod
    def load(cls, filename) -> "CVResults":
        with open(filename, "rb") as f:
            return pickle.load(f)

    def __init__(self) -> None:
        self.results = list()

    def add_result(self, y_true, y_score) -> None:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        self.results.append(SingleCVResult(fpr, tpr, thresholds, auc, avg_precision))

    def print_report(self) -> None:
        aucs = np.array([res.auc for res in self.results])
        precisions = np.array([res.avg_precision for res in self.results])

        auc_interval = st.t.interval(
            alpha=0.95, df=len(aucs) - 1, loc=aucs.mean(), scale=st.sem(aucs)
        )
        precision_interval = st.t.interval(
            alpha=0.95,
            df=len(precisions) - 1,
            loc=precisions.mean(),
            scale=st.sem(precisions),
        )
        print(f"AUROC: {aucs.mean()} {auc_interval}")
        print(f"Avg Precision: {precisions.mean()} {precision_interval}")

    def get_precisions(self) -> list:
        return [res.avg_precision for res in self.results]

    def get_precision_avg(self) -> float:
        precisions = np.array([res.avg_precision for res in self.results])
        return np.mean(precisions)

    def get_precision_std(self) -> float:
        precisions = np.array([res.avg_precision for res in self.results])
        return np.std(precisions)

    def save_report(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)


# Hack to workaround discrepancies between torch and sklearn shape expectations
# https://github.com/skorch-dev/skorch/issues/442
def my_auroc(net, X, y):
    y_proba = net.predict_proba(X)
    return roc_auc_score(y, y_proba[:, 1])


def my_auprc(net, X, y):
    y_proba = net.predict_proba(X)
    return average_precision_score(y, y_proba[:, 1])


def my_f1(net, X, y):
    y_proba = net.predict_proba(X)
    return f1_score(y, y_proba[:, 1].round())
