import copy
import os
import pickle
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.stats as st
import skorch
import torch
import torch.utils.data
from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor
from mvtst.optimizers import AdamW
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
    roc_curve,
)
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)
from torch.optim.optimizer import Optimizer

from tabsep import config

CORES_AVAILABLE = len(os.sched_getaffinity(0))


@dataclass
class SingleCVResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


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

    def generate_optuna_params(self):
        # Drop d_model_multiplier
        ret = {
            param_name: param_val
            for param_name, param_val in self.__dict__.items()
            if param_name != "d_model_multiplier"
        }
        # Add prefix
        ret = {
            f"module__{param_name}": param_val for param_name, param_val in ret.items()
        }
        return ret


@dataclass
class TSTRunConfig:
    """
    A config for the TST model training loop with "sensible" params
    as suggested by the paper
    """

    batch_size: int = 128
    lr: float = 1e-4

    def generate_optuna_params(self):
        return {
            "iterator_train__batch_size": self.batch_size,
            "iterator_valid__batch_size": self.batch_size,
            "optimizer__lr": self.lr,
        }


@dataclass
class TSTCombinedConfig:
    save_path: str
    model_config: TSTModelConfig
    run_config: TSTRunConfig
    optimizer_cls: type[Optimizer] = AdamW

    def generate_optuna_params(self):
        return {
            **self.model_config.generate_optuna_params(),
            **self.run_config.generate_optuna_params(),
            "optimizer": self.optimizer_cls,
        }


def tst_skorch_factory(
    tst_config: TSTCombinedConfig, ds: torch.utils.data.Dataset, pruner=None
):
    """
    Generate TSTs wrapped in standard skorch wrapper
    """
    os.makedirs(tst_config.save_path, exist_ok=True)

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=3),
        Checkpoint(
            load_best=True,
            fn_prefix=f"{tst_config.save_path}/",
            f_pickle="whole_model.pkl",
        ),
        EpochScoring(my_auroc, name="auroc", lower_is_better=False),
        EpochScoring(my_auprc, name="auprc", lower_is_better=False),
    ]

    tst = NeuralNetBinaryClassifier(
        TSTransformerEncoderClassiregressor,
        criterion=torch.nn.BCEWithLogitsLoss,
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
        max_epochs=15,
        **tst_config.generate_optuna_params(),
    )

    return tst


class CVResults:
    """
    Standardized storage of CV Results + util functions
    """

    @classmethod
    def load(cls, filename) -> "CVResults":
        with open(filename, "rb") as f:
            return pickle.load(f)

    def __init__(self, clf_name) -> None:
        self.results = list()
        self.clf_name = clf_name

    def add_result(self, y_true, y_score) -> float:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        self.results.append(SingleCVResult(fpr, tpr, thresholds, auc))

        # For compatibility w/sklearn scorers
        return auc

    def get_scorer(self) -> Callable:
        metric = lambda y_t, y_s: self.add_result(y_t, y_s)
        return make_scorer(metric, needs_proba=False)

    def print_report(self) -> None:
        aucs = np.array([res.auc for res in self.results])
        print(f"All scores: {aucs}")
        print(f"Score mean: {aucs.mean()}")
        print(f"Score std: {aucs.std()}")
        print(
            f"95% CI: {st.t.interval(alpha=0.95, df=len(aucs)-1, loc=aucs.mean(), scale=st.sem(aucs))}"
        )

        with open(f"results/{self.clf_name}.cvresult", "wb") as f:
            pickle.dump(self, f)


# Hack to workaround discrepancies between torch and sklearn shape expectations
# https://github.com/skorch-dev/skorch/issues/442
def my_auroc(net, X, y):
    y_proba = net.predict_proba(X)
    return roc_auc_score(y, y_proba[:, 1])


def my_auprc(net, X, y):
    y_proba = net.predict_proba(X)
    return average_precision_score(y, y_proba[:, 1])

