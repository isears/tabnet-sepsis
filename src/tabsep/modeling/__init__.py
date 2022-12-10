import copy
import os
import pickle
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.stats as st
import torch
from sklearn.metrics import (average_precision_score, make_scorer,
                             roc_auc_score, roc_curve)

CORES_AVAILABLE = len(os.sched_getaffinity(0))


@dataclass
class SingleCVResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


class CVResults:
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

