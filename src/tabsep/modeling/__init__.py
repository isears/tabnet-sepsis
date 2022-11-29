import copy
import os
import pickle
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.stats as st
import torch
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve

CORES_AVAILABLE = len(os.sched_getaffinity(0))


# Inspired by https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience=0, min_lr=1e-6, factor=0.5):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


# TODO: update with best weights on validation set
class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.saved_best_weights = None

    def __call__(self, val_loss, weights):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.saved_best_weights = copy.deepcopy(weights)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.saved_best_weights = copy.deepcopy(weights)
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("Early stopping")
                self.early_stop = True

    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.best_loss = None
        self.saved_best_weights = None


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

