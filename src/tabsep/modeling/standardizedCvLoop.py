import os

# Be deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.use_deterministic_algorithms(True)

from matplotlib.pyplot import vlines
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from tabsep.dataProcessing.loadAllData import load_from_disk
import sys
import numpy as np
import scipy.stats as st


class PretrainingTabNetClf(TabNetClassifier):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        print("Running unsupervised pretraining...")
        unsupervised = TabNetPretrainer()
        unsupervised.fit(X)
        print("Pretraining complete, moving onto supervised training")
        super().fit(X, y, from_unsupervised=unsupervised)


class ValidatingTabNetClf(TabNetClassifier):
    def __init__(self, verbose, random_state):
        self.random_state = random_state
        super().__init__(verbose=verbose, seed=random_state)

    def fit(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.10, random_state=42
        )
        super().fit(X_train, y_train, eval_set=[(X_valid, y_valid)])


if __name__ == "__main__":

    models = {
        "tabnet": (ValidatingTabNetClf, dict(verbose=0)),
        "lr": (LogisticRegression, dict(solver="sag", max_iter=1e7)),
        "rf": (RandomForestClassifier, dict()),
    }

    model_name = sys.argv[1]

    if model_name in models:
        clf_cls, kwargs = models[model_name]
    else:
        raise ValueError(f"No model named {sys.argv[1]}. Pick from {models.keys()}")

    if model_name == "tabnet":
        n_jobs = 1  # For some reason parallel runs of tabnet are non-deterministic
    else:
        n_jobs = -1

    combined_data = load_from_disk()

    scores = np.array([])

    for seed in range(0, 5):
        clf = make_pipeline(StandardScaler(), clf_cls(random_state=seed, **kwargs))

        X = combined_data[
            [col for col in combined_data.columns if col != "label"]
        ].values
        y = combined_data["label"].values

        cv_splitter = StratifiedKFold(n_splits=5, shuffle=False)
        curr_scores = cross_val_score(
            clf, X, y, cv=cv_splitter, scoring="roc_auc", n_jobs=n_jobs
        )

        scores = np.concatenate([scores, curr_scores])

    print("Final results:")
    print(f"All scores: {scores}")
    print(f"Score mean: {scores.mean()}")
    print(f"Score std: {scores.std()}")
    print(
        f"95% CI: {st.t.interval(alpha=0.95, df=len(scores)-1, loc=scores.mean(), scale=st.sem(scores))}"
    )
