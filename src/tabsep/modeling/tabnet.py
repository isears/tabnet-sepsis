from typing import Any

import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import average_precision_score


class AUPRC(Metric):
    def __init__(self):
        self._name = "AUPRC"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return average_precision_score(y_true, y_score[:, 1])


class CompatibleTabnet(TabNetClassifier):
    def __init__(self, **kwargs):
        # Convert tuning params to constructor params
        constructor_args = {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("optimizer_") and not k.startswith("fit_")
        }

        if "n_a" in constructor_args:
            constructor_args["n_a"] = constructor_args["n_d"]

        optimizer_params = {
            k[len("optimizer_") :]: v
            for k, v in kwargs.items()
            if k.startswith("optimizer_")
        }
        constructor_args["optimizer_params"] = optimizer_params

        self.fit_params = {
            k[len("fit_") :]: v for k, v in kwargs.items() if k.startswith("fit_")
        }

        super().__init__(**constructor_args)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.numpy()
        y = y.numpy()

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        super().fit(
            X_train,
            y_train,
            patience=5,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["auc", "logloss", AUPRC],
            **self.fit_params
        )

    # For captum compatibility
    def __call__(self, X):
        # Based on: https://github.com/dreamquark-ai/tabnet/blob/9ba89918ab6e6dec4f9ea10060b005fe64e7f9ef/pytorch_tabnet/tab_model.py#L102C13-L103C81
        output, _ = self.network(X)
        predictions = torch.nn.Softmax(dim=1)(output)
        return torch.unsqueeze(predictions[:, 1], dim=-1)
