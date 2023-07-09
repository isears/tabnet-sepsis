import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split

BEST_PARAMS = {
    "n_d": 63,
    "n_a": 61,
    "n_steps": 9,
    "gamma": 1.3614767469824574,
    "n_independent": 1,
    "momentum": 0.19130622869056657,
    "mask_type": "entmax",
    "optimizer_params": {"lr": 0.01147088774852625},
}


class CompatibleTabnet(TabNetClassifier):
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.numpy()
        y = y.numpy()

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

        super().fit(
            X_train,
            y_train,
            patience=3,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["logloss", "auc"],
        )
