import os
import pickle
import sys

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabsep.modeling import CVResults


class BaseModelRunner:
    save_dir: str
    name: str
    configured_model_factory: callable

    def __init__(self, default_cmd="cv") -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self.default_cmd = default_cmd
        self.data_src = "cache/sparse_labeled_12.pkl"

    def _load_data(self):
        raise NotImplementedError()

    def _save_model(self, m) -> None:
        with open(f"{self.save_dir}/{self.data_src_label}_model.pkl", "wb") as f:
            pickle.dump(m, f)

    def parse_cmdline(self):
        if len(sys.argv) == 1:
            cmd = self.default_cmd
        else:
            cmd = sys.argv[1]

        if len(sys.argv) == 3:
            self.data_src = sys.argv[2]

        self.data_src_label = self.data_src.split("/")[-1].split(".")[0]
        print(f"[+] Runner instantiated with cmd {cmd} and data {self.data_src}")
        f = getattr(self, cmd)
        f()

    def cv(self):
        X, y = self._load_data()

        skf = StratifiedKFold(n_splits=10)
        res = CVResults()

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            model = self.configured_model_factory()

            print(f"[CrossValidation] Starting fold {fold_idx}")

            model.fit(X[train_idx], y[train_idx])
            preds = model.predict_proba(X[test_idx])[:, 1]

            res.add_result(y[test_idx], preds)

        res.print_report()
        res.save_report(f"{self.save_dir}/{self.data_src_label}_cvresult.pkl")

        return res

    def train(self):
        X, y = self._load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        m = self.configured_model_factory()
        m.fit(X_train, y_train)
        preds = m.predict_proba(X_test)[:, 1]

        print(f"Final AUROC: {roc_auc_score(y_test, preds)}")
        print(f"Final AUPRC: {average_precision_score(y_test, preds)}")

        self._save_model(m)

        torch.save(X_train, f"{self.save_dir}/{self.data_src_label}_X_train.pt")
        torch.save(X_test, f"{self.save_dir}/{self.data_src_label}_X_test.pt")
        torch.save(y_train, f"{self.save_dir}/{self.data_src_label}_y_train.pt")
        torch.save(y_test, f"{self.save_dir}/{self.data_src_label}_y_test.pt")
        torch.save(
            torch.Tensor(preds), f"{self.save_dir}/{self.data_src_label}_preds.pt"
        )
