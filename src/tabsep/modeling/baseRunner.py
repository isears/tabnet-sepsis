import os
import sys
from typing import Tuple

import torch
from sklearn.model_selection import StratifiedKFold

from tabsep.modeling import CVResults


class BaseModelRunner:
    save_dir: str
    name: str
    configured_model_factory: callable

    def __init__(self, default_cmd="cv") -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self.default_cmd = default_cmd

    def _load_data(self) -> Tuple(torch.Tensor, torch.Tensor):
        raise NotImplementedError()

    def parse_cmdline(self):
        if len(sys.argv) == 1:
            cmd = self.default_cmd
        else:
            cmd = sys.argv[1]

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
        res.save_report(f"{self.save_dir}/cvresult.pkl")

        return res

    def tuning(self):
        raise NotImplementedError()

    def importance(self):
        raise NotImplementedError()
