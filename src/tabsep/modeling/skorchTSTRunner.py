import os
import pickle

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import TSTConfig
from tabsep.modeling.baseRunner import BaseModelRunner
from tabsep.modeling.commonCaptum import captum_runner
from tabsep.modeling.skorchTST import BEST_PARAMS, tst_factory


class TSTRunner(BaseModelRunner):
    name = "TST"
    save_dir = "cache/TST"

    def __init__(self, default_cmd="cv") -> None:
        conf = TSTConfig(save_path="cache/models/skorchCvTst", **BEST_PARAMS)
        self.configured_model_factory = lambda: tst_factory(conf)
        super().__init__(default_cmd)

    def _load_data(self):
        d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
        X = d.get_dense_normalized()
        y = d.get_labels()

        return X, y

    def train(self):
        X, y = self._load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        m = tst_factory(TSTConfig(save_path="cache/models/skorchCvTst", **BEST_PARAMS))

        m.fit(X_train, y_train)
        preds = m.predict_proba(X_test)[:, 1]

        print(f"Final AUROC: {roc_auc_score(y_test, preds)}")
        print(f"Final AUPRC: {average_precision_score(y_test, preds)}")

        with open(f"{self.save_dir}/model.pkl", "wb") as f:
            m.module_ = m.module_.to("cpu")
            pickle.dump(m, f)

        torch.save(X_train, f"{self.save_dir}/X_train.pt")
        torch.save(X_test, f"{self.save_dir}/X_test.pt")
        torch.save(y_train, f"{self.save_dir}/y_train.pt")
        torch.save(y_test, f"{self.save_dir}/y_test.pt")
        torch.save(torch.Tensor(preds), f"{self.save_dir}/preds.pt")

        print(f"[+] Saved to {self.save_dir}")

    def captum(self):
        with open(f"{self.save_dir}/model.pkl", "rb") as f:
            model = pickle.load(f)

        X = torch.load(f"{self.save_dir}/X_test.pt")
        attributions = captum_runner(model.module_, X)
        torch.save(attributions, f"{self.save_dir}/attributions.pt")


if __name__ == "__main__":
    r = TSTRunner()
    r.parse_cmdline()
