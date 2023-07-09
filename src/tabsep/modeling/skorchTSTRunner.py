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

    def captum(self):
        with open(f"{self.save_dir}/model.pkl", "rb") as f:
            model = pickle.load(f).module_

        X = torch.load(f"{self.save_dir}/X_test.pt")
        attributions = captum_runner(model.module_, X)
        torch.save(attributions, f"{self.save_dir}/attributions.pt")


if __name__ == "__main__":
    r = TSTRunner()
    r.parse_cmdline()
