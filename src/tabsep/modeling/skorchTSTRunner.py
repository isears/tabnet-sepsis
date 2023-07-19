import os
import pickle

import torch

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import TSTConfig
from tabsep.modeling.baseRunner import BaseModelRunner
from tabsep.modeling.captumUtil import captum_runner
from tabsep.modeling.skorchTST import BEST_PARAMS, tst_factory


class TSTRunner(BaseModelRunner):
    name = "TST"
    save_dir = "cache/TST"

    params_by_window = {
        12: {
            "lr": 2.1256858803320374e-05,
            "dropout": 0.11238558423398054,
            "d_model_multiplier": 4,
            "num_layers": 6,
            "n_heads": 8,
            "dim_feedforward": 342,
            "batch_size": 101,
            "pos_encoding": "fixed",
            "activation": "relu",
            "norm": "BatchNorm",
            "weight_decay": 0.01,
        }
    }

    def __init__(self, default_cmd="cv") -> None:
        super().__init__(default_cmd)
        conf = TSTConfig(
            save_path="cache/models/skorchCvTst",
            **self.params_by_window[self.prediction_window],
        )
        self.configured_model_factory = lambda: tst_factory(conf)

    def _load_data(self):
        d = LabeledSparseTensor.load_from_pickle(self.data_src)
        X = d.get_dense_normalized()
        y = d.get_labels()

        return X, y

    def captum(self):
        with open(f"{self.save_dir}/{self.data_src_label}_model.pkl", "rb") as f:
            model = pickle.load(f).module_

        X = torch.load(f"{self.save_dir}/{self.data_src_label}_X_test.pt")

        attributions = captum_runner(model, X)
        torch.save(
            attributions, f"{self.save_dir}/{self.data_src_label}_attributions.pt"
        )


if __name__ == "__main__":
    r = TSTRunner(default_cmd="train")
    r.parse_cmdline()
