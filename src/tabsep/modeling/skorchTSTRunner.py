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

    tuning_params = {
        "lr": 4.0149853574390136e-05,
        "dropout": 0.1281287413929199,
        "d_model_multiplier": 2,
        "num_layers": 9,
        "n_heads": 8,
        "dim_feedforward": 394,
        "batch_size": 233,
        "pos_encoding": "fixed",
        "activation": "gelu",
        "norm": "BatchNorm",
        "weight_decay": 0,
    }

    def __init__(self, default_cmd="cv") -> None:
        super().__init__(default_cmd)
        conf = TSTConfig(
            save_path="cache/models/skorchCvTst",
            **self.tuning_params,
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
