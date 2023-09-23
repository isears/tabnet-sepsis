import os
import pickle

import torch

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling.baseRunner import BaseModelRunner
from tabsep.modeling.captumUtil import captum_runner
from tabsep.modeling.skorchNN import nn_factory


class NNRunner(BaseModelRunner):
    name = "NN"
    save_dir = "cache/NN"

    params_by_window = {
        3: {
            "lr": 0.0018927017342011572,
            "n_hidden": 1,
            "width": 84,
            "activation_fn": "relu",
        },
        6: {
            "lr": 0.0018927017342011572,
            "n_hidden": 1,
            "width": 84,
            "activation_fn": "relu",
        },
        12: {
            "lr": 0.0018927017342011572,
            "n_hidden": 1,
            "width": 84,
            "activation_fn": "relu",
        },
        24: {
            "lr": 0.0018927017342011572,
            "n_hidden": 1,
            "width": 84,
            "activation_fn": "relu",
        },
    }

    def __init__(self, default_cmd="cv") -> None:
        super().__init__(default_cmd)

        self.configured_model_factory = lambda: nn_factory(
            **self.params_by_window[self.prediction_window]
        )

    def _load_data(self):
        d = LabeledSparseTensor.load_from_pickle(self.data_src)
        X = d.get_snapshot_los()
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
    r = NNRunner(default_cmd="train")
    r.parse_cmdline()
