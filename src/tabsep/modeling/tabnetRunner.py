import os
import pickle

import torch

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import TSTConfig
from tabsep.modeling.baseRunner import BaseModelRunner
from tabsep.modeling.tabnet import BEST_PARAMS, CompatibleTabnet


class TabnetRunner(BaseModelRunner):
    name = "Tabnet"
    save_dir = "cache/Tabnet"

    def __init__(self, default_cmd="cv") -> None:
        self.configured_model_factory = lambda: CompatibleTabnet(**BEST_PARAMS)
        super().__init__(default_cmd)

    def _load_data(self):
        d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
        X = d.get_snapshot()
        y = d.get_labels()

        return X, y

    def captum(self):
        raise NotImplementedError()


if __name__ == "__main__":
    r = TabnetRunner()
    r.parse_cmdline()
