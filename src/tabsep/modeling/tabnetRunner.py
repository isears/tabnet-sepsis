import os
import pickle

import torch
from captum.attr import (
    DeepLift,
    FeaturePermutation,
    GuidedBackprop,
    GuidedGradCam,
    InputXGradient,
    IntegratedGradients,
    Saliency,
    ShapleyValueSampling,
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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
        with open(f"{self.save_dir}/model.pkl", "rb") as f:
            trained_model = pickle.load(f)

        X = torch.load(f"{self.save_dir}/X_test.pt").to("cuda")
        y = torch.load(f"{self.save_dir}/y_test.pt")

        trained_model.network.eval()

        preds = trained_model(X)

        print(f"[*] Sanity check:")
        print(roc_auc_score(y, preds.detach().cpu().numpy()))

        X.requires_grad = True

        attributor = Saliency(trained_model)
        attributions = list()
        batch_size = 256

        for batch_idx in tqdm(range(0, X.shape[0], batch_size)):
            end_idx = min(batch_idx + batch_size, X.shape[0])
            attributions.append(attributor.attribute(X[batch_idx:end_idx], target=0))

        attributions = torch.concat(attributions, dim=0)

        assert attributions.shape == X.shape
        torch.save(attributions, f"{self.save_dir}/attributions.pt")


if __name__ == "__main__":
    r = TabnetRunner(default_cmd="captum")
    r.parse_cmdline()
