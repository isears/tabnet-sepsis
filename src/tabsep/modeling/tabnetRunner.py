import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
from tabsep.modeling.tabnet import CompatibleTabnet


class TabnetRunner(BaseModelRunner):
    name = "Tabnet"
    save_dir = "cache/Tabnet"

    tuning_params = {
        "n_d": 43,
        "n_steps": 3,
        "gamma": 1.0197386155084536,
        "n_independent": 1,
        "momentum": 0.0650212053821212,
        "mask_type": "entmax",
        "optimizer_lr": 0.008125514275405706,
        "fit_batch_size": 256,
    }

    def __init__(self, default_cmd="cv") -> None:
        super().__init__(default_cmd)
        self.configured_model_factory = lambda: CompatibleTabnet(**self.tuning_params)

    def _load_data(self):
        d = LabeledSparseTensor.load_from_pickle(self.data_src)
        X = d.get_snapshot_los()
        y = d.get_labels()

        return X, y

    def captum(self):
        with open(f"{self.save_dir}/{self.data_src_label}_model.pkl", "rb") as f:
            trained_model = pickle.load(f)

        X = torch.load(f"{self.save_dir}/{self.data_src_label}_X_test.pt").to("cuda")
        y = torch.load(f"{self.save_dir}/{self.data_src_label}_y_test.pt")

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
        torch.save(
            attributions, f"{self.save_dir}/{self.data_src_label}_attributions.pt"
        )

    def global_importance(self):
        attributions = torch.load(
            f"{self.save_dir}/{self.data_src_label}_attributions.pt"
        )
        ordered_features = LabeledSparseTensor.load_from_pickle(
            f"cache/{self.data_src_label}.pkl"
        ).features + ["los"]
        importances = torch.sum(torch.abs(attributions), dim=0)

        importances = pd.DataFrame(
            data={
                "Variable": ordered_features,
                "Summed Attribution": torch.sum(torch.abs(attributions), dim=0)
                .cpu()
                .detach()
                .numpy(),
            }
        )

        topn = importances.nlargest(10, columns="Summed Attribution")
        # sns.set_theme()
        sns.set(rc={"figure.figsize": (7, 7)})
        ax = sns.barplot(
            x="Summed Attribution", y="Variable", data=topn, orient="h", color="blue"
        )
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{self.data_src_label}_global_importances.png")
        plt.clf()
        topn.to_csv(
            f"{self.save_dir}/{self.data_src_label}_global_importances.csv", index=False
        )

        print("done")


if __name__ == "__main__":
    r = TabnetRunner(default_cmd="train")
    r.parse_cmdline()
