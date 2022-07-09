from captum.attr import (
    IntegratedGradients,
    InputXGradient,
    GuidedGradCam,
    GuidedBackprop,
)
import torch
import os
from tabsep.modeling.tstImpl import TstOneInput, TSTransformerEncoderClassiregressor
from tabsep.modeling.timeseriesCV import TensorBasedDataset
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


CORES_AVAILABLE = len(os.sched_getaffinity(0))


class TensorBasedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
        )


if __name__ == "__main__":
    model_id = "singleTst_2022-07-09_08:26:13"

    if torch.cuda.is_available():
        print("Detected GPU, using cuda")
        device = "cuda"
    else:
        device = "cpu"

    # TODO: sync these params up with trainer
    model = TSTransformerEncoderClassiregressor(
        feat_dim=621,
        d_model=128,
        dim_feedforward=256,
        max_len=120,
        n_heads=16,
        num_classes=1,
        num_layers=3,
    ).to(device)

    model.load_state_dict(
        torch.load(
            f"cache/models/{model_id}/model.pt", map_location=torch.device(device),
        )
    )

    model.eval()
    model.zero_grad()

    X_test = torch.load(f"cache/models/{model_id}/X_test.pt")
    y_test = torch.load(f"cache/models/{model_id}/y_test.pt")

    # # Pos examples only
    # X_test = X_test[y_test == 1]
    # y_test = y_test[y_test == 1]

    dl = torch.utils.data.DataLoader(
        TensorBasedDataset(X_test, y_test),
        batch_size=256,
        num_workers=CORES_AVAILABLE,
        pin_memory=True,
        drop_last=False,
    )

    attributions_list = list()

    for xbatch, _ in dl:
        xbatch = xbatch.to(device)

        pad_masks = xbatch[:, :, -1] == 1
        xbatch = xbatch[:, :, :-1]

        xbatch.requires_grad = True

        # ig = IntegratedGradients(model.forward)
        # attributions = ig.attribute(X_test, additional_forward_args=pad_masks)

        ig = InputXGradient(model)  # TODO: are there more modern methods?
        attributions = ig.attribute(xbatch, additional_forward_args=pad_masks, target=0)
        attributions_list.append(attributions.cpu())

    attributions_all = torch.concat(attributions_list, dim=0)
    print("Got attributions")

    ##########
    # Get top 20 features w/maximum attribution at any point during icustay
    ##########

    # Max over time series dimension, average over batch dimension
    max_attributions = torch.amax(attributions_all, dim=1)
    min_attributions = torch.amin(attributions_all, dim=1)
    min_mask = (
        torch.max(torch.abs(max_attributions), torch.abs(min_attributions))
        > max_attributions
    )
    max_mask = torch.logical_not(min_mask)

    assert (max_mask.int() + min_mask.int() == 1).all()

    max_absolute_attributions = (
        max_attributions * max_mask.int() + min_attributions * min_mask.int()
    )

    max_absolute_attributions_avg = torch.mean(max_absolute_attributions, dim=0)
    importances = pd.DataFrame(
        data={
            "Feature": get_feature_labels(),
            "Average Max Absolute Attribution": max_absolute_attributions_avg.to("cpu")
            .detach()
            .numpy(),
        }
    )

    # Just temporary for topn calculation
    importances["abs"] = importances["Average Max Absolute Attribution"].apply(np.abs)

    topn = importances.nlargest(20, columns="abs")
    topn = topn.drop(columns="abs")

    print(topn)
    ax = sns.barplot(x="Feature", y="Average Max Absolute Attribution", data=topn)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig("results/importances.png")

    ##########
    # Local attribution of first sepsis patient (idx 49 is longest ICU stay)
    ##########
    sample_idx = 49
    sample_case = pd.DataFrame(attributions_all[sample_idx].cpu().detach().numpy())

    sample_case.columns = get_feature_labels()
    # Truncate by padding mask
    sample_case = sample_case[pad_masks[sample_idx].tolist()]
    max_absolute_attribution = sample_case.abs().apply(lambda col: col.max())
    top_n_features = max_absolute_attribution.nlargest(n=20).index

    sample_case = sample_case.drop(
        columns=[c for c in sample_case.columns if c not in top_n_features]
    )

    ax = sns.heatmap(sample_case)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.savefig("results/local_importance.png", bbox_inches="tight")
