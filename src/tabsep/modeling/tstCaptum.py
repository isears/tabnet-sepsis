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
    # model.zero_grad()

    X_test = torch.load(f"cache/models/{model_id}/X_test.pt")
    y_test = torch.load(f"cache/models/{model_id}/y_test.pt")

    # # Pos examples only
    # X_test = X_test[y_test == 1]
    # y_test = y_test[y_test == 1]

    dl = torch.utils.data.DataLoader(
        TensorBasedDataset(X_test, y_test),
        batch_size=8,
        num_workers=CORES_AVAILABLE,
        pin_memory=True,
        drop_last=False,
    )

    attributions_list = list()
    pad_mask_list = list()

    for xbatch, _ in dl:
        xbatch = xbatch.to(device)

        pad_masks = xbatch[:, :, -1] == 1
        xbatch = xbatch[:, :, :-1]

        xbatch.requires_grad = True

        # ig = IntegratedGradients(model.forward)
        # attributions = ig.attribute(X_test, additional_forward_args=pad_masks)

        ig = IntegratedGradients(model)  # TODO: are there more modern methods?
        attributions = ig.attribute(xbatch, additional_forward_args=pad_masks, target=0)
        attributions_list.append(attributions.cpu())
        pad_mask_list.append(pad_masks.cpu())

    attributions_all = torch.concat(attributions_list, dim=0)
    pad_masks_all = torch.concat(pad_mask_list, dim=0)
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

    max_absolute_attributions_avg = torch.median(
        max_absolute_attributions, dim=0
    ).values
    importances = pd.DataFrame(
        data={
            "Feature": get_feature_labels(),
            "Median Max Absolute Attribution": max_absolute_attributions_avg.to("cpu")
            .detach()
            .numpy(),
        }
    )

    # Just temporary for topn calculation
    importances["abs"] = importances["Median Max Absolute Attribution"].apply(np.abs)

    topn = importances.nlargest(20, columns="abs")
    topn = topn.drop(columns="abs")

    print(topn)
    ax = sns.barplot(
        x="Feature", y="Median Max Absolute Attribution", data=topn, color="blue"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig("results/importances.png")

    ##########
    # Local attribution of first sepsis patient (idx 49 is longest ICU stay)
    ##########
    preds = torch.load(f"cache/models/{model_id}/preds.pt")

    # # First septic stay w/length > 10 indices
    # sample_idx = np.argmax(
    #     torch.logical_and(y_test == 1, torch.sum(X_test[:, :, -1], dim=1) > 10)
    # )

    # # Most confident predictions
    # sample_idx = torch.argmax(preds)

    # First septic stay w/length > 5, but less than 15 and predicted correctly
    # TODO: we can do better than nested ands...
    sample_idx = np.argmax(
        torch.logical_and(
            torch.logical_and(y_test == 1, torch.sum(X_test[:, :, -1], dim=1) > 3,),
            torch.logical_and(torch.sum(X_test[:, :, -1], dim=1) < 25, preds > 0.5,),
        )
    )

    print(f"Analyzing local importance of idx {sample_idx}")
    sample_case = pd.DataFrame(attributions_all[sample_idx].cpu().detach().numpy())

    sample_case.columns = get_feature_labels()
    # Truncate by padding mask
    sample_case = sample_case[pad_masks_all[sample_idx].tolist()]
    max_absolute_attribution = sample_case.abs().apply(lambda col: col.sum())
    top_n_features = max_absolute_attribution.nlargest(n=20).index

    sample_case = sample_case.drop(
        columns=[c for c in sample_case.columns if c not in top_n_features]
    )

    sample_case.index = (
        sample_case.index * 6
    )  # TODO: this will change if timestep changes

    sample_case.index.name = "Time in ICU (hrs.)"

    ax = sns.heatmap(sample_case, linewidths=0.01, linecolor="black")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(
        f"Validation set idx {sample_idx} prediction {preds[sample_idx]:.2f} actual {y_test[sample_idx]:.2f}"
    )
    plt.savefig("results/local_importance.png", bbox_inches="tight")
