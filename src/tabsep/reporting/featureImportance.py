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
import pickle
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from tabsep.dataProcessing import LabeledSparseTensor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Indices of features by source mimic table
featuregroup_indices = {
    "Vital Signs": [0, 1, 2, 3, 4, 5],
    "Blood Gas": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "Chemistry": [6] + list(range(22, 33)),
    "Complete Blood Count": [18, 19, 20, 21] + list(range(39, 75)),
    "Coagulation Panel": list(range(33, 39)),
    "Ventilator Settings": list(range(75, 85)),
}


def build_attributions(model_name: str, window: int, batch_size: int = 8):
    if os.path.exists(f"cache/{model_name}/attribs_{window}.pkl"):
        return torch.load(f"cache/{model_name}/attribs_{window}.pkl")

    print(f"[*] Attributing {model_name}.{window}...")
    with open(f"cache/{model_name}/sparse_labeled_{window}_model.pkl", "rb") as f:
        trained_model = pickle.load(f)

    X = torch.load(f"cache/{model_name}/sparse_labeled_{window}_X_test.pt").to("cuda")
    y = torch.load(f"cache/{model_name}/sparse_labeled_{window}_y_test.pt")

    if model_name == "TST":
        trained_model.module_.eval()
        m = trained_model.module_
    elif model_name == "Tabnet":
        trained_model.network.eval()
        m = trained_model

    X.requires_grad = True

    attributor = IntegratedGradients(m)
    attributions = list()

    for batch_idx in tqdm(range(0, X.shape[0], batch_size)):
        end_idx = min(batch_idx + batch_size, X.shape[0])
        attributions.append(attributor.attribute(X[batch_idx:end_idx], target=0))

    attributions = torch.concat(attributions, dim=0)

    assert attributions.shape == X.shape

    torch.save(attributions, f"cache/{model_name}/attribs_{window}.pkl")
    return attributions


def plot_top_importances(
    path: str,
    attribs: torch.Tensor,
    n=10,
    y_axis_name="Variable",
    x_axis_name="Importance",
):
    data = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled_3.pkl")

    # For tabnet, last feature is LOS
    if len(attribs) > len(data.features):
        features = data.features + ["Length of Stay"]
    else:
        features = data.features

    plottable = pd.DataFrame(
        data={y_axis_name: features, x_axis_name: attribs.cpu().detach().numpy()}
    )

    plottable = plottable.nlargest(10, columns=x_axis_name)

    sns.set(rc={"figure.figsize": (10, 10)})
    sns.set_theme()
    ax = sns.barplot(
        x=x_axis_name, y=y_axis_name, data=plottable, orient="h", color="b"
    )
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


def plot_groupped_importances(path: str, tst_attribs: torch.Tensor, tabnet_attribs):
    tst_df = pd.DataFrame(data={"Importance": tst_attribs.cpu().detach().numpy()})
    tabnet_df = pd.DataFrame(
        data={"Importance": tabnet_attribs[0:85].cpu().detach().numpy()}
    )

    tst_df["Model"] = "Time Series Transformer"
    tabnet_df["Model"] = "TabNet"

    plottable = pd.concat([tst_df, tabnet_df])

    for name, idx in featuregroup_indices.items():
        plottable.loc[idx, "Feature"] = name

    assert not plottable["Feature"].isna().any()

    plottable = plottable.groupby(["Feature", "Model"], as_index=False).agg(
        {"Importance": "mean"}
    )
    print(plottable)
    sns.set(rc={"figure.figsize": (10, 10)})
    sns.set_theme()
    ax = sns.barplot(
        x="Feature",
        y="Importance",
        data=plottable,
        hue="Model",
        hue_order=["Time Series Transformer", "TabNet"],
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


if __name__ == "__main__":
    for window in [3, 24]:
        tabnet_atts = build_attributions("Tabnet", window, 256)
        tst_atts = build_attributions("TST", window, 10)

        # Get absolute importances
        tst_absolute = torch.sum(torch.sum(torch.abs(tst_atts), dim=1), dim=0)
        tabnet_absolute = torch.sum(torch.abs(tabnet_atts), dim=0)

        tst_normal = tst_absolute / torch.sum(tst_absolute)
        tabnet_normal = tabnet_absolute / torch.sum(tabnet_absolute)

        plot_groupped_importances(
            f"results/importances_{window}.png", tst_normal, tabnet_normal
        )

    print("[+] Done")
