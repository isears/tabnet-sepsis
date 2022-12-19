import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap

from tabsep import config
from tabsep.dataProcessing import get_feature_labels
from tabsep.reporting import pretty_feature_names

if __name__ == "__main__":
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    y_test = torch.load(f"{config.model_path}/y_test.pt")
    preds = torch.load(f"{config.model_path}/preds.pt")
    attributions = torch.load(f"{config.model_path}/attributions.pt")

    pad_masks = X_test[:, :, -1] == 1
    # X_test = X_test[:, :, :-1]

    sample_idx = np.argmax(
        torch.logical_and(
            torch.logical_and(y_test == 1, torch.sum(X_test[:, :, -1], dim=1) > 5,),
            torch.logical_and(torch.sum(X_test[:, :, -1], dim=1) < 25, preds > 0.5),
        )
    )

    print(f"Analyzing local importance of idx {sample_idx}")
    sample_case_attrs_values = attributions[sample_idx].cpu().detach().numpy()
    # Simplify for poster presentation
    sample_case_attrs_values = np.abs(sample_case_attrs_values)
    sample_case_attrs = pd.DataFrame(sample_case_attrs_values)
    sample_case_values = pd.DataFrame(X_test[sample_idx, :, :-1].cpu().detach().numpy())

    sample_case_attrs.columns = get_feature_labels()
    sample_case_values.columns = get_feature_labels()

    # Truncate by padding mask
    sample_case_attrs = sample_case_attrs[pad_masks[sample_idx].tolist()]
    sample_case_values = sample_case_values[pad_masks[sample_idx].tolist()]

    max_absolute_attribution = sample_case_attrs.abs().apply(lambda col: col.sum())
    top_n_features = max_absolute_attribution.nlargest(n=10).index

    sample_case_attrs = sample_case_attrs.drop(
        columns=[c for c in sample_case_attrs.columns if c not in top_n_features]
    )

    sample_case_values = sample_case_values.drop(
        columns=[c for c in sample_case_values.columns if c not in top_n_features]
    )

    sample_case_attrs.index = sample_case_attrs.index * (
        config.timestep_seconds / (60 * 60)
    )
    sample_case_values.index = sample_case_values.index * (
        config.timestep_seconds / (60 * 60)
    )

    sample_case_attrs.index.name = "Time in ICU (hours)"
    sample_case_attrs.index = sample_case_attrs.index.astype(int)
    sample_case_attrs = sample_case_attrs.rename(columns=pretty_feature_names)

    sns.set(font_scale=2)
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(
        sample_case_attrs.transpose(),
        linewidths=0.1,
        linecolor="black",
        # annot=sample_case_values.transpose(),
        # fmt=".2f",
        ax=ax,
        cmap="Reds",
        # cbar=False,
        # cbar_kws={"label": "Attribution (darker = more important)"},
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Attribution Value", labelpad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel("Variable")
    # ax.set_title(
    #     f"Validation set idx {sample_idx} prediction {preds[sample_idx]:.2f} actual {y_test[sample_idx]:.2f}"
    # )
    plt.savefig("results/local_importance.png", bbox_inches="tight")

    plt.clf()

    # Now save a "blank" colorless heatmap for poster
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(
        sample_case_attrs.transpose(),
        linewidths=0.1,
        linecolor="black",
        annot=sample_case_values.transpose(),
        fmt=".2f",
        ax=ax,
        cmap=ListedColormap(["white"]),
        cbar=False,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # ax.set_title(
    #     f"Validation set idx {sample_idx} prediction {preds[sample_idx]:.2f} actual {y_test[sample_idx]:.2f}"
    # )
    plt.savefig("results/local_importance_blank.png", bbox_inches="tight")

    plt.clf()
