import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
from tabsep import config


if __name__ == "__main__":
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    y_test = torch.load(f"{config.model_path}/y_test.pt")
    preds = torch.load(f"{config.model_path}/preds.pt")
    attributions = torch.load(f"{config.model_path}/attributions.pt")

    pad_masks = X_test[:, :, -1] == 1
    # X_test = X_test[:, :, :-1]

    sample_idx = np.argmax(
        torch.logical_and(
            torch.logical_and(y_test == 1, torch.sum(X_test[:, :, -1], dim=1) > 7,),
            torch.logical_and(torch.sum(X_test[:, :, -1], dim=1) < 40, preds > 0.7),
        )
    )

    print(f"Analyzing local importance of idx {sample_idx}")
    sample_case_attrs = pd.DataFrame(attributions[sample_idx].cpu().detach().numpy())
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

    sample_case_attrs.index.name = "Time in ICU (hrs.)"

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(
        sample_case_attrs.transpose(),
        linewidths=0.01,
        linecolor="black",
        # annot=sample_case_values.transpose(),
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(
        f"Validation set idx {sample_idx} prediction {preds[sample_idx]:.2f} actual {y_test[sample_idx]:.2f}"
    )
    plt.savefig("results/local_importance.png", bbox_inches="tight")
