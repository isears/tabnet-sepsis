##########
# Get top 20 features w/maximum attribution at any point during icustay
##########
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
from tabsep import config


def revise_pad(pm):
    cut_early = np.copy(pm)
    cut_late = np.copy(pm)

    for idx in range(0, pm.shape[0]):
        mask_len = int(pm[idx, :].sum())
        half_mask = mask_len // 2

        cut_early[idx, :] = np.concatenate(
            [np.ones(half_mask), np.zeros(pm.shape[1] - half_mask)]
        )

        cut_late[idx, :] = np.concatenate(
            [
                np.zeros(half_mask),
                np.ones(half_mask),
                np.zeros(pm.shape[1] - (2 * half_mask)),
            ]
        )

    return [pm, cut_early, cut_late]


if __name__ == "__main__":

    attributions = torch.load(f"{config.model_path}/attributions.pt").detach()
    X_test = torch.load(f"{config.model_path}/X_test.pt").detach()
    pad_masks = X_test[:, :, -1]

    titles = ["all", "early", "late"]

    for revised_pad, title in zip(revise_pad(pad_masks), titles):
        att = np.multiply(attributions, np.expand_dims(revised_pad, axis=-1))

        # Max over time series dimension, average over batch dimension
        max_attributions = torch.amax(att, dim=1)
        min_attributions = torch.amin(att, dim=1)
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
                "Median Max Absolute Attribution": max_absolute_attributions_avg.to(
                    "cpu"
                )
                .detach()
                .numpy(),
            }
        )

        # Just temporary for topn calculation
        importances["abs"] = importances["Median Max Absolute Attribution"].apply(
            np.abs
        )

        topn = importances.nlargest(20, columns="abs")
        topn = topn.drop(columns="abs")

        print(topn)
        ax = sns.barplot(
            x="Feature", y="Median Max Absolute Attribution", data=topn, color="blue"
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_title(f"Global Feature Importance ({title})")
        plt.tight_layout()
        plt.savefig(f"results/{title}_importances.png")
        plt.clf()
