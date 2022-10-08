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


def summative_importances(att):
    """
    Aggregate importances by summing up all their attributions
    """

    # Separate into negative and positive attrs
    neg_mask = att < 0.0
    neg_attrs = att * neg_mask * -1
    pos_attrs = att * torch.logical_not(neg_mask)

    # Sum over time series dimension, then over batch dimension
    sum_neg_attr = torch.sum(torch.sum(neg_attrs, dim=1), dim=0)
    sum_neg_attr = sum_neg_attr
    sum_pos_attrs = torch.sum(torch.sum(pos_attrs, dim=1), dim=0)

    importances = pd.DataFrame(
        data={
            "Feature": get_feature_labels(),
            "Positive": sum_pos_attrs.to("cpu").detach().numpy(),
            "Negative": sum_neg_attr.to("cpu").detach().numpy(),
        }
    )

    importances["Summed Absolute Attributions"] = (
        importances["Positive"] + importances["Negative"]
    )

    return importances


if __name__ == "__main__":

    attributions = torch.load(f"{config.model_path}/attributions.pt").detach()
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    X_train = torch.load(f"{config.model_path}/X_train.pt")
    X_combined = torch.cat((X_test, X_train), 0).detach()

    print("Loaded data")

    pad_masks = X_combined[:, :, -1]
    pm, early_pm, late_pm = revise_pad(pad_masks)

    # Global importance over entire stay
    att = np.multiply(attributions, np.expand_dims(pm, axis=-1))
    importances = summative_importances(att)
    print("Got global importances")
    topn = importances.nlargest(20, columns="Summed Absolute Attributions")

    sns.set_theme()

    ax = sns.barplot(
        x="Summed Absolute Attributions", y="Feature", data=topn, orient="h", color="b"
    )
    ax.set_title(f"Global Feature Importance")
    plt.tight_layout()
    plt.savefig(f"results/global_importances.png")
    plt.clf()
    topn.to_csv(f"results/global_importances.csv", index=False)

    # Early vs. Late importance
    early_att = np.multiply(attributions, np.expand_dims(early_pm, axis=-1))
    early_importances = summative_importances(early_att)
    print("Got early importances")
    early_importances["Window"] = "Early"

    late_att = np.multiply(attributions, np.expand_dims(late_pm, axis=-1))
    late_importances = summative_importances(late_att)
    print("Got late importances")
    late_importances["Window"] = "Late"

    plottable = pd.concat([early_importances, late_importances])
    plottable = plottable[plottable["Feature"].isin(topn["Feature"])]

    ax = sns.barplot(
        x="Summed Absolute Attributions",
        y="Feature",
        data=plottable,
        orient="h",
        color="b",
        hue="Window",
        palette={"Early": "tab:blue", "Late": "tab:red"},
        order=topn["Feature"],
    )
    ax.set_title("Early vs. Late Feature Importance")
    plt.tight_layout()
    plt.savefig("results/evl_importances.png")

    print("[+] Done")
