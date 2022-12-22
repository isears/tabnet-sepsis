##########
# Get top 20 features w/maximum attribution at any point during icustay
##########
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
from tabsep.reporting import pretty_feature_names


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
            "Variable": get_feature_labels(),
            "Positive": sum_pos_attrs.to("cpu").detach().numpy(),
            "Negative": sum_neg_attr.to("cpu").detach().numpy(),
        }
    )

    importances["Summed Absolute Attributions"] = (
        importances["Positive"] + importances["Negative"]
    )

    return importances


if __name__ == "__main__":

    attributions = torch.load(f"{config.tst_path}/attributions.pt").detach()

    print("Loaded data")

    # Global importance over entire stay
    importances = summative_importances(attributions)
    print("Got global importances")
    topn = importances.nlargest(20, columns="Summed Absolute Attributions")
    topn["Variable"] = topn["Variable"].apply(
        lambda x: pretty_feature_names[x] if x in pretty_feature_names else x
    )

    plottable_topn = topn[["Variable", "Positive", "Negative"]].melt(
        id_vars="Variable",
        var_name="Attribution Direction",
        value_name="Summed Attribution",
    )
    sns.set_theme()

    sns.set(rc={"figure.figsize": (15, 15)})
    ax = sns.barplot(
        x="Summed Attribution",
        y="Variable",
        data=plottable_topn,
        hue="Attribution Direction",
        orient="h",
        palette="Set1",
    )
    ax.set_title(f"Global Importance (Time Series Transformer Attributions)")
    plt.tight_layout()
    plt.savefig(f"results/global_importances_tst.png")
    plt.clf()
    topn.to_csv(f"results/global_importances_tst.csv", index=False)

    # Now do LR
    lr_odds_ratios = pd.read_csv(f"{config.lr_path}/odds_ratios.csv")
    lr_odds_ratios["Absolute Coefficients"] = np.abs(lr_odds_ratios["Coefficients"])

    topn = lr_odds_ratios.nlargest(20, columns="Absolute Coefficients")
    topn["Variable"] = topn["Variable"].apply(
        lambda x: pretty_feature_names[x] if x in pretty_feature_names else x
    )
    ax = sns.barplot(x="Coefficients", y="Variable", data=topn, orient="h", color="b")
    ax.set_title(f"Global Importance (Logistic Regression Coefficients)")

    plt.tight_layout()
    plt.savefig(f"results/global_importances_lr.png")
    plt.clf()
