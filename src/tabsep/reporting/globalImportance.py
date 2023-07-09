##########
# Get top 20 features w/maximum attribution at any point during icustay
##########
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from tabsep import config
from tabsep.dataProcessing import LabeledSparseTensor


def summative_bidirectional_importances(att, ordered_features):
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
            "Variable": ordered_features,
            "Positive": sum_pos_attrs.to("cpu").detach().numpy(),
            "Negative": sum_neg_attr.to("cpu").detach().numpy(),
        }
    )

    importances["Summed Absolute Attributions"] = (
        importances["Positive"] + importances["Negative"]
    )

    return importances


def summative_absolute_importances(att, ordered_features):
    att = torch.abs(att)

    sum_attr = torch.sum(torch.sum(att, dim=1), dim=0)

    importances = pd.DataFrame(
        data={
            "Variable": ordered_features,
            "Summed Attribution": sum_attr.to("cpu").detach().numpy(),
        }
    )

    return importances


if __name__ == "__main__":
    model_dir = sys.argv[1]
    attributions = torch.load(f"{model_dir}/attributions.pt").detach()
    ordered_features = LabeledSparseTensor.load_from_pickle(
        "cache/sparse_labeled.pkl"
    ).features

    print("Loaded data")

    # Global importance over entire stay
    importances = summative_absolute_importances(attributions, ordered_features)
    print("Got global importances")
    topn = importances.nlargest(10, columns="Summed Attribution")
    # sns.set_theme()
    sns.set(rc={"figure.figsize": (7, 7)})
    ax = sns.barplot(
        x="Summed Attribution", y="Variable", data=topn, orient="h", color="blue"
    )
    plt.tight_layout()
    plt.savefig(f"{model_dir}/global_importances.png")
    plt.clf()
    topn.to_csv(f"{model_dir}/global_importances.csv", index=False)
