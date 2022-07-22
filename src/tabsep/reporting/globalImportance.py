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


if __name__ == "__main__":

    attributions = torch.load(f"{config.model_path}/attributions.pt")

    # Max over time series dimension, average over batch dimension
    max_attributions = torch.amax(attributions, dim=1)
    min_attributions = torch.amin(attributions, dim=1)
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
