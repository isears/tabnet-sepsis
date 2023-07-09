import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling.skorchTST import AutoPadmaskingTST

if __name__ == "__main__":
    if len(sys.argv) == 2:
        feature_of_interest = sys.argv[1]
    else:
        feature_of_interest = "systolic_bp"

    X = torch.load("cache/TST/X_test.pt")
    y = torch.load("cache/TST/y_test.pt")
    attributions = torch.load("cache/TST/attributions.pt")
    padmasks = AutoPadmaskingTST.autopadmask(X)
    features = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl").features

    feature_idx = features.index(feature_of_interest)

    pos_attr = attributions[y == 1, :, :]
    pos_padmasks = padmasks[y == 1, :]
    seq_lengths = pos_padmasks.sum(dim=1)

    datapoints = list()

    for idx in range(0, 48):
        remaining_valid = pos_attr[seq_lengths > idx, :, :]
        remaining_valid = remaining_valid[:, :, feature_idx]
        gather_indices = (seq_lengths[seq_lengths > idx] - idx - 1).type(torch.int64)

        summed_attribs = torch.sum(remaining_valid[:, gather_indices].diag())
        datapoints.append((summed_attribs / len(remaining_valid)).item())

    plottable = pd.DataFrame(
        data={
            "Hours prior to sepsis onset": range(0, 48),
            "Average attribution": datapoints,
        }
    )

    sns.lineplot(
        data=plottable, x="Hours prior to sepsis onset", y="Average attribution"
    )

    plt.savefig(f"cache/TST/iot_{feature_of_interest}.png")

    print("done")
