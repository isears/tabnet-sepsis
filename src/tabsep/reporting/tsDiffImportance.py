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

    attributions = torch.load(f"{config.model_path}/attributions.pt").detach().numpy()
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    X_train = torch.load(f"{config.model_path}/X_train.pt")
    X_combined = torch.cat((X_test, X_train), 0).detach().numpy()
    pad_masks = X_combined[:, :, -1]
    X_combined = X_combined[:, :, :-1]

    feature_labels = get_feature_labels()

    print("Loaded data")

    # Crude check for static variables
    delete_indices = list()
    for idx, fl in enumerate(feature_labels):
        this_feature_only_vals = X_combined[:, :, idx]
        has_at_least_one_mask = np.logical_not(
            np.all(this_feature_only_vals == 0.0, axis=-1)
        )
        assert len(has_at_least_one_mask) == X_combined.shape[0]
        stays_with_at_least_one = this_feature_only_vals[has_at_least_one_mask]

        num_nonzero = np.sum((stays_with_at_least_one != 0.0).astype("int"), axis=1)
        median_nonzero = np.median(num_nonzero)

        if median_nonzero < 2:
            print(f"Detected static feature: {fl}")
            delete_indices.append(idx)

    attributions = np.delete(attributions, delete_indices, axis=-1)
    ammended_fl = [
        f for idx, f in enumerate(feature_labels) if idx not in delete_indices
    ]

    print(len(ammended_fl))
    print(attributions.shape)

    titles = ["all", "early", "late"]

    pm, early_pm, late_pm = revise_pad(pad_masks)
    early_att = np.multiply(attributions, np.expand_dims(early_pm, axis=-1))
    late_att = np.multiply(attributions, np.expand_dims(late_pm, axis=-1))

    early_att = np.sum(np.sum(early_att, axis=1), axis=0)
    late_att = np.sum(np.sum(late_att, axis=1), axis=0)

    att_diff = late_att - early_att

    df = pd.DataFrame(data={"Feature": ammended_fl, "Delta": att_diff,})
    sns.set_theme()
    ax = sns.barplot(
        x="Delta",
        y="Feature",
        data=df.nlargest(n=20, columns="Delta"),
        orient="h",
        color="b",
    )

    ax.set_title(f"Change in Attributions")
    plt.tight_layout()
    plt.savefig(f"results/delta_importances.png")
    plt.clf()
