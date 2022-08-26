from scipy.stats import spearmanr
from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
from tabsep.reporting.globalImportance import revise_pad
import pandas as pd

import torch
import numpy as np
from scipy.stats import rankdata


# Sum over single icu stays (output (# icustays, # features))
def sum_attrs(att, pm):
    att = att * np.expand_dims(pm, axis=-1)
    summed_att = np.sum(np.sum(np.abs(att), axis=0), axis=0)

    return rankdata(summed_att, method="min")


if __name__ == "__main__":
    att = torch.load(f"{config.model_path}/attributions.pt").detach().numpy()
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    X_train = torch.load(f"{config.model_path}/X_train.pt")
    # shape: # icu stays, seq_len, # features
    X_combined = torch.cat((X_test, X_train), 0).detach().numpy()
    pad_masks = X_combined[:, :, -1]
    X_combined = X_combined[:, :, :-1]
    feature_labels = get_feature_labels()

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

    att = np.delete(att, delete_indices, axis=-1)

    pm, early_pm, late_pm = revise_pad(pad_masks)
    early_importances = sum_attrs(att, early_pm)
    late_importances = sum_attrs(att, late_pm)

    print(spearmanr(early_importances, late_importances))
