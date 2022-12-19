import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import ttest_ind

from tabsep import config
from tabsep.dataProcessing import get_feature_labels
from tabsep.reporting import pretty_feature_names


def get_early_late_means(single_stay_atts):
    assert single_stay_atts.ndim == 1

    original_len = len(single_stay_atts)
    assert original_len < 121  # Max stay length

    halfway_pt = original_len // 2

    if original_len % 2 == 0:  # Leave out middle if odd
        early_atts = single_stay_atts[:halfway_pt]
        late_atts = single_stay_atts[halfway_pt:]
    else:
        early_atts = single_stay_atts[:halfway_pt]
        late_atts = single_stay_atts[halfway_pt + 1 :]

    return np.mean(early_atts), np.mean(late_atts)


if __name__ == "__main__":
    att = torch.load(f"{config.model_path}/attributions.pt").detach().numpy()
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    X_train = torch.load(f"{config.model_path}/X_train.pt")
    # shape: # icu stays, seq_len, # features
    X_combined = torch.cat((X_test, X_train), 0).detach().numpy()
    pad_masks = X_combined[:, :, -1]
    X_combined = X_combined[:, :, :-1]
    feature_labels = get_feature_labels()

    # isolate only top 20 features
    # top = pd.read_csv("results/global_importances.csv")["Variable"].to_list()
    # inv_map = {v: k for k, v in pretty_feature_names.items()}
    # top_raw = [inv_map[n] if n in inv_map else n for n in top]

    # kept_feature_indices = [feature_labels.index(n) for n in top_raw]

    # att = att[:, :, kept_feature_indices]

    # Collapse features dimension (sum), new shape (# icu stays, seq_len)
    # att = np.sum(np.abs(att), axis=-1)
    # att = np.abs(att)

    ttests = list()

    for feature_idx in range(0, att.shape[-1]):
        this_feature_att = att[:, :, feature_idx]
        att_distribution = np.zeros(100)

        early_mean_atts = list()
        late_mean_atts = list()

        for stay_idx in range(0, att.shape[0]):  # Iterate over all stays
            stay_len = int(np.sum(pad_masks[stay_idx]))
            trimmed = this_feature_att[stay_idx, 0:stay_len]

            early_mean_att, late_mean_att = get_early_late_means(trimmed)

            early_mean_atts.append(early_mean_att)
            late_mean_atts.append(late_mean_att)

        ttests.append(ttest_ind(early_mean_atts, late_mean_atts))

    earlier_important_count = 0
    later_important_count = 0
    insignificant_count = 0

    for idx, res in enumerate(ttests):
        stat, pval = res

        if pval > 0.05 / 621:  # Multiple hypothesis correction
            # print(f"{top_raw[idx]} has no skew")

            insignificant_count += 1
            continue

        elif stat < 0:
            # print(f"{top_raw[idx]} skews later")
            later_important_count += 1
        elif stat > 0:
            # print(f"{top_raw[idx]} skews earlier")
            earlier_important_count += 1

    print(earlier_important_count)
    print(later_important_count)
    print(insignificant_count)
