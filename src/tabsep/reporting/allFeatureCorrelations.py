from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.feature_selection import mutual_info_regression

from tabsep import config
from tabsep.dataProcessing import get_feature_labels

if __name__ == "__main__":
    attributions = torch.load(f"{config.model_path}/attributions.pt").detach().numpy()
    labels = get_feature_labels()

    # Sum of absolute values of attributions over entire time series
    # Output is of shape (# icustays, # features)
    abssum_attrs = np.sum(np.abs(attributions), axis=1)

    # Desired result is a matrix of shape (# features, # features) with mutual info at every point
    mutual_info_heatmap = np.zeros((abssum_attrs.shape[-1], abssum_attrs.shape[-1]))

    all_procs = list()

    with ProcessPoolExecutor(max_workers=config.cores_available) as executor:

        def get_mi(feat_attrs, label_idx):
            mi = mutual_info_regression(abssum_attrs, feat_attrs, random_state=42)
            return mi, label_idx

        for idx in range(0, abssum_attrs.shape[-1]):
            # In theory, copy-on-write should protect from memory explosion
            # May not work on Windows
            all_procs.append(executor.submit(get_mi, abssum_attrs[:, idx], idx))

        print(f"[*] All {len(all_procs)} jobs submitted, waiting for resutls...")

        for proc in all_procs:
            mi, idx = proc.result()
            mutual_info_heatmap[idx] = mi
            print(f"[+] Got result for feature {idx} ({labels[idx]})")

    print(f"[*] Set diagonal to 0.0...")

    # Manually set the diagonal to 0, b/c not interested in mi of feature with itself
    for idx in range(0, mutual_info_heatmap.shape[0]):
        mutual_info_heatmap[idx, idx] = 0.0

    ax = sns.heatmap(mutual_info_heatmap)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.savefig("results/feature_correlation_heatmap.png", bbox_inches="tight")

    top_n = 100

    for idx in range(0, top_n):
        max_index = np.unravel_index(
            np.argmax(mutual_info_heatmap), mutual_info_heatmap.shape
        )
        print(
            f"{labels[max_index[0]]} <-> {labels[max_index[1]]} ({mutual_info_heatmap[max_index]})"
        )

        mutual_info_heatmap[max_index] = 0.0
        mutual_info_heatmap[max_index[1], max_index[0]] = 0.0
