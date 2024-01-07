"""
Identify the set of patients that were true-positives for the TST but also false-negatives for TabNet

Compare this group with the group of patients that were true-positives for both models

Threshold set to maximize differences
"""
import torch
from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.reporting.featureImportance import build_attributions
import numpy as np
from scipy.stats import ttest_ind_from_stats
import pandas as pd


def compare(a, b):
    m_a = np.nanmean(a, axis=0)
    std_a = np.nanstd(a, axis=0)

    m_b = np.nanmean(b, axis=0)
    std_b = np.nanstd(b, axis=0)

    stat, p = ttest_ind_from_stats(
        m_a,
        std_a,
        a.shape[0],
        m_b,
        std_b,
        b.shape[0],
    )

    feature_labels = LabeledSparseTensor.load_from_pickle(
        "cache/sparse_labeled_3.pkl"
    ).features + ["los"]

    sig_df = pd.DataFrame(
        data={
            "feature": feature_labels,
            "mean_tst_better": m_a,
            "mean_other": m_b,
            "p": p,
        }
    )

    sig_df["p_adj"] = sig_df["p"] * len(sig_df)

    return sig_df


if __name__ == "__main__":
    X_tst = torch.load(f"cache/TST/sparse_labeled_3_X_test.pt")
    X_tabnet = torch.load(f"cache/Tabnet/sparse_labeled_3_X_test.pt")

    y_tst = torch.load(f"cache/TST/sparse_labeled_3_y_test.pt").to("cpu")
    preds_tst = torch.load(f"cache/TST/sparse_labeled_3_preds.pt").to("cpu")

    y_tabnet = torch.load(f"cache/Tabnet/sparse_labeled_3_y_test.pt").to("cpu")
    preds_tabnet = torch.load(f"cache/Tabnet/sparse_labeled_3_preds.pt").to("cpu")

    # Reduce timeseries to snapshot
    X_tst = LabeledSparseTensor.get_snapshot_los_util(X_tst)
    # Make sure we're dealing with the same data
    assert (X_tst == X_tabnet).all()
    assert (y_tst == y_tabnet).all()
    y = y_tst
    X = X_tst
    X[X == -1] = torch.nan

    preds_tst_pos = preds_tst[y == 1]
    preds_tabnet_pos = preds_tabnet[y == 1]
    X_pos = X[y == 1]

    max_n = 0
    opt_threshold = None
    for idx in range(1, 100):
        threshold = idx / 100
        n = X_pos[
            torch.logical_and(preds_tst_pos > threshold, preds_tabnet_pos < threshold)
        ].shape[0]

        if n > max_n:
            max_n = n
            opt_threshold = threshold

    print(f"[*] Calibrated models at threshold value of {opt_threshold}")

    X_tst_better = X_pos[
        torch.logical_and(
            preds_tst_pos > opt_threshold, preds_tabnet_pos < opt_threshold
        )
    ].numpy()

    X_other = X_pos[
        ~torch.logical_and(
            preds_tst_pos > opt_threshold, preds_tabnet_pos < opt_threshold
        )
    ].numpy()

    feature_comparison = compare(X_tst_better, X_other)
    print(feature_comparison.nsmallest(20, columns="p_adj"))

    # What features were most important in this subgroup?
    attribs_tst = build_attributions("TST", 3, 10).abs().sum(dim=1)
    attribs_tst = (
        attribs_tst[y == 1][
            torch.logical_and(
                preds_tst_pos > opt_threshold, preds_tabnet_pos < opt_threshold
            )
        ]
        .detach()
        .cpu()
        .numpy()
    )

    attribs_tst = attribs_tst / attribs_tst.sum()

    attribs_tabnet = build_attributions("Tabnet", 3, 256).abs()
    attribs_tabnet = (
        attribs_tabnet[y == 1][
            torch.logical_and(
                preds_tst_pos > opt_threshold, preds_tabnet_pos < opt_threshold
            )
        ]
        .detach()
        .cpu()
        .numpy()
    )

    attribs_tabnet = attribs_tabnet / attribs_tabnet.sum()

    # Add one feature onto TST (LOS)
    importance_comparison = compare(
        np.hstack([attribs_tst, np.zeros((attribs_tst.shape[0], 1))]),
        attribs_tabnet,
    )

    print("=" * 10)

    print(importance_comparison.nsmallest(20, columns="p_adj"))
