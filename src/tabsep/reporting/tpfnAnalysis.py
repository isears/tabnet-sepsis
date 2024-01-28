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
import matplotlib.pyplot as plt
import seaborn as sns

PRETTY_NAME_MAP = {
    "aado2": "Alveolar-arterial Oxygen Gradient",
    "fibrinogen": "Fibrinogen",
    "ReticulocyteCountAutomated": "Automated Reticulocyte Count",
    "ReticulocyteCountAbsolute": "Absolute Reticulocyte Count",
    "AbsoluteEosinophilCount": "Absolute Eosinophil Count",
    "NucleatedRedCells": "Nucleated Red Cells",
    "heart_rate": "Heart Rate",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "resp_rate": "Respiration Rate",
    "creatinine": "Creatinine",
    "PlateletCount": "Platelet Count",
    "los": "Length of Stay",
}


def plotdiff(tst_better, other, feats):
    feature_labels = LabeledSparseTensor.load_from_pickle(
        "cache/sparse_labeled_3.pkl"
    ).features + ["los"]

    df_a = pd.DataFrame(data=tst_better, columns=feature_labels)
    df_b = pd.DataFrame(data=other, columns=feature_labels)

    df_a["Legend"] = "TST true positive | TabNet false negative"
    df_b["Legend"] = "Others"

    df = pd.concat([df_a, df_b])
    df = df[feats + ["Legend"]]
    # df.drop(columns=["AbsoluteEosinophilCount", "NucleatedRedCells"], inplace=True)
    df.rename(columns=PRETTY_NAME_MAP, inplace=True)
    plottable = df.melt(
        id_vars="Legend", var_name="Variable", value_name="Value (normalized)"
    )

    ax = sns.boxplot(
        data=plottable,
        y="Variable",
        x="Value (normalized)",
        hue="Legend",
        orient="h",
        showfliers=False,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("results/boxplot.png")

    print("Created graph")


def compare(a, b):
    m_a = np.nanmean(a, axis=0)
    std_a = np.nanstd(a, axis=0)
    count_a = np.count_nonzero(~np.isnan(a), axis=0)

    m_b = np.nanmean(b, axis=0)
    std_b = np.nanstd(b, axis=0)
    count_b = np.count_nonzero(~np.isnan(b), axis=0)

    stat, p = ttest_ind_from_stats(
        m_a,
        std_a,
        count_a,
        m_b,
        std_b,
        count_b,
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
            "count_a": count_a,
            "count_b": count_b,
        }
    )

    # Drop all features with 1 obs or less as p-vals will be invalid
    sig_df = sig_df[(sig_df["count_a"] > 1) & (sig_df["count_b"] > 1)]
    sig_df.drop(columns=["count_a", "count_b"], inplace=True)
    sig_df["ratio"] = sig_df["mean_tst_better"] / sig_df["mean_other"]

    sig_df = sig_df[sig_df["p"] < 0.05]

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
    print(feature_comparison.nsmallest(20, columns="p"))
    feature_comparison["difference"] = (
        feature_comparison["mean_tst_better"] - feature_comparison["mean_other"]
    ).abs()

    plotdiff(
        X_tst_better,
        X_other,
        feature_comparison.nlargest(n=5, columns="difference").feature.to_list(),
    )

    # What features were most important in this subgroup?
    # attribs_tst = build_attributions("TST", 3, 10).abs().sum(dim=1)
    # attribs_tst = (
    #     attribs_tst[y == 1][
    #         torch.logical_and(
    #             preds_tst_pos > opt_threshold, preds_tabnet_pos < opt_threshold
    #         )
    #     ]
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )

    # attribs_tst = attribs_tst / attribs_tst.sum()

    # attribs_tabnet = build_attributions("Tabnet", 3, 256).abs()
    # attribs_tabnet = (
    #     attribs_tabnet[y == 1][
    #         torch.logical_and(
    #             preds_tst_pos > opt_threshold, preds_tabnet_pos < opt_threshold
    #         )
    #     ]
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )

    # attribs_tabnet = attribs_tabnet / attribs_tabnet.sum()

    # # Add one feature onto TST (LOS)
    # importance_comparison = compare(
    #     np.hstack([attribs_tst, np.zeros((attribs_tst.shape[0], 1))]),
    #     attribs_tabnet,
    # )

    # print("=" * 10)

    # print(importance_comparison.nsmallest(20, columns="p_adj"))
