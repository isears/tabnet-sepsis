from tabsep.reporting.featureImportance import build_attributions, featuregroup_indices
from tabsep.modeling.skorchTST import AutoPadmaskingTST
from tabsep.dataProcessing import LabeledSparseTensor
import seaborn as sns
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

if __name__ == "__main__":
    if not os.path.exists("cache/attribs_cached.pt"):
        tst_attribs = build_attributions("TST", 3, 10)
        torch.save(tst_attribs, "cache/attribs_cached.pt")
    else:
        tst_attribs = torch.load("cache/attribs_cached.pt")

    tst_attribs = tst_attribs.to("cpu")

    X = torch.load(f"cache/TST/sparse_labeled_3_X_test.pt").to("cpu")
    y = torch.load(f"cache/TST/sparse_labeled_3_y_test.pt").to("cpu")
    preds = torch.load(f"cache/TST/sparse_labeled_3_preds.pt").to("cpu")

    pad_masks = AutoPadmaskingTST.autopadmask(X).bool()

    sample_idx = np.argmax(
        torch.logical_and(
            torch.logical_and(
                y == 1,
                torch.sum(pad_masks, dim=1) > 5,
            ),
            torch.logical_and(torch.sum(pad_masks, dim=1) < 100, preds > 0.5),
        )
    ).numpy()

    assert sample_idx != 0, "[-] Sampling criteria too specific"

    print(f"Analyzing local importance of idx {sample_idx}")
    sample_case_attrs_values = tst_attribs[sample_idx].detach().numpy()
    # Simplify for poster presentation
    sample_case_attrs_values = np.abs(sample_case_attrs_values)
    sample_case_attrs = pd.DataFrame(sample_case_attrs_values)
    sample_case_values = pd.DataFrame(X[sample_idx, :, :].numpy())

    for name, indices in featuregroup_indices.items():
        sample_case_attrs.rename(
            columns={i: name for i in indices},
            inplace=True,
        )

    sample_case_attrs = sample_case_attrs.groupby(
        sample_case_attrs.columns, axis=1
    ).agg("mean")

    # Truncate by padding mask
    sample_case_attrs = sample_case_attrs[pad_masks[sample_idx].tolist()]
    sample_case_values = sample_case_values[pad_masks[sample_idx].tolist()]

    max_absolute_attribution = sample_case_attrs.abs().apply(lambda col: col.sum())
    top_n_features = max_absolute_attribution.nlargest(n=10).index

    sample_case_attrs.index.name = "Time in ICU (hours)"
    sample_case_attrs.index = sample_case_attrs.index.astype(int)
    # sample_case_attrs = sample_case_attrs.rename(columns=pretty_features)

    sns.set(font_scale=2)
    fig, ax = plt.subplots(figsize=(30, 12))
    sns.heatmap(
        sample_case_attrs.transpose(),
        linewidths=0.1,
        linecolor="black",
        # annot=sample_case_values.transpose(),
        # fmt=".2f",
        ax=ax,
        cmap="Reds",
        # cbar=False,
        # cbar_kws={"label": "Attribution (darker = more important)"},
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Attribution Value", labelpad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel("Variable")
    # ax.set_title(
    #     f"Validation set idx {sample_idx} prediction {preds[sample_idx]:.2f} actual {y[sample_idx]:.2f}"
    # )
    plt.savefig("results/local_importance_example.png", bbox_inches="tight")

    plt.clf()

    # # Now save a "blank" colorless heatmap for poster
    # fig, ax = plt.subplots(figsize=(15, 15))
    # sns.heatmap(
    #     sample_case_attrs.transpose(),
    #     linewidths=0.1,
    #     linecolor="black",
    #     # annot=sample_case_values.transpose(),
    #     fmt=".2f",
    #     ax=ax,
    #     cmap=ListedColormap(["white"]),
    #     cbar=False,
    # )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
    # # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # # ax.set_title(
    # #     f"Validation set idx {sample_idx} prediction {preds[sample_idx]:.2f} actual {y_test[sample_idx]:.2f}"
    # # )
    # plt.savefig("cache/TST/local_importance_blank.png", bbox_inches="tight")

    # plt.clf()
