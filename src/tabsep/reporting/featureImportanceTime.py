from tabsep.reporting.featureImportance import build_attributions, featuregroup_indices
from tabsep.dataProcessing import LabeledSparseTensor
import torch
import pandas as pd
import seaborn as sns
from tabsep.modeling.skorchTST import AutoPadmaskingTST
import matplotlib.pyplot as plt


if __name__ == "__main__":
    tst_atts = build_attributions("TST", 3, 10)

    # Get absolute importances
    tst_absolute = torch.sum(torch.abs(tst_atts), dim=0)

    tst_normal = tst_absolute / torch.sum(tst_absolute)

    feature_labels = LabeledSparseTensor.load_from_pickle(
        "cache/sparse_labeled_3.pkl"
    ).features
    original_data = LabeledSparseTensor.load_from_pickle(
        "cache/sparse_labeled_3.pkl"
    ).get_dense_raw()

    pad_masks = AutoPadmaskingTST.autopadmask(original_data).bool()

    # Need to shift all icu stays so that padding is at beginning rather than end
    attribs_shifted = torch.full_like(tst_atts, float("nan"))

    for idx in range(0, attribs_shifted.shape[0]):
        this_sample_mask = pad_masks[idx]
        this_sample_atts = tst_atts[idx]

        nan_indices = torch.where(this_sample_mask == False)[0]
        if len(nan_indices) == 0:
            seq_len = 120
        else:
            seq_len = nan_indices[0].item()

        attribs_shifted[idx] = torch.nn.functional.pad(
            this_sample_atts[0:seq_len],
            (0, 0, pad_masks.shape[1] - seq_len, 0),
            value=float("nan"),
        )

    # Absolute average
    attribs_shifted = torch.nanmean(torch.abs(attribs_shifted), dim=0)
    plottable = pd.DataFrame(
        data=attribs_shifted.detach().cpu(), columns=feature_labels
    )

    for name, idx in featuregroup_indices.items():
        plottable[name] = plottable.iloc[:, idx].sum(axis=1) / len(idx)

    plottable.drop(columns=feature_labels, inplace=True)
    plottable["Hours to Sepsis Prediction"] = (
        len(plottable) - plottable.index.to_series()
    )

    plottable = plottable.melt(
        "Hours to Sepsis Prediction",
        var_name="Variable Type",
        value_name="Average Absolute Attribution",
    )

    sns.set_theme(style="whitegrid", font_scale=1.75, rc={"figure.figsize": (10, 10)})
    ax = sns.lineplot(
        data=plottable,
        x="Hours to Sepsis Prediction",
        y="Average Absolute Attribution",
        hue="Variable Type",
    )
    ax.invert_xaxis()
    # plt.xticks(rotation=45)
    plt.legend(title="Legend")
    plt.tight_layout()
    plt.savefig("results/importanceTime.png")
    plt.clf()

    # plot_groupped_importances(f"results/importancesTime.png", tst_normal, tabnet_normal)

    print("[+] Done")
