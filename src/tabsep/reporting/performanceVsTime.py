import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.metrics import auc

import scipy.stats
import random


def bootstrapping_auprc(y_test, preds):
    random.seed(42)
    bootstraps = 50
    n = preds.shape[0]
    assert n == y_test.shape[0]
    assert n != 1

    indices = list(range(0, n))

    auprcs = list()

    for bootstrap_idx in range(0, bootstraps):
        this_group_indices = random.choices(indices, k=n)
        group_labels = y_test[this_group_indices]
        group_preds = preds[this_group_indices]

        # precision, recall, thresholds = precision_recall_curve(
        #     group_labels, group_preds
        # )
        # auprc = auc(recall, precision)
        auprcs.append(average_precision_score(group_labels, group_preds))

    return auprcs


if __name__ == "__main__":
    plottable = {
        "Model": list(),
        "Area Under the Precision-Recall Curve (AUPRC)": list(),
        "Prediction Window (Hours)": list(),
    }
    pretty_model_names = {
        "TST": "Time Series Transformer",
        "Tabnet": "TabNet",
        "LR": "Logistic Regression",
    }

    for window_idx in [3, 6, 12, 24]:
        saved_distrib = {}

        for model in pretty_model_names.keys():
            y_test = torch.load(
                f"cache/{model}/sparse_labeled_{window_idx}_y_test.pt",
                weights_only=True,
            )
            preds = torch.load(
                f"cache/{model}/sparse_labeled_{window_idx}_preds.pt", weights_only=True
            )

            bootstrapped_auprcs = bootstrapping_auprc(y_test, preds)

            plottable["Model"] += [pretty_model_names[model]] * len(bootstrapped_auprcs)
            plottable[
                "Area Under the Precision-Recall Curve (AUPRC)"
            ] += bootstrapped_auprcs
            plottable["Prediction Window (Hours)"] += [window_idx] * len(
                bootstrapped_auprcs
            )

            avg_auprc = sum(bootstrapped_auprcs) / len(bootstrapped_auprcs)
            lower_ci, upper_ci = scipy.stats.norm.interval(
                0.99,
                loc=avg_auprc,
                scale=scipy.stats.sem(bootstrapped_auprcs),
            )
            s, p = scipy.stats.normaltest(bootstrapped_auprcs)
            print(
                f"{model[0]} window {window_idx:02d}: AUPRC {avg_auprc:.2f} ({lower_ci:.2f}-{upper_ci:.2f}) (normtest {p:.3f})"
            )

            saved_distrib[model] = bootstrapped_auprcs

        _, p_tst_nn = scipy.stats.ttest_ind(
            saved_distrib["TST"], saved_distrib["Tabnet"]
        )
        _, p_tst_lr = scipy.stats.ttest_ind(saved_distrib["TST"], saved_distrib["LR"])
        _, p_nn_lr = scipy.stats.ttest_ind(saved_distrib["Tabnet"], saved_distrib["LR"])
        print(f"P TST vs Tabnet: {p_tst_nn}")
        print(f"P TST vs LR: {p_tst_lr}")
        print(f"P Tabnet vs LR: {p_nn_lr}")

    plottable = pd.DataFrame(data=plottable)

    sns.set_theme(style="whitegrid", font_scale=1.75, rc={"figure.figsize": (15, 12)})

    ax = sns.barplot(
        data=plottable,
        x="Area Under the Precision-Recall Curve (AUPRC)",
        y="Prediction Window (Hours)",
        hue="Model",
        hue_order=pretty_model_names.values(),
        errorbar=(
            lambda x: scipy.stats.norm.interval(
                0.99,
                loc=x.mean(),
                scale=scipy.stats.sem(x),
            )
        ),
        orient="h",
        capsize=0.1,
        # width=2.0,
        # dodge=False,
    )

    plottable.to_csv("results/performanceVsTime.csv", index=False)

    ax.set_xlim(0.3, 0.5)
    ax.set_xticks([0.3, 0.35, 0.4, 0.45, 0.5])
    sns.move_legend(ax, "upper left", title="Legend", bbox_to_anchor=(1, 1))
    # plt.legend(title="Legend")
    # ax.axhline(y=0.07, color="black", linestyle="--")

    plt.tight_layout()

    plt.savefig("results/performanceVsTime.png")

    print("done")
