import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc

from tabsep.modeling import CVResults
import scipy.stats
import random


def bootstrapping_auprc(y_test, preds):
    random.seed(42)
    bootstraps = 100
    n = preds.shape[0]
    assert n == y_test.shape[0]
    assert n != 1

    sample_size = n // 10
    indices = list(range(0, n))

    auprcs = list()

    for bootstrap_idx in range(0, bootstraps):
        this_group_indices = random.sample(indices, sample_size)
        group_labels = y_test[this_group_indices]
        group_preds = preds[this_group_indices]

        precision, recall, thresholds = precision_recall_curve(
            group_labels, group_preds
        )
        auprc = auc(recall, precision)
        auprcs.append(auprc)

    return auprcs


if __name__ == "__main__":
    plottable = {
        "Model": list(),
        "AUPRC Scores": list(),
        "Prediction Window (Hrs.)": list(),
    }
    pretty_model_names = {
        "TST": "Time Series Transformer",
        "NN": "Neural Network",
        "LR": "Logistic Regression",
    }

    for window_idx in [3, 6, 12, 24]:
        for model in pretty_model_names.keys():
            y_test = torch.load(f"cache/{model}/sparse_labeled_{window_idx}_y_test.pt")
            preds = torch.load(f"cache/{model}/sparse_labeled_{window_idx}_preds.pt")

            bootstrapped_auprcs = bootstrapping_auprc(y_test, preds)

            plottable["Model"] += [pretty_model_names[model]] * len(bootstrapped_auprcs)
            plottable["AUPRC Scores"] += bootstrapped_auprcs
            plottable["Prediction Window (Hrs.)"] += [window_idx] * len(
                bootstrapped_auprcs
            )

            avg_auprc = sum(bootstrapped_auprcs) / len(bootstrapped_auprcs)
            lower_ci, upper_ci = scipy.stats.norm.interval(
                0.95,
                loc=avg_auprc,
                scale=scipy.stats.sem(bootstrapped_auprcs),
            )
            s, p = scipy.stats.normaltest(bootstrapped_auprcs)
            print(
                f"{model[0]} window {window_idx:02d}: AUPRC {avg_auprc:.5f} +/- {avg_auprc - lower_ci:.5f} (normtest {p:.3f})"
            )

    plottable = pd.DataFrame(data=plottable)

    sns.set(rc={"figure.figsize": (11.7, 8.27)})

    sns.barplot(
        data=plottable,
        x="Prediction Window (Hrs.)",
        y="AUPRC Scores",
        hue="Model",
        hue_order=pretty_model_names.values(),
        errorbar=(
            lambda x: scipy.stats.norm.interval(
                0.95,
                loc=x.mean(),
                scale=scipy.stats.sem(x),
            )
        ),
        capsize=0.1,
    )

    plt.savefig("results/performanceVsTime.png")

    print("done")
