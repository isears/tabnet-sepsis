from distutils.fancy_getopt import fancy_getopt
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from sklearn.metrics import auc as auc_fn

from tabsep.modeling import CVResults

if __name__ == "__main__":
    fancy_names = {
        "LogisticRegression": "Logistic Regression (Snapshot)",
        "MLPClassifier": "Feed-forward Neural Network (Snapshot)",
        "NeuralNetBinaryClassifier": "Time Series Transformer (Time Series)",
    }

    colors = ["r", "b", "g", "y"]

    plt.rcParams["figure.figsize"] = (15, 10)

    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    for idx, (name, fancy_name) in enumerate(fancy_names.items()):
        print(f"Generating ROC curves for model {fancy_name}")

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        cv_res = CVResults.load(f"results/{name}.cvresult")

        for res in cv_res.results:
            # plt.plot(res.fpr, res.tpr, colors[idx], lw=1, alpha=0.3)
            interp_tpr = np.interp(mean_fpr, res.fpr, res.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(res.auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc_fn(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(
            mean_fpr,
            mean_tpr,
            color=colors[idx],
            label=r"%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)"
            % (fancy_name, mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        # std_auc = np.std(aucs)
        # std_tpr = np.std(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # plt.fill_between(
        #     mean_fpr,
        #     tprs_lower,
        #     tprs_upper,
        #     color=colors[idx],
        #     alpha=0.1,
        #     label=r"$\pm$ 1 standard deviation",
        # )

    plt.legend(loc="lower right")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.savefig(f"results/roc_curve_all.png")
