import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc

from tabsep.modeling import CVResults

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
        for model in ["LR", "NN", "TST"]:
            y_test = torch.load(f"cache/{model}/sparse_labeled_{window_idx}_y_test.pt")
            preds = torch.load(f"cache/{model}/sparse_labeled_{window_idx}_preds.pt")

            precision, recall, thresholds = precision_recall_curve(y_test, preds)
            auprc = auc(recall, precision)

            plottable["Model"].append(pretty_model_names[model])
            plottable["AUPRC Scores"].append(auprc)
            plottable["Prediction Window (Hrs.)"].append(window_idx)

            print(f"{model} window {window_idx}: AUPRC {auprc}")

    plottable = pd.DataFrame(data=plottable)

    sns.set(rc={"figure.figsize": (11.7, 8.27)})

    sns.barplot(
        data=plottable,
        x="Prediction Window (Hrs.)",
        y="AUPRC Scores",
        hue="Model",
        hue_order=pretty_model_names.values(),
    )

    plt.savefig("results/performanceVsTime.png")

    print("done")
