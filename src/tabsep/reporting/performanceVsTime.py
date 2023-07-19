import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from tabsep.modeling import CVResults

if __name__ == "__main__":
    plottable = {
        "Model": list(),
        "AUPRC Scores": list(),
        "AUROC Scores": list(),
        "Prediction Window (Hrs.)": list(),
    }
    pretty_model_names = {
        "TST": "Time Series Transformer",
        "LR": "Logistic Regression",
        "Tabnet": "TabNet",
    }

    for window_idx in [3, 6, 12, 24]:
        for model in ["LR", "Tabnet", "TST"]:
            y_test = torch.load(f"cache/{model}/sparse_labeled_{window_idx}_y_test.pt")
            preds = torch.load(f"cache/{model}/sparse_labeled_{window_idx}_preds.pt")
            auprc = average_precision_score(y_test, preds)
            auroc = roc_auc_score(y_test, preds)

            plottable["Model"].append(pretty_model_names[model])
            plottable["AUPRC Scores"].append(auprc)
            plottable["AUROC Scores"].append(auroc)
            plottable["Prediction Window (Hrs.)"].append(window_idx)

    plottable = pd.DataFrame(data=plottable)
    sns.barplot(
        data=plottable,
        x="Prediction Window (Hrs.)",
        y="AUPRC Scores",
        hue="Model",
        hue_order=pretty_model_names.values(),
    )
    plt.savefig("results/performanceVsTime.png")

    print("done")
