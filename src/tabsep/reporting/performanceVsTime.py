import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tabsep.modeling import CVResults


def load_cvresult(path: str) -> CVResults:
    with open(path, "rb") as f:
        res = pickle.load(f)

    return res


if __name__ == "__main__":
    plottable = {
        "Model": list(),
        "AUPRC Scores": list(),
        "Prediction Window (Hrs.)": list(),
    }
    pretty_model_names = {
        "TST": "Time Series Transformer",
        "LR": "Logistic Regression",
        "Tabnet": "TabNet",
    }

    for window_idx in [3, 6, 9, 12, 15, 18, 21, 24]:
        for model in ["LR", "Tabnet", "TST"]:
            r = load_cvresult(f"cache/{model}/sparse_labeled_{window_idx}_cvresult.pkl")
            precisions = r.get_precisions()
            plottable["Model"] += [pretty_model_names[model]] * len(precisions)
            plottable["AUPRC Scores"] += precisions
            plottable["Prediction Window (Hrs.)"] += [window_idx] * len(precisions)

    plottable = pd.DataFrame(data=plottable)
    sns.barplot(
        data=plottable,
        x="Prediction Window (Hrs.)",
        y="AUPRC Scores",
        hue="Model",
        hue_order=pretty_model_names.values(),
        # capsize=0.1,
        errorbar=None,
    )
    plt.savefig("results/performanceVsTime.png")

    print("done")
