import torch
from tabsep import config
import numpy as np
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        feature = sys.argv[1]
    else:
        feature = "Temperature Fahrenheit"

    sns.set_theme()

    attributions = torch.load(f"{config.model_path}/attributions.pt").detach().numpy()
    labels = get_feature_labels()
    feature_idx = labels.index(feature)

    abssum_attrs = np.sum(np.abs(attributions), axis=1)
    mi = mutual_info_regression(abssum_attrs, abssum_attrs[:, feature_idx])

    df = pd.DataFrame(data={"Feature": labels, "Mutual Information": mi})
    mi_std = df["Mutual Information"].std()

    topn = df.nlargest(columns="Mutual Information", n=11)

    print(topn)

    # Top correlator should be the original feature; if so, drop
    if topn.iloc[0]["Feature"] != feature:
        print("[-] Warning: feature did not correlate with self!")
        print(f"Top correlator: {topn.iloc[0]['Feature']}")
    else:
        topn = topn.iloc[1:, :]

    plt.figure()
    ax = sns.barplot(
        x="Mutual Information", y="Feature", data=topn, orient="h", color="b"
    )
    ax.set_title(f"Top features correlating with: {feature}", wrap=True)

    plt.axvline(x=mi_std, color="r", linestyle="dotted")
    plt.axvline(x=mi_std * 2, color="r", linestyle="dotted")
    plt.tight_layout()
    plt.savefig(f"results/correlations.png")
    plt.clf()
