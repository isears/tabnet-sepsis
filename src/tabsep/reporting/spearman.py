from scipy.stats import spearmanr
from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
from tabsep.reporting.globalImportance import revise_pad
import pandas as pd

import torch
import numpy as np
from scipy.stats import rankdata


# Sum over single icu stays (output (# icustays, # features))
def sum_attrs(att, pm):
    att = att * np.expand_dims(pm, axis=-1)
    summed_att = np.sum(np.sum(np.abs(att), axis=0), axis=0)

    return rankdata(summed_att, method="min")


if __name__ == "__main__":
    att = torch.load(f"{config.model_path}/attributions.pt").detach().numpy()
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    X_train = torch.load(f"{config.model_path}/X_train.pt")
    X_combined = torch.cat((X_test, X_train), 0)
    pad_masks = X_combined[:, :, -1].detach().numpy()
    feature_labels = get_feature_labels()

    pm, early_pm, late_pm = revise_pad(pad_masks)

    early_importances = sum_attrs(att, early_pm)
    late_importances = sum_attrs(att, late_pm)

    print(spearmanr(early_importances, late_importances))

    # s_correlations = list()
    # s_pvals = list()
    # for idx, fname in enumerate(feature_labels):
    #     coeff, pvalue = spearmanr(early_importances[:, idx], late_importances[:, idx])
    #     s_correlations.append(coeff)
    #     s_pvals.append(pvalue)

    # spearman_df = pd.DataFrame(
    #     data={
    #         "Feature": feature_labels,
    #         "Spearman Correlation": s_correlations,
    #         "P Value": s_pvals,
    #     }
    # )

    # spearman_df = spearman_df.dropna()
    # spearman_df = spearman_df.sort_values(by="Spearman Correlation")

    # print("Top 20 negative coefficients:")
    # print(spearman_df.head(20))
    # print("Top 20 positive coefficients:")
    # print(spearman_df.tail(20))

    # spearman_df.head(20).to_csv("results/spearman_top_neg.csv", index=False)
    # spearman_df.tail(20).to_csv("results/spearman_top_pos.csv", index=False)

