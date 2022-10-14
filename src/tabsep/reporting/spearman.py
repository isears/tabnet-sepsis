import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
from tabsep.reporting import pretty_feature_names
from tabsep.reporting.globalImportance import revise_pad


# Sum over single icu stays (output (# icustays, # features))
def sum_attrs(att, pm):
    att = att * np.expand_dims(pm, axis=-1)
    summed_att = np.sum(np.sum(np.abs(att), axis=0), axis=0)

    return rankdata(summed_att, method="min")


if __name__ == "__main__":
    att = torch.load(f"{config.model_path}/attributions.pt").detach().numpy()
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    X_train = torch.load(f"{config.model_path}/X_train.pt")
    # shape: # icu stays, seq_len, # features
    X_combined = torch.cat((X_test, X_train), 0).detach().numpy()
    pad_masks = X_combined[:, :, -1]
    X_combined = X_combined[:, :, :-1]
    feature_labels = get_feature_labels()

    # isolate only top 20 features
    top = pd.read_csv("results/global_importances.csv")["Variable"].to_list()
    inv_map = {v: k for k, v in pretty_feature_names.items()}
    top_raw = [inv_map[n] if n in inv_map else n for n in top]

    kept_feature_indices = [feature_labels.index(n) for n in top_raw]

    att = att[:, :, kept_feature_indices]

    pm, early_pm, late_pm = revise_pad(pad_masks)
    early_importances = sum_attrs(att, early_pm)
    late_importances = sum_attrs(att, late_pm)

    print(spearmanr(early_importances, late_importances))
