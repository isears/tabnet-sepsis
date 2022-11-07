"""
Test goodness-of-fit to uniform distribution for distribution of attributions across ICU stays
Use 1-d linear interpolation to compare time series of different lengths
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import get_feature_labels
from tabsep.reporting import pretty_feature_names


def interp_attrib(single_stay_atts):
    """
    Linear 1-d interpolation of all ICU stays up (or down) to a stay length of 100
    """
    assert single_stay_atts.ndim == 1

    original_len = len(single_stay_atts)
    assert original_len < 121  # Max stay length

    x = np.arange(0, 100 + (100 / original_len), 100 / (original_len - 1))
    f = interp1d(x, single_stay_atts)

    # Interpolate to range(0,100)
    interpolated = f(np.arange(0.0, 100.0, 1.0))
    assert len(interpolated) == 100

    return interpolated


def pdf_to_dist(pdf_arr):
    """
    Converts the probability distribution of attribution over ICU stay to a raw distribution
    for use in goodness-of-fit testing
    """
    assert pdf_arr.ndim == 1

    dist_arr = list()
    for idx in range(0, len(pdf_arr)):
        dist_arr += [idx] * int(pdf_arr[idx])

    return np.array(dist_arr) / len(pdf_arr)


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
    sns.set_theme()

    for feature_idx in range(0, att.shape[2]):
        this_feature_att = att[:, :, feature_idx]

        att_interp_agg = np.zeros(100)
        for stay_idx in range(0, att.shape[0]):
            stay_len = int(np.sum(pad_masks[stay_idx]))

            if stay_len > 2:  # Need at least three values for interpolation
                att_interp = interp_attrib(this_feature_att[stay_idx, 1:stay_len])

                # att_interp_agg += np.abs(att_interp)
                att_interp_agg += att_interp

        # TODO: get r^2 vals and add to legend
        # https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
        # res = np.polyfit(
        #     np.array(range(0, len(att_interp_agg))), att_interp_agg, deg=25
        # )

        x = np.expand_dims(np.array(range(0, len(att_interp_agg))), axis=-1)
        polynomial_features = PolynomialFeatures(degree=5)
        xp = polynomial_features.fit_transform(x)

        polyfit = sm.OLS(att_interp_agg, xp).fit()
        best_fit_curve = polyfit.predict(xp)

        plottable = pd.DataFrame(
            data={"% Stay": range(0, 100), "Absolute Attribution": att_interp_agg}
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.barplot(
            data=plottable, x="% Stay", y="Absolute Attribution", color="r", ax=ax
        )

        sns.lineplot(
            x=np.array(range(0, 100)), y=best_fit_curve, color="b", ax=ax, lw=5,
        )
        ax.set_title(f"{top[feature_idx]}\nP-value: {polyfit.f_pvalue}")
        ax.xaxis.set_major_locator(ticker.LinearLocator(10))  # TODO: fix this
        plt.tight_layout()
        plt.savefig(f"results/top20Dist/{top[feature_idx]}.png")
        plt.clf()

