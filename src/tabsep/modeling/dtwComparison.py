"""
Test similarities of time series within and between sepsis / no sepsis groups
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind
from tslearn.metrics import dtw_path

from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset


def iterative_dtw(dl: torch.utils.data.DataLoader, sample_size: int = 100):
    all_scores = list()

    for batch_X, _ in dl:
        X1 = batch_X["X"][0, :, :]
        X2 = batch_X["X"][1, :, :]

        assert X1.ndim == 2
        assert X2.ndim == 2

        path, distance = dtw_path(X1, X2)
        all_scores.append(distance)

        if len(all_scores) == sample_size:
            break

    print(f"Mean: {np.mean(all_scores)}")
    print(f"Median: {np.median(all_scores)}")
    print(f"Std Dev: {np.std(all_scores)}")

    return np.array(all_scores)


if __name__ == "__main__":
    dtw_sample_size = 500

    examples = pd.read_csv("cache/train_examples.csv")
    sepsis_examples = examples[examples["label"] == 1]
    no_sepsis_examples = examples[examples["label"] == 0]

    sepsis_ds = FileBasedDataset(examples=sepsis_examples, shuffle=True)
    no_sepsis_ds = FileBasedDataset(examples=no_sepsis_examples, shuffle=True)

    sepsis_dl = sepsis_ds.dataloader_factory(batch_size=2)
    no_sepsis_dl = no_sepsis_ds.dataloader_factory(batch_size=2)

    print("\nSepsis DTW statistics:")
    all_scores_sepsis = iterative_dtw(sepsis_dl, sample_size=dtw_sample_size)

    print("\nNo sepsis DTW statistics:")
    all_scores_nosepsis = iterative_dtw(no_sepsis_dl, sample_size=dtw_sample_size)

    # Need to regenerate DLs probably
    sepsis_dl = sepsis_ds.dataloader_factory(batch_size=1)
    no_sepsis_dl = no_sepsis_ds.dataloader_factory(batch_size=1)

    between_group_scores = list()

    for (sepsis_batch_X, _), (nosepsis_batch_X, _) in zip(sepsis_dl, no_sepsis_dl):
        sepsis_X = sepsis_batch_X["X"][0, :, :]
        nosepsis_X = nosepsis_batch_X["X"][0, :, :]

        assert sepsis_X.ndim == 2
        assert nosepsis_X.ndim == 2

        path, distance = dtw_path(sepsis_X, nosepsis_X)
        between_group_scores.append(distance)

        if len(between_group_scores) == dtw_sample_size:
            break

    between_group_scores = np.array(between_group_scores)

    print("\nBetween Group DTW Scores:")
    print(f"Mean: {np.mean(between_group_scores)}")
    print(f"Median: {np.median(between_group_scores)}")
    print(f"Std Dev: {np.std(between_group_scores)}")
    print("")

    print("T-test: sepsis dtws vs no sepsis dtws:")
    print(ttest_ind(all_scores_sepsis, all_scores_nosepsis))
    print("T-test: between-groups dtws vs no sepsis dtws:")
    print(ttest_ind(between_group_scores, all_scores_nosepsis))
    print("T-test: between-groups dtws vs sepsis dtws:")
    print(ttest_ind(between_group_scores, all_scores_sepsis))
