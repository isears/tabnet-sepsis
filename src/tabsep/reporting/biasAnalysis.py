"""
Subgroup performance analysis (race, sex)
"""

import pandas as pd
import torch
import argparse
from tabsep.reporting.performanceVsTime import bootstrapping_auprc
from tabsep.dataProcessing import LabeledSparseTensor
from sklearn.model_selection import train_test_split
import scipy
import random

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Subgroup performance analysis")

    # args = parser.parse_args()

    y_test = (
        torch.load(f"cache/Tabnet/sparse_labeled_3_y_test.pt").detach().cpu().numpy()
    )
    preds = torch.load(f"cache/Tabnet/sparse_labeled_3_preds.pt").detach().cpu().numpy()

    lst = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled_3.pkl")

    x = lst.get_snapshot()
    # count missingness
    x[x != -1] = 0.0
    x[x == -1] = 1
    missingness = x.sum(dim=1)

    icustays = pd.read_parquet("mimiciv_derived/icustay_detail.parquet")
    admissions = pd.read_csv("mimiciv/core/admissions.csv")
    icustays = icustays[icustays["stay_id"].isin(lst.stay_ids)]

    icustays = pd.merge(
        icustays, admissions[["hadm_id", "ethnicity"]], how="left", on="hadm_id"
    )

    assert len(icustays) == icustays.stay_id.nunique()
    assert len(icustays) == len(lst.stay_ids)

    icustays = icustays.set_index("stay_id").reindex(lst.stay_ids)

    _, test_icustays = train_test_split(icustays, test_size=0.1, random_state=42)
    _, missingness = train_test_split(missingness, test_size=0.1, random_state=42)

    assert len(test_icustays) == len(y_test)

    test_icustays["y_test"] = y_test
    test_icustays["preds"] = preds
    test_icustays["missingness"] = missingness.detach().cpu().numpy()

    test_icustays["ethnicity"] = test_icustays["ethnicity"].apply(
        lambda e: "WHITE" if e == "WHITE" else "NON-WHITE"
    )

    baseline_auprc = bootstrapping_auprc(
        test_icustays["y_test"].to_numpy(), test_icustays["preds"].to_numpy()
    )

    avg_auprc = sum(baseline_auprc) / len(baseline_auprc)
    lower_ci, upper_ci = scipy.stats.norm.interval(
        0.99,
        loc=avg_auprc,
        scale=scipy.stats.sem(baseline_auprc),
    )

    print(f"BASELINE: {avg_auprc:.3f} [{lower_ci:.5f}, {upper_ci:.5f}]")

    for groupname, data in test_icustays.groupby("ethnicity"):
        if data["y_test"].sum() == 0:
            print(f"[-] {groupname} had no sepsis examples")
            continue

        auprcs = bootstrapping_auprc(
            data["y_test"].to_numpy(), data["preds"].to_numpy()
        )
        avg_auprc = sum(auprcs) / len(auprcs)
        lower_ci, upper_ci = scipy.stats.norm.interval(
            0.99,
            loc=avg_auprc,
            scale=scipy.stats.sem(auprcs),
        )

        print(f"{groupname}: {avg_auprc:.3f} [{lower_ci:.5f}, {upper_ci:.5f}]")
        print(f"Missingness: {data['missingness'].median()}")
        print(f"Sepsis: {data['y_test'].sum() / len(data):.5f}")

    for groupname, data in test_icustays.groupby("gender"):
        if data["y_test"].sum() == 0:
            print(f"[-] {groupname} had no sepsis examples")
            continue

        auprcs = bootstrapping_auprc(
            data["y_test"].to_numpy(), data["preds"].to_numpy()
        )
        avg_auprc = sum(auprcs) / len(auprcs)
        lower_ci, upper_ci = scipy.stats.norm.interval(
            0.99,
            loc=avg_auprc,
            scale=scipy.stats.sem(auprcs),
        )

        print(f"{groupname}: {avg_auprc:.3f} [{lower_ci:.5f}, {upper_ci:.5f}]")
        print(f"Missingness: {data['missingness'].median()}")
        print(f"Sepsis: {data['y_test'].sum() / len(data):.5f}")
