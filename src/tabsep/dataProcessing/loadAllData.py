"""
Load all data with dask like a wild man
"""
import pandas as pd
import dask
import dask.dataframe as dd
from tabsep.dataProcessing.util import all_inclusive_dtypes
import datetime
import random


def load_it_all(df_in, fname_out: str):
    dask.config.set(scheduler="processes")
    random.seed(42)

    included_stay_ids = pd.read_csv("cache/included_stayids.csv").squeeze("columns")
    included_features = pd.read_csv("cache/included_features.csv").squeeze("columns")
    print(f"Initiated data loading with {len(included_stay_ids)} ICU stays")

    print("Randomizing cut times before processing")
    icustays = pd.read_csv(
        "mimiciv/icu/icustays.csv",
        usecols=["stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )

    sepsis_df = pd.read_csv(
        "mimiciv/derived/sepsis3.csv",
        parse_dates=[
            "sofa_time",
            "suspected_infection_time",
            "culture_time",
            "antibiotic_time",
        ],
    )

    icustays = icustays[icustays["stay_id"].isin(included_stay_ids)]
    sepsis_df = sepsis_df[sepsis_df["stay_id"].isin(included_stay_ids)]

    sepsis_df["sepsis_time"] = sepsis_df.apply(
        lambda row: max(row["sofa_time"], row["suspected_infection_time"]), axis=1
    )

    sepsis_df["sepsis3"] = 1.0

    icustays = icustays.merge(
        sepsis_df[["stay_id", "sepsis_time", "sepsis3"]], how="left", on="stay_id"
    )

    # Add labels
    icustays = icustays.rename(columns={"sepsis3": "label"})
    icustays["label"] = icustays["label"].fillna(0.0)

    def generate_random_cut_time(row):
        if row["sepsis_time"] is not pd.NaT:
            return row["sepsis_time"] - datetime.timedelta(hours=12)
        else:
            delta = row["outtime"] - row["intime"]
            # Always ensure at least 6 hours of ICU time (6 * 60 * 60 = )
            random_index_second = random.randrange(21600, delta.total_seconds())
            return row["intime"] + datetime.timedelta(seconds=random_index_second)

    icustays["cut_time"] = icustays.apply(generate_random_cut_time, axis=1)

    # Check work
    icustays["sepsis_time"] = icustays["sepsis_time"].fillna(datetime.datetime.max)
    assert (icustays["cut_time"] < icustays["sepsis_time"]).all()
    assert (icustays["cut_time"] > icustays["intime"]).all()

    print("[+] Cut times randomized, running processor...")

    # filter inclusion criteria
    df_in = df_in[df_in["stay_id"].isin(included_stay_ids)]
    df_in = df_in[df_in["itemid"].isin(included_features)]
    df_in = df_in.dropna(how="any")

    df_in = df_in.merge(icustays[["stay_id", "cut_time"]], how="left", on="stay_id")

    df_in = df_in[df_in["cut_time"].gt(df_in["charttime"])]

    avg_vals = (
        df_in[["stay_id", "itemid", "valuenum"]]
        .groupby(["stay_id", "itemid"])
        .apply("mean")
        .compute()  # Dask doesn't implement unstack, so have to compute here
        .unstack(fill_value=0.0)
    )

    # set new columns to be itemids
    avg_vals.columns = avg_vals.columns.map(lambda col: col[1])

    def get_max_event(g):
        if len(g) == 0:
            return pd.NA
        else:
            return g.loc[g["charttime"].idxmax()]["valuenum"]

    latest_vals = (
        df_in[["stay_id", "itemid", "charttime", "valuenum"]]
        .groupby(["stay_id", "itemid"])
        .apply(get_max_event)
        .compute()
        .unstack(fill_value=0.0)
    )

    combined_vals = avg_vals.merge(
        latest_vals, left_index=True, right_index=True, suffixes=("_avg", "_latest")
    )

    # Add in labels
    combined_vals = combined_vals.merge(
        icustays[["stay_id", "label"]], how="left", left_index=True, right_on="stay_id"
    )

    assert not combined_vals["label"].isna().any()
    combined_vals.to_csv(fname_out, index=False)


def load_chartevents():
    chartevents = dd.read_csv(
        "mimiciv/icu/chartevents.csv",
        # "testce.csv",
        assume_missing=True,
        blocksize=1e7,
        dtype=all_inclusive_dtypes,
        usecols=["stay_id", "itemid", "valuenum", "charttime"],
        parse_dates=["charttime"],
    )

    load_it_all(chartevents, "cache/processed_chartevents.csv")


def load_outputevents():
    outputevents = dd.read_csv(
        "mimiciv/icu/outputevents.csv",
        assume_missing=True,
        blocksize=1e7,
        dtype=all_inclusive_dtypes,
        usecols=["stay_id", "itemid", "value", "charttime"],
        parse_dates=["charttime"],
    )

    outputevents = outputevents.rename(columns={"value": "valuenum"})

    load_it_all(outputevents, "cache/processed_outputevents.csv")


def load_inputevents():
    raise NotImplemented


if __name__ == "__main__":
    load_outputevents()
