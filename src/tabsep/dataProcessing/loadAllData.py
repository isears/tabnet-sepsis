"""
Load all data with dask like a wild man
"""
import pandas as pd
import dask
import dask.dataframe as dd
from tabsep.dataProcessing.util import all_inclusive_dtypes
import datetime
import random


class megaloader:
    def __init__(self) -> None:
        dask.config.set(scheduler="processes")
        random.seed(42)

        included_stay_ids = pd.read_csv("cache/included_stayids.csv").squeeze("columns")
        included_features = pd.read_csv("cache/included_features.csv").squeeze(
            "columns"
        )
        print(f"Initiated data loading with {len(included_stay_ids)} ICU stays")

        print("Randomizing cut times before processing")
        icustays = pd.read_csv(
            "mimiciv/icu/icustays.csv",
            usecols=["stay_id", "hadm_id", "subject_id", "intime", "outtime"],
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
        icustays["cut_los"] = icustays["cut_time"] - icustays["intime"]
        icustays["cut_los"] = icustays["cut_los"].apply(lambda x: x.total_seconds())

        # Check work
        icustays["sepsis_time"] = icustays["sepsis_time"].fillna(datetime.datetime.max)
        assert (icustays["cut_time"] < icustays["sepsis_time"]).all()
        assert (icustays["cut_time"] > icustays["intime"]).all()

        icustays[["stay_id", "cut_los", "label"]].to_csv(
            "cache/processed_metadata.csv", index=False
        )

        self.icustays = icustays
        self.included_stay_ids = included_stay_ids
        self.included_features = included_features
        print("[+] Cut times randomized, running processor...")

    def load_it_all(self, df_in, agg_fn: str):

        # filter inclusion criteria
        df_in = df_in[df_in["stay_id"].isin(self.included_stay_ids)]
        df_in = df_in[df_in["itemid"].isin(self.included_features)]
        df_in = df_in.dropna(how="any")

        df_in = df_in.merge(
            self.icustays[["stay_id", "cut_time"]], how="left", on="stay_id"
        )

        df_in = df_in[df_in["cut_time"].gt(df_in["charttime"])]

        avg_vals = (
            df_in[["stay_id", "itemid", "valuenum"]]
            .groupby(["stay_id", "itemid"])
            .apply(agg_fn, meta=pd.DataFrame({"valuenum": [0.0]}))
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
            .apply(get_max_event, meta=pd.Series([0.0]))
            .compute()
            .unstack(fill_value=0.0)
        )

        combined_vals = avg_vals.merge(
            latest_vals, left_index=True, right_index=True, suffixes=("_avg", "_latest")
        )

        # Add in labels
        combined_vals = combined_vals.merge(
            self.icustays[["stay_id"]],
            how="left",
            left_index=True,
            right_on="stay_id",
        )

        return combined_vals

    def load_chartevents(self):
        chartevents = dd.read_csv(
            "mimiciv/icu/chartevents.csv",
            dtype=all_inclusive_dtypes,
            usecols=["stay_id", "itemid", "valuenum", "charttime"],
            parse_dates=["charttime"],
        )

        return self.load_it_all(chartevents, "mean")

    def load_outputevents(self):
        outputevents = dd.read_csv(
            "mimiciv/icu/outputevents.csv",
            dtype=all_inclusive_dtypes,
            usecols=["stay_id", "itemid", "value", "charttime"],
            parse_dates=["charttime"],
        )

        outputevents = outputevents.rename(columns={"value": "valuenum"})

        return self.load_it_all(outputevents, "sum")

    def load_inputevents(self):
        inputevents = dd.read_csv(
            "mimiciv/icu/inputevents.csv",
            dtype=all_inclusive_dtypes,
            usecols=["stay_id", "itemid", "starttime", "rate"],
            parse_dates=["starttime"],
        )

        inputevents = inputevents.rename(
            columns={"rate": "valuenum", "starttime": "charttime"}
        )

        return self.load_it_all(inputevents, "mean")

    def load_procedureevents(self):
        procedureevents = dd.read_csv(
            "mimiciv/icu/procedureevents.csv",
            dtype=all_inclusive_dtypes,
            usecols=["stay_id", "itemid", "starttime", "value"],
            parse_dates=["starttime"],
        )

        procedureevents = procedureevents.rename(
            columns={"value": "valuenum", "starttime": "charttime"}
        )

        return self.load_it_all(procedureevents, "sum")

    def load_static(self):
        admissions = pd.read_csv("mimiciv/core/admissions.csv")

        static_df = self.icustays[["stay_id", "hadm_id"]].merge(
            admissions[
                [
                    "hadm_id",
                    "admission_type",
                    "admission_location",
                    "insurance",
                    "language",
                    "marital_status",
                    "ethnicity",
                ]
            ],
            how="left",
            on="hadm_id",
        )

        static_df = static_df.drop(columns=["hadm_id"]).set_index("stay_id")
        static_df = pd.get_dummies(static_df)

        # Just extract age from patients
        patients_df = pd.read_csv("mimiciv/core/patients.csv")
        other_static = self.icustays.merge(
            patients_df[["anchor_age", "anchor_year", "subject_id", "gender"]],
            how="left",
            on="subject_id",
        )

        other_static["age_on_admission"] = other_static.apply(
            lambda row: row["intime"].year - row["anchor_year"] + row["anchor_age"],
            axis=1,
        )

        other_static["gender"] = other_static["gender"].apply(
            lambda x: 1.0 if x == "M" else 0.0
        )

        other_static = other_static.set_index("stay_id")

        static_df = static_df.merge(
            other_static[["gender", "age_on_admission"]],
            how="left",
            left_index=True,
            right_index=True,
        )

        return static_df


def load_from_disk() -> pd.DataFrame:
    df_out = pd.read_csv(
        "cache/processed_metadata.csv", index_col="stay_id", low_memory=False
    )

    for csv in [
        "cache/processed_chartevents.csv",
        "cache/processed_inputevents.csv",
        "cache/processed_outputevents.csv",
        "cache/processed_procedureevents.csv",
        "cache/processed_static.csv",
    ]:
        df = pd.read_csv(csv, index_col="stay_id", low_memory=False)
        df_out = df_out.merge(df, how="left", left_index=True, right_index=True)

    df_out = df_out.fillna(0.0)

    return df_out


if __name__ == "__main__":
    ml = megaloader()

    # processed_inputevents = ml.load_inputevents()
    # processed_inputevents.to_csv("cache/processed_inputevents.csv", index=False)

    # processed_chartevents = ml.load_chartevents()
    # processed_chartevents.to_csv("cache/processed_chartevents.csv", index=False)

    # processed_procedureevents = ml.load_procedureevents()
    # processed_procedureevents.to_csv("cache/processed_procedureevents.csv", index=False)

    static = ml.load_static()
    static.to_csv("cache/processed_static.csv")
