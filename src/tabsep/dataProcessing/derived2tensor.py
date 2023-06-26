import pickle
import random

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch

from tabsep.dataProcessing import LabeledSparseTensor


class DerivedDataReader:
    def __init__(self, root_data_path, lookahead_hours=24) -> None:
        self.root = root_data_path
        self.icustays = pd.read_parquet(f"{root_data_path}/icustay_detail.parquet")

        def agg_fn(hadm_group):
            return hadm_group.to_dict(orient="records")

        self.hadm_to_stay_mapper = (
            self.icustays[["hadm_id", "stay_id", "icu_intime", "icu_outtime"]]
            .groupby("hadm_id")
            .apply(agg_fn)
        )

        self.icustays = self.icustays.set_index("stay_id")
        sepsis = pd.read_parquet(f"{root_data_path}/sepsis3.parquet").set_index(
            "stay_id"
        )
        sepsis["sepsis_time"] = sepsis.apply(
            lambda x: max(x["sofa_time"], x["suspected_infection_time"]), axis=1
        )

        self.icustays = self.icustays.merge(
            sepsis["sepsis_time"], how="left", left_index=True, right_index=True
        )

        self.icustays["sepsis_tidx"] = self.icustays.apply(
            lambda x: int(
                (x["sepsis_time"] - x["icu_intime"]).total_seconds() / (60 * 60)
            )
            if not pd.isna(x["sepsis_time"])
            else 0,
            axis=1,
        )

        # Drop stays < 24 hrs.
        before_len = len(self.icustays)
        self.icustays = self.icustays[self.icustays["los_icu"] > 1]
        print(
            f"[*] Dropped {before_len - len(self.icustays)} with icu stay length < 24 hrs"
        )

        # Drop sepsis within 48 hrs.
        before_len = len(self.icustays)
        self.icustays = self.icustays[
            (self.icustays["sepsis_tidx"] == 0) | (self.icustays["sepsis_tidx"] > 48)
        ]

        print(
            f"[*] Dropped {before_len - len(self.icustays)} with sepsis within the first 48 hrs"
        )

        # Set a truncation index that's either random between 24 hrs and end of icu stay or 24 hrs before sepsis
        random.seed(42)

        def set_truncation(row_in):
            if row_in["sepsis_tidx"] > 0:
                return row_in["sepsis_tidx"] - lookahead_hours
            else:
                return random.randint(24, int(row_in["los_icu"] * 24))

        self.icustays["tidx_max"] = self.icustays.apply(set_truncation, axis=1)

        # We're not going to use data that's more than 5 days older than the truncation index
        self.icustays["tidx_min"] = self.icustays["tidx_max"].apply(
            lambda t: t - (5 * 24) if t > (5 * 24) else 0
        )

        assert not (self.icustays["tidx_max"] <= 0).any()

        print(f"Final n: {len(self.icustays)}")
        print(
            f"% Sepsis: {100* len(self.icustays[self.icustays['sepsis_tidx'] > 0]) / len(self.icustays)}"
        )

        print("[+] Setup complete")

    """
    UTIL FUNCTIONS
    """

    def _populate_stay_ids(self, row):
        """
        Some tables only have hadm_id and charttime, not stay_id
        up to us to determine if measurement happened during ICU stay
        """
        if not row["hadm_id"] in self.hadm_to_stay_mapper.index:
            return None

        time_col = "charttime" if "charttime" in row.index else "starttime"

        for icustay_metadata in self.hadm_to_stay_mapper[row["hadm_id"]]:
            if (
                row[time_col] > icustay_metadata["icu_intime"]
                and row[time_col] < icustay_metadata["icu_outtime"]
            ):
                return icustay_metadata["stay_id"]
        else:
            return None

    """
    Individual table loaders, some tables or groups of tables have unique data processing requirements
    """

    def _load_common(self, path: str):
        df = pd.read_parquet(path)

        if "stay_id" not in df.columns:
            df = df[~df["hadm_id"].isna()]
            df["stay_id"] = df.apply(self._populate_stay_ids, axis=1)

        df = df[~df["stay_id"].isna()]
        df = df[df["stay_id"].isin(self.icustays.index)]

        df = df.merge(
            self.icustays[["icu_intime", "tidx_max", "tidx_min"]],
            how="left",
            left_on="stay_id",
            right_index=True,
        )

        df["tidx"] = df.apply(
            lambda x: int(
                (x["charttime"] - x["icu_intime"]).total_seconds() / (60 * 60)
            ),
            axis=1,
        )

        df = df[(df["tidx"] < df["tidx_max"]) & (df["tidx"] >= df["tidx_min"])]
        df["tidx"] = df["tidx"] - df["tidx_min"]

        df = df.drop(columns=["icu_intime", "charttime", "tidx_max", "tidx_min"])

        return df

    def load_vitals_table(self, table_name):
        df = self._load_common(f"{self.root}/{table_name}.parquet")

        df["systolic_bp"] = df[["sbp", "sbp_ni"]].mean(axis=1)
        df["diastolic_bp"] = df[["dbp", "dbp_ni"]].mean(axis=1)

        df["temperature"] = df["temperature"].astype("float")

        return df[
            [
                "stay_id",
                "tidx",
                "heart_rate",
                "systolic_bp",
                "diastolic_bp",
                "resp_rate",
                "temperature",
                "spo2",
                "glucose",
            ]
        ]

    def load_bg_table(self, table_name):
        df = self._load_common(f"{self.root}/{table_name}.parquet")
        df["fio2"] = df[["fio2", "fio2_chartevents"]].mean(axis=1)

        return df[
            [
                "stay_id",
                "tidx",
                "so2",
                "po2",
                "pco2",
                "fio2",
                "aado2",
                "aado2_calc",
                "pao2fio2ratio",
                "ph",
                "baseexcess",
                "bicarbonate",
                "totalco2",
                "hematocrit",
                "hemoglobin",
                "carboxyhemoglobin",
                "methemoglobin",
                "chloride",
                "calcium",
                "temperature",
                "potassium",
                "sodium",
                "lactate",
                "glucose",
            ]
        ]

    def load_measurement_table(self, table_name):
        df = self._load_common(f"{self.root}/{table_name}.parquet")

        usable_columns = [
            c
            for c in df.columns
            if c
            not in [
                "subject_id",
                "hadm_id",
                "specimen_id",
                "rdwsd",
                "Microcytes",
            ]
        ]

        return df[usable_columns]

    def load_meds_table(self, table_name):
        # TODO
        raise NotImplementedError()


if __name__ == "__main__":
    pd.set_option("mode.chained_assignment", None)

    reader = DerivedDataReader("./mimiciv_derived")

    tables = {
        "vitalsign": reader.load_vitals_table,
        "bg": reader.load_bg_table,
        "chemistry": reader.load_measurement_table,
        "coagulation": reader.load_measurement_table,
        "differential_detailed": reader.load_measurement_table,
        "complete_blood_count": reader.load_measurement_table,
        "enzyme": reader.load_measurement_table,
        "inflammation": reader.load_measurement_table,
        "icp": reader.load_measurement_table,
    }

    gathered_dfs = list()

    for t, load_fn in tables.items():
        print(f"[*] Loading {t}")
        standardized_df = load_fn(t)

        feature_columns = [
            c for c in standardized_df.columns if c not in ["stay_id", "tidx"]
        ]
        for c in feature_columns:
            single_feature_df = standardized_df[["stay_id", c, "tidx"]]
            single_feature_df["feature_name"] = c
            single_feature_df = single_feature_df.rename(columns={c: "value"})
            single_feature_df = single_feature_df.dropna(subset="value")

            gathered_dfs.append(single_feature_df)

    combined_df = pd.concat(gathered_dfs)

    stay_ids = combined_df["stay_id"].unique().tolist()
    features = combined_df["feature_name"].unique().tolist()

    combined_df["stay_id"] = combined_df["stay_id"].apply(lambda s: stay_ids.index(s))
    combined_df["feature_name"] = combined_df["feature_name"].apply(
        lambda f: features.index(f)
    )
    # Drop data that may have been repeated across multiple tables
    combined_df = combined_df.drop_duplicates(
        subset=["stay_id", "feature_name", "tidx"]
    )

    combined_df = combined_df[combined_df["tidx"] >= 0]

    sparse_tensor = torch.sparse_coo_tensor(
        combined_df[["stay_id", "feature_name", "tidx"]].values.transpose(),
        combined_df["value"].values,
    )

    y = reader.icustays["sepsis_tidx"] != 0
    y = y.reindex(stay_ids)
    y = torch.tensor(y.values)

    lst = LabeledSparseTensor(stay_ids, features, sparse_tensor, y)

    with open("cache/sparse_labeled.pkl", "wb") as f:
        pickle.dump(lst, f)

    print("done")
