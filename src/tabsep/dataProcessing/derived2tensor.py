import pickle

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch

from tabsep.dataProcessing import LabeledSparseTensor


class NamedSparseTensor:
    def __init__(self) -> None:
        self.gathered_dfs = list()
        self.sparse_tensor = None
        self.stay_ids = list()
        self.features = list()

        # Columns containing either metadata or no useful information
        self.unwanted_columns = [
            "subject_id",
            "hadm_id",
            "stay_id",
            "rdwsd",
            "Microcytes",
            "tidx",
        ]

        self.isinitialized

    def add_data(self, df_in):
        feature_columns = [c for c in df_in.columns if c not in self.unwanted_columns]
        for c in feature_columns:
            single_feature_df = df_in[["stay_id", c, "tidx"]]
            single_feature_df["feature_name"] = c
            single_feature_df = single_feature_df.rename(columns={c: "value"})
            single_feature_df = single_feature_df.dropna(subset="value")

            self.gathered_dfs.append(single_feature_df)

    def build_tensor(self):
        combined_df = pd.concat(self.gathered_dfs)
        self.stay_ids = combined_df["stay_id"].unique().tolist()
        self.features = combined_df["feature_name"].unique().tolist()

        combined_df["stay_id"] = combined_df["stay_id"].apply(
            lambda s: self.stay_ids.index(s)
        )
        combined_df["feature_name"] = combined_df["feature_name"].apply(
            lambda f: self.features.index(f)
        )
        # Drop data that may have been repeated across multiple tables
        combined_df = combined_df.drop_duplicates(
            subset=["stay_id", "feature_name", "tidx"]
        )

        self.sparse_tensor = torch.sparse_coo_tensor(
            combined_df[["stay_id", "feature_name", "tidx"]].values.transpose(),
            combined_df["value"].values,
        )


class DerivedDataReader:
    def __init__(self, root_data_path) -> None:
        self.root = root_data_path
        self.icustays = pd.read_parquet("mimiciv_derived/icustay_detail.parquet")

        def agg_fn(hadm_group):
            return hadm_group.to_dict(orient="records")

        self.hadm_to_stay_mapper = (
            self.icustays[["hadm_id", "stay_id", "icu_intime", "icu_outtime"]]
            .groupby("hadm_id")
            .apply(agg_fn)
        )

        self.icustays = self.icustays.set_index("stay_id")

        # Decrease time resolution to hourly
        for time_col in ["icu_intime", "icu_outtime"]:
            self.icustays[time_col] = self.icustays[time_col].apply(
                lambda x: x.replace(minute=0, second=0, microsecond=0)
            )

        self.save_path = "./processed"

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
        df = df.merge(
            self.icustays["icu_intime"], how="left", left_on="stay_id", right_index=True
        )

        df["tidx"] = df.apply(
            lambda x: int(
                (x["charttime"] - x["icu_intime"]).total_seconds() / (60 * 60)
            ),
            axis=1,
        )

        df = df.drop(columns=["icu_intime", "charttime"])

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

    def load_sepsis_table(self, table_name):
        df = pd.read_parquet(f"{self.root}/{table_name}.parquet")

        df["sepsis_time"] = df.apply(
            lambda x: max(x["sofa_time"], x["suspected_infection_time"]), axis=1
        )

        df = df.merge(
            self.icustays["icu_intime"], how="left", left_on="stay_id", right_index=True
        )

        df["tidx"] = df.apply(
            lambda x: int(
                (x["sepsis_time"] - x["icu_intime"]).total_seconds() / (60 * 60)
            ),
            axis=1,
        )

        df["sepsis3"] = df["sepsis3"].astype(float)

        return df[["stay_id", "tidx", "sepsis3"]]

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
        "sepsis3": reader.load_sepsis_table,
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

    lst = LabeledSparseTensor(stay_ids, features, sparse_tensor)

    with open("cache/sparse_labeled.pkl", "wb") as f:
        pickle.dump(lst, f)

    print("done")
