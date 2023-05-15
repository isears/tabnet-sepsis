"""
MIMIC derived tables (list of timestamped events w/event type and value) to 
multiple individual icu stay timelines (time on axis 1, feature type on axis 2)
"""

import pandas as pd


class Derived2Ts:
    def __init__(self) -> None:
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

    def _convert_to_hourly_resolution(self, df_in, ts_col):
        df_in[ts_col] = df_in[ts_col].apply(
            lambda x: x.replace(minute=0, second=0, microsecond=0)
        )

        return df_in

    def _generate_daterange(self, stay_id):
        intime = self.icustays.loc[stay_id]["icu_intime"]
        outtime = self.icustays.loc[stay_id]["icu_outtime"]

        daterange = pd.date_range(intime, outtime, freq="H")

        return daterange

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

    def _load_clean(self, path: str):
        """
        Populate stay_id, if needed; drop rows that have no associated stay_id / hadm_id
        """
        df = pd.read_parquet(path)

        if "stay_id" not in df.columns:
            df = df[~df["hadm_id"].isna()]
            df["stay_id"] = df.apply(self._populate_stay_ids, axis=1)

        df = df[~df["stay_id"].isna()]
        return df

    """
    AGGREGATION FUNCTIONS
    """

    def agg_vitals_group(self, stay_group):
        stay_id = stay_group.name
        daterange = self._generate_daterange(stay_id)

        # Decrease charttime resolution to hourly
        stay_group = self._convert_to_hourly_resolution(stay_group, "charttime")

        stay_group["systolic_bp"] = stay_group[["sbp", "sbp_ni"]].mean(axis=1)
        stay_group["diastolic_bp"] = stay_group[["dbp", "dbp_ni"]].mean(axis=1)

        stay_group["temperature"] = stay_group["temperature"].astype("float")

        stay_group = (
            stay_group[
                [
                    "charttime",
                    "heart_rate",
                    "systolic_bp",
                    "diastolic_bp",
                    "resp_rate",
                    "temperature",
                    "spo2",
                    "glucose",
                ]
            ]
            .groupby("charttime")
            .agg("mean")
        )

        stay_group = stay_group.reindex(daterange)

        # TODO: planning to leave this to the dataset so that normalization can be applied before imputation
        # Forward fill where possible, otherwise fill w/0.0
        # stay_group = stay_group.fillna(method="ffill")
        # stay_group = stay_group.fillna(0.0)

        stay_group.to_parquet(f"{self.save_path}/{stay_id}.vitalsign.parquet")

    def agg_bg_group(self, stay_group):
        stay_id = int(stay_group.name)
        daterange = self._generate_daterange(stay_id)
        stay_group = self._convert_to_hourly_resolution(stay_group, "charttime")

        stay_group["fio2"] = stay_group[["fio2", "fio2_chartevents"]].mean(axis=1)

        stay_group = (
            stay_group[
                [
                    "charttime",
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
            .groupby("charttime")
            .agg("mean")
        )

        stay_group = stay_group.reindex(daterange)
        stay_group.to_parquet(f"{self.save_path}/{stay_id}.bg.parquet")

    def agg_default_measurement_group(self, stay_group, name):
        """
        Most groups will require similar processing
        """
        stay_id = int(stay_group.name)
        daterange = self._generate_daterange(stay_id)
        stay_group = self._convert_to_hourly_resolution(stay_group, "charttime")

        aggable_columns = [
            c
            for c in stay_group.columns
            if c not in ["subject_id", "stay_id", "hadm_id", "specimen_id"]
        ]

        stay_group = stay_group[aggable_columns].groupby("charttime").agg("mean")

        stay_group = stay_group.reindex(daterange)
        stay_group.to_parquet(f"{self.save_path}/{stay_id}.{name}.parquet")

    def run_all(self):
        # print("[*] Running vitals...")
        # vitals = pd.read_parquet("mimiciv_derived/vitalsign.parquet")
        # vitals.groupby("stay_id").apply(self.agg_vitals_group)

        # print("[*] Running bg...")
        # bg = self._load_clean("mimiciv_derived/bg.parquet")
        # bg.groupby("stay_id").apply(self.agg_bg_group)

        standard_tables = [
            "chemistry",
            "coagulation",
            "differential_detailed",
            "complete_blood_count",
            "enzyme",
            "inflammation",
            "icp",
        ]

        for table_name in standard_tables:
            print(f"[*] Aggregating {table_name}...")
            df = self._load_clean(f"mimiciv_derived/{table_name}.parquet")
            df.groupby("stay_id").apply(
                lambda g: self.agg_default_measurement_group(g, table_name)
            )


if __name__ == "__main__":
    runner = Derived2Ts()
    runner.run_all()

    print("done")
