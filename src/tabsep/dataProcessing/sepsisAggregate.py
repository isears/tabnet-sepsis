from mimic2ts import BaseAggregator, all_inclusive_dtypes
import dask.dataframe as dd
from typing import List
import pandas as pd


class SepsisAggregator(BaseAggregator):
    def __init__(
        self,
        mimic_path: str,
        dst_path: str,
        stay_ids: List[int],
        timestep_seconds: int = 3600,
        blocksize=10e6,
    ):

        self.data = dd.read_csv(
            f"{mimic_path}/derived/sepsis3.csv",
            assume_missing=True,
            blocksize=blocksize,
            dtype=all_inclusive_dtypes,
        )

        self.sepsis_feature_id = 77700

        super().__init__(
            mimic_path,
            dst_path,
            stay_ids,
            self.sepsis_feature_id,
            timestep_seconds,
            "sepsis3",
        )

    def _stime_parser(self, row):
        # Return latest of sofa and infection
        infection_time = pd.to_datetime(row["suspected_infection_time"])
        sofa_time = pd.to_datetime(row["sofa_time"])
        if infection_time > sofa_time:
            return row["suspected_infection_time"]
        elif sofa_time > infection_time:
            return row["sofa_time"]

    def _etime_parser(self, row):
        # No information about when sepsis is "over"
        # Will probably clip timesries just before sepsis onset in dataset
        return self._stime_parser(row)

    def _value_parser(self, row):
        # If entry exists, there's sepsis
        return 1.0

    def _feature_id_parser(self, row):
        # Made up feature id
        # Convention to start derived feature ids with 777
        return self.sepsis_feature_id


if __name__ == "__main__":
    # TODO: replace this with inclusion criteria
    stay_ids = pd.read_csv("./mimiciv/icu/icustays.csv")["stay_id"].to_list()
    se = SepsisAggregator(
        "./mimiciv",
        "./cache/mimicts",
        stay_ids,
    )

    se.do_agg()
