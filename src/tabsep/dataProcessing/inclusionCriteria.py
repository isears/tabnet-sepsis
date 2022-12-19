"""
Apply inclusion criteria to generate a list of included stay ids
"""
import datetime

import dask.dataframe as dd
import pandas as pd

from tabsep.dataProcessing import label_itemid, min_seq_len
from tabsep.dataProcessing.util import all_inclusive_dtypes


class InclusionCriteria:
    def __init__(self):
        self.all_stays = pd.read_csv(
            "mimiciv/icu/icustays.csv",
            usecols=["stay_id", "intime", "outtime"],
            dtype={"stay_id": "int", "intime": "str", "outtime": "str"},
            parse_dates=["intime", "outtime"],
        )

    def _exclude_nolabels(self):
        """
        Guarantee that there are labels > 24 hours into ICU stay
        """
        chartevents = dd.read_csv(
            "mimiciv/icu/chartevents.csv",
            usecols=["stay_id", "subject_id", "itemid", "charttime"],
            dtype=all_inclusive_dtypes,
            blocksize=100e6,
        )

        chartevents = chartevents[chartevents["itemid"] == label_itemid].compute(
            scheduler="processes"
        )

        chartevents = chartevents[
            chartevents["stay_id"].isin(self.all_stays["stay_id"])
        ]

        chartevents = chartevents.merge(
            self.all_stays[["stay_id", "intime", "outtime"]], how="left", on="stay_id"
        )

        # Drop anything that's too close to intime or after outtime
        for time_column in ["charttime", "intime", "outtime"]:
            chartevents[time_column] = pd.to_datetime(chartevents[time_column])

        chartevents["difftime"] = chartevents["charttime"] - chartevents["intime"]
        chartevents = chartevents[chartevents["difftime"] > min_seq_len]
        chartevents = chartevents[chartevents["charttime"] < chartevents["outtime"]]

        chartevents_stay_ids = chartevents["stay_id"].unique()
        self.all_stays = self.all_stays[
            self.all_stays["stay_id"].isin(chartevents_stay_ids)
        ]

    def _exclude_short_stays(self, time_hours=24):
        self.all_stays = self.all_stays[
            self.all_stays.apply(
                lambda row: (row["outtime"] - row["intime"])
                > datetime.timedelta(hours=time_hours),
                axis=1,
            )
        ]

    def _exclude_long_stays(self, time_hours=(24 * 30)):
        self.all_stays = self.all_stays[
            self.all_stays.apply(
                lambda row: (row["outtime"] - row["intime"])
                < datetime.timedelta(hours=time_hours),
                axis=1,
            )
        ]

    def get_included(self):
        order = [
            self._exclude_nolabels,
            self._exclude_short_stays,
            self._exclude_long_stays,
        ]

        for func in order:
            count_before = len(self.all_stays)
            func()
            count_after = len(self.all_stays)
            print(f"{func.__name__} excluded {count_before - count_after} stay ids")

        print(f"Saving remaining {len(self.all_stays)} stay ids to disk")
        self.all_stays["stay_id"].to_csv("cache/included_stayids.csv", index=False)
        return self.all_stays["stay_id"].to_list()


if __name__ == "__main__":
    ic = InclusionCriteria()
    stay_ids = ic.get_included()
