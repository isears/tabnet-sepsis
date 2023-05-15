"""
Apply inclusion criteria to generate a list of included stay ids
"""
import datetime
import glob

import pandas as pd

from tabsep.dataProcessing.util import all_inclusive_dtypes


class InclusionCriteria:
    def __init__(self):
        self.all_stays = pd.read_parquet("mimiciv_derived/icustay_detail.parquet")

    def _exclude_nodata(self):
        """
        Exclude patients w/no associated vitals measurements
        """
        sids = list(
            set(
                [
                    int(s.split("/")[1].split(".")[0])
                    for s in glob.glob("processed/*.vitalsign.parquet")
                ]
            )
        )

        self.all_stays = self.all_stays[self.all_stays["stay_id"].isin(sids)]

    def _exclude_short_stays(self, time_hours=24):
        self.all_stays = self.all_stays[
            self.all_stays.apply(
                lambda row: (row["icu_outtime"] - row["icu_intime"])
                > datetime.timedelta(hours=time_hours),
                axis=1,
            )
        ]

    def _exclude_long_stays(self, time_hours=(24 * 14)):
        self.all_stays = self.all_stays[
            self.all_stays.apply(
                lambda row: (row["icu_outtime"] - row["icu_intime"])
                < datetime.timedelta(hours=time_hours),
                axis=1,
            )
        ]

    def _exclude_early_sepsis(self, time_hours=24):
        """
        Exclude patients that arrive to the ICU qualifying for sepsis3
        """
        sepsis_df = pd.read_parquet("mimiciv_derived/sepsis3.parquet")
        sepsis_df["sepsis_time"] = sepsis_df.apply(
            lambda row: max(row["sofa_time"], row["suspected_infection_time"]), axis=1
        )

        sepsis_df = sepsis_df.merge(
            self.all_stays[["stay_id", "icu_intime"]], how="left", on="stay_id"
        )

        early_sepsis_stays = sepsis_df[
            sepsis_df.apply(
                lambda row: row["sepsis_time"] - row["icu_intime"]
                < datetime.timedelta(hours=time_hours),
                axis=1,
            )
        ]["stay_id"]

        self.all_stays = self.all_stays[
            ~self.all_stays["stay_id"].isin(early_sepsis_stays)
        ]

    def _include_sicu(self):
        self.all_stays = self.all_stays[
            self.all_stays["first_careunit"].isin(
                ["Trauma SICU (TSICU)", "Surgical Intensive Care Unit (SICU)"]
            )
        ]

    def get_included(self):
        order = [
            self._exclude_nodata,
            # self._include_sicu,
            self._exclude_short_stays,
            self._exclude_long_stays,
            self._exclude_early_sepsis,
        ]

        for func in order:
            count_before = len(self.all_stays)
            func()
            count_after = len(self.all_stays)
            print(f"{func.__name__} excluded {count_before - count_after} stay ids")

        print(f"Saving remaining {len(self.all_stays)} stay ids to disk")
        self.all_stays["stay_id"].to_csv("cache/included_stayids.csv", index=False)
        return self.all_stays["stay_id"]


if __name__ == "__main__":
    ic = InclusionCriteria()
    ic.get_included().to_csv("cache/included_stay_ids.csv", index=False)
