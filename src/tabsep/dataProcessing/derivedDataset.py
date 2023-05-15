import datetime
import glob
import os
import random

import pandas as pd
import torch
from torch.nn.functional import pad
from tqdm import tqdm

from tabsep import config

random.seed(42)


class DerivedDataset(torch.utils.data.Dataset):
    used_tables = [
        "vitalsign",
        "chemistry",
        "coagulation",
        "differential_detailed",
        "bg",
        "enzyme",
        "inflammation",
        "dobutamine",
        "epinephrine",
        "invasive_line",
        "milrinone",
        "norepinephrine",
        "phenylephrine",
        "vasopressin",
        "ventilation",
    ]

    def __init__(
        self,
        stay_ids: list[int],
        shuffle: bool = True,
        pm_type=torch.bool,  # May require pad mask to be different type
    ):
        print(f"[{type(self).__name__}] Initializing dataset...")
        self.stay_ids = stay_ids

        if shuffle:
            random.shuffle(self.stay_ids)

        print(f"\tExamples: {len(self.stay_ids)}")

        self.features = self._populate_features()
        print(f"\tFeatures: {len(self.features)}")

        self.pm_type = pm_type
        # TODO:
        # self.max_len = self.examples["cutidx"].max() + 1
        # print(f"\tMax length: {self.max_len}")

        self.stats = pd.read_parquet("processed/stats.parquet")

    def _populate_features(self) -> list:
        feature_names = list()

        for table_name in self.used_tables:
            example_table = pd.read_parquet(
                glob.glob(f"processed/*.{table_name}.parquet")[0]
            )

            feature_names += example_table.columns.to_list()

        return feature_names

    def __getitem_X__(self, stay_id: int) -> pd.DataFrame:
        loaded_dfs = list()

        for table_name in self.used_tables:
            if os.path.exists(f"processed/{stay_id}.{table_name}.parquet"):
                loaded_dfs.append(
                    pd.read_parquet(f"processed/{stay_id}.{table_name}.parquet")
                )

        combined = pd.concat(loaded_dfs, axis="columns")
        # Different data sources may measure same values (for example, blood gas and chemistries)
        # When that happens, just take the mean
        combined = combined.groupby(by=combined.columns, axis=1).mean()

        # Min / max normalization
        for col in combined.columns:
            if col in self.stats.columns:
                combined[col] = (combined[col] - self.stats[col].loc["min"]) / (
                    self.stats[col].loc["max"] - self.stats[col].loc["min"]
                )

        # Fill nas w/-1
        combined = combined.fillna(-1.0)

        return combined

    def __getitem_Y__(self, stay_id: int) -> float:
        if os.path.exists(f"processed/{stay_id}.sepsis3.parquet"):
            return 1.0
        else:
            return 0.0

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X = self.__getitem_X__(stay_id)
        Y = self.__getitem_Y__(stay_id)

        if Y == 1.0:  # Do just-before-sepsis cut
            sepsis_df = pd.read_parquet(f"processed/{stay_id}.sepsis3.parquet")
            t_sepsis = sepsis_df[sepsis_df["sepsis3"] == 1].index[0]
            t_cut = t_sepsis - datetime.timedelta(hours=12)

            if t_cut > X.index[0]:
                daterange = pd.date_range(X.index[0], t_cut, freq="H")
                X = X.loc[daterange]
            else:
                print(
                    f"[-] Warning: stay id {stay_id} doesn't have sufficient data to do pre-sepsis cut"
                )

        else:  # Do random cut
            # TODO: this should be coupled with inclusion criteria
            if len(X.index) <= 24:
                print(
                    f"[-] Warning: stay id {stay_id} doesn't have sufficient data to do random cut"
                )
            else:
                t_cut = random.choice(X.index[24:])
                daterange = pd.date_range(X.index[0], t_cut, freq="H")
                X = X.loc[daterange]

        return X, Y, stay_id


if __name__ == "__main__":
    sids = list(set([s.split("/")[1].split(".")[0] for s in glob.glob("processed/*")]))

    ds = DerivedDataset(stay_ids=sids)

    for X, Y, stay_id in ds:
        print(X)

        break
