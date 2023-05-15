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

        # TODO: standard scaling?

    def first_pass_processing(self):
        """
        Need to do one full pass of dataset in order to calculate:
        - mean by feature
        - std by feature
        - max feature values
        - min features values
        - max seq len

        Save to cache somehow?
        """
        raise NotImplementedError()

    def _populate_features(self) -> list:
        feature_names = list()

        for table_name in self.used_tables:
            example_table = pd.read_parquet(
                glob.glob(f"processed/*.{table_name}.parquet")[0]
            )

            feature_names += example_table.columns.to_list()

        return feature_names

    def get_labels(self) -> torch.Tensor:
        raise NotImplementedError()

    def __getitem_X__(self, stay_id: int):
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

        # TODO: current nan filling strategy: ffill and 0 out anything at the beginning
        # combined = combined.fillna(method="ffill")
        combined = combined.fillna(-1.0)

        return combined

    def __getitem_Y__(self, stay_id: int):
        raise NotImplementedError()

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X = self.__getitem_X__(stay_id)
        # Y = self.__getitem_Y__(stay_id)

        # assert not torch.isnan(X).any()
        # assert not torch.isnan(Y).any()

        # return X, Y, stay_id

        return X


if __name__ == "__main__":
    examples = pd.read_csv("cache/test_examples.csv")
    ds = DerivedDataset(stay_ids=examples["stay_id"].to_list())

    print(ds[0])
