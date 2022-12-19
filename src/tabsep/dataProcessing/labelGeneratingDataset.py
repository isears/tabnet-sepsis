import json
import os.path

import pandas as pd
import torch
from torch.nn.functional import pad

from tabsep import config
from tabsep.dataProcessing import label_itemid, min_seq_len
from tabsep.dataProcessing.inclusionCriteria import InclusionCriteria


class LabelGeneratingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stay_ids,
        label_itemid: int,
        dropped_itemids,
        processed_mimic_path: str = "./mimicts",
        pm_type=torch.bool,  # May require pad mask to be different type
    ):

        print(f"[{type(self).__name__}] Initializing dataset...")

        self.feature_ids = (
            pd.read_csv("cache/included_features.csv").squeeze("columns").to_list()
        )

        self.stay_ids = stay_ids
        self.label_itemid = label_itemid
        # Only need to drop itemids that are already in the featureids
        self.dropped_itemids = [i for i in dropped_itemids if i in self.feature_ids]
        self.processed_mimic_path = processed_mimic_path
        self.pm_type = pm_type

        try:
            with open("cache/metadata.json", "r") as f:
                self.max_len = int(json.load(f)["max_len"])
        except FileNotFoundError:
            print(
                f"[{type(self).__name__}] Failed to load metadata. Computing maximum length, this may take some time..."
            )
            self.max_len = 0
            for sid in self.stay_ids:
                ce = pd.read_csv(
                    f"{processed_mimic_path}/{sid}/chartevents_features.csv",
                    nrows=1,
                )
                seq_len = len(ce.columns) - 1

                if seq_len > self.max_len:
                    self.max_len = seq_len

            with open("cache/metadata.json", "w") as f:
                f.write(json.dumps({"max_len": self.max_len}))

        print(f"\tMax length: {self.max_len}")
        print(f"\tExamples: {len(self.stay_ids)}")
        print(f"\tFeatures: {len(self.feature_ids) - len(self.dropped_itemids)}")
        print(f"\tLabel itemid: {self.label_itemid}")
        print(f"\tDropped itemids: {self.dropped_itemids}")

    def maxlen_padmask_collate(self, batch):
        """
        Pad and return third value (the pad mask)
        Returns X, y, padmask
        """
        for idx, (X, y) in enumerate(batch):
            actual_len = X.shape[1]

            assert (
                actual_len <= self.max_len
            ), f"Actual: {actual_len}, Max: {self.max_len}"

            pad_mask = torch.ones(actual_len)
            X_mod = pad(X, (0, self.max_len - actual_len), mode="constant", value=0.0)

            pad_mask = pad(
                pad_mask, (0, self.max_len - actual_len), mode="constant", value=0.0
            )

            batch[idx] = (X_mod.T, y, pad_mask)

        X = torch.stack([X for X, _, _ in batch], dim=0)
        y = torch.stack([Y for _, Y, _ in batch], dim=0)
        pad_mask = torch.stack([pad_mask for _, _, pad_mask in batch], dim=0)

        return dict(X=X.float(), padding_masks=pad_mask.to(self.pm_type)), y.squeeze().float()

    def maxlen_padmask_collate_combined(self, batch):
        """
        For compatibility with scikit learn, add the padmask as the last feature in X
        * If using this method, remember to remove the padmask from X in model
        """
        X, y, pad_mask = self.maxlen_padmask_collate(batch)
        X_and_pad = torch.cat((X, torch.unsqueeze(pad_mask, dim=-1)), dim=-1)
        return X_and_pad, y

    def maxlen_collate(self, batch):
        """
        Pad, but don't include padmask at all (either as separate return value or part of X)
        """
        X, y, _ = self.maxlen_padmask_collate(batch)
        return X, y

    def last_nonzero_collate(self, batch):
        """
        Just return the last nonzero value in X
        """
        x_out = list()
        for idx, (X, y) in enumerate(batch):
            # Shape will be 1 dim (# of features)
            last_nonzero_indices = (X.shape[1] - 1) - torch.argmax(
                torch.flip(X, dims=(1,)).ne(0.0).int(), dim=1
            )
            x_out.append(X[torch.arange(X.shape[0]), last_nonzero_indices])

        y = torch.stack([Y for _, Y in batch], dim=0)
        X = torch.stack(x_out, dim=0)

        return X.float(), y.float()

    def get_num_features(self) -> int:
        return len(self.feature_ids)

    def get_labels(self) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        # Features
        # Ensures every example has a sequence length of at least 1
        combined_features = pd.DataFrame([])

        for feature_file in [
            "chartevents_features.csv",
            "outputevents_features.csv",
            "inputevent_features.csv",
        ]:
            full_path = f"{self.processed_mimic_path}/{stay_id}/{feature_file}"

            if os.path.exists(full_path):
                curr_features = pd.read_csv(
                    full_path,
                    index_col="feature_id",
                )

                combined_features = pd.concat([combined_features, curr_features])

        # Make sure all itemids are represented in order, add 0-tensors where missing
        combined_features = combined_features.reindex(
            self.feature_ids
        )  # Need to add any itemids that are missing
        combined_features = combined_features.fillna(0.0)

        # Randomly select a non-zero value from label itemids
        possible_labels = combined_features.loc[self.label_itemid]
        min_idx = min_seq_len / config.timestep
        possible_indices = [
            int(i)
            for i in possible_labels[possible_labels != 0].index.to_list()
            if int(i) >= min_idx
        ]

        assert (
            len(possible_indices) > 0
        ), f"Stay id {stay_id} did not have a valid label event"

        selected_label = possible_labels[possible_indices].sample(1, random_state=42)
        cut_idx = int(selected_label.index[0])
        Y = torch.tensor(selected_label.to_list())
        Y = Y > 35.0  # TODO: remove if doing regression

        # Cut X one timestep prior to selected label and drop specified features
        kept_cols = [str(i) for i in range(0, cut_idx)]
        combined_features = combined_features[kept_cols]
        # TODO: consider dropping the feature we're trying to predict
        # Would have to adjust # of features in constructor as well
        # combined_features = combined_features.drop(
        #     index=self.dropped_itemids + [self.label_itemid]
        # )

        X = torch.tensor(combined_features.values)

        return X.float(), Y.float()


class CoagulopathyDataset(LabelGeneratingDataset):
    def __init__(
        self,
        stay_ids,
        processed_mimic_path: str = "./mimicts",
        pm_type=torch.bool,
    ):
        super().__init__(
            stay_ids,
            label_itemid,
            [],
            processed_mimic_path,
            pm_type,
        )


def demo(dl):
    print("Printing first few batches:")
    for batchnum, (X, Y) in enumerate(dl):
        print(f"Batch number: {batchnum}")
        print(f"X shape: {X['X'].shape}")
        print(f"Y shape: {Y.shape}")
        print(X)
        print(Y)

        if batchnum == 5:
            break


def get_label_prevalence(dl):
    y_tot = torch.tensor(0.0)
    for batchnum, (X_dict, Y) in enumerate(dl):
        y_tot += torch.sum(Y)

    print(f"Postivie Ys: {y_tot / (batchnum * dl.batch_size)}")


if __name__ == "__main__":
    ds = CoagulopathyDataset(
        pd.read_csv("cache/included_stayids.csv").squeeze("columns").to_list()
    )

    dl = torch.utils.data.DataLoader(
        ds,
        collate_fn=ds.maxlen_padmask_collate,
        num_workers=config.cores_available,
        batch_size=4,
        pin_memory=True,
    )

    print("Iteratively getting label prevalence...")
    get_label_prevalence(dl)

    # print("Demoing first few batches...")
    # demo(dl)

    # print("Testing label getter:")
    # print(ds.get_labels().shape)
