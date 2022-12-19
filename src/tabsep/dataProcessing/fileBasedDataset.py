import json
import os.path

import pandas as pd
import torch
from torch.nn.functional import pad

from tabsep import config


class FileBasedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cutsample_indices,
        processed_mimic_path: str = "./mimicts",
        pm_type=torch.bool,  # May require pad mask to be different type
    ):

        print(f"[{type(self).__name__}] Initializing dataset...")

        self.feature_ids = (
            pd.read_csv("cache/included_features.csv").squeeze("columns").to_list()
        )

        self.cut_sample = pd.read_csv("cache/sample_cuts.csv")
        # NOTE: maxlen calculated before filtering down to indices so that
        # datasets all have uniform seq lengths
        self.max_len = self.cut_sample["cutidx"].max()
        self.cut_sample = self.cut_sample.loc[cutsample_indices].reset_index(drop=True)

        # Must shuffle otherwise all pos labels will be @ end
        self.cut_sample = self.cut_sample.sample(frac=1, random_state=42)

        print(f"\tExamples: {len(self.cut_sample)}")
        print(f"\tFeatures: {len(self.feature_ids)}")

        self.processed_mimic_path = processed_mimic_path
        self.pm_type = pm_type

        print(f"\tMax length: {self.max_len}")

        print(f"[{type(self).__name__}] Dataset initialization complete")

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

        return dict(X=X.float(), padding_masks=pad_mask.to(self.pm_type)), y.float()

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
        Y = torch.tensor(self.cut_sample["label"].to_list())
        # Y = torch.unsqueeze(Y, 1)
        return Y

    def __len__(self):
        return len(self.cut_sample)

    def __getitem__(self, index: int):
        stay_id = self.cut_sample["stay_id"].iloc[index]
        Y = torch.tensor(self.cut_sample["label"].iloc[index])
        cutidx = self.cut_sample["cutidx"].iloc[index]

        # Features
        # Ensures every example has a sequence length of at least 1
        combined_features = pd.DataFrame(columns=["feature_id", "0"])

        for feature_file in [
            "chartevents_features.csv",
            "outputevents_features.csv",
            "inputevent_features.csv",
        ]:
            full_path = f"{self.processed_mimic_path}/{stay_id}/{feature_file}"

            if os.path.exists(full_path):
                curr_features = pd.read_csv(
                    full_path,
                    usecols=list(range(0, cutidx)),
                    index_col="feature_id",
                )

                combined_features = pd.concat([combined_features, curr_features])

        # Make sure all itemids are represented in order, add 0-tensors where missing
        combined_features = combined_features.reindex(
            self.feature_ids
        )  # Need to add any itemids that are missing
        combined_features = combined_features.fillna(0.0)
        X = torch.tensor(combined_features.values)

        return X.float(), Y.float()


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
    for batchnum, (X, Y, pad_mask) in enumerate(dl):
        y_tot += torch.sum(Y)

    print(f"Postivie Ys: {y_tot / (batchnum * dl.batch_size)}")


if __name__ == "__main__":
    sample_cuts = pd.read_csv("cache/sample_cuts.csv")
    ds = FileBasedDataset(sample_cuts["stay_id"])

    dl = torch.utils.data.DataLoader(
        ds,
        collate_fn=ds.maxlen_padmask_collate,
        num_workers=config.cores_available,
        batch_size=4,
        pin_memory=True,
    )

    print("Testing label getter:")
    print(ds.get_labels().shape)

    # print("Iteratively getting label prevalence...")
    # get_label_prevalence(dl)

    print("Demoing first few batches...")
    demo(dl)
