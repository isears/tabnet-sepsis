import os.path
import pickle

import pandas as pd
import torch
from torch.nn.functional import pad
from tqdm import tqdm

from tabsep import config


def get_feature_labels():
    """
    Returns feature labels in-order of their appearance in X
    """
    feature_ids = (
        pd.read_csv("cache/included_features.csv").squeeze("columns").to_list()
    )
    d_items = pd.read_csv("mimiciv/icu/d_items.csv", index_col="itemid")
    d_items = d_items.reindex(feature_ids)

    assert len(d_items) == len(feature_ids)

    return d_items["label"].to_list()


class FileBasedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: str | pd.DataFrame,
        shuffle: bool = True,
        processed_mimic_path: str = "./mimicts",
        pm_type=torch.bool,  # May require pad mask to be different type
        standard_scale=False,
    ):

        print(f"[{type(self).__name__}] Initializing dataset...")

        self.feature_ids = (
            pd.read_csv("cache/included_features.csv").squeeze("columns").to_list()
        )

        if type(examples) == str:
            self.examples = pd.read_csv(examples)
        elif type(examples) == pd.DataFrame:
            self.examples = examples
        else:
            raise ValueError(
                f"Cannot except type {type(examples)} for argument 'examples'"
            )

        if shuffle:
            # May need to shuffle otherwise all pos labels will be @ end
            self.examples = self.examples.sample(frac=1, random_state=42)

        print(f"\tExamples: {len(self.examples)}")
        print(f"\tFeatures: {len(self.feature_ids)}")

        self.processed_mimic_path = processed_mimic_path
        self.pm_type = pm_type
        self.max_len = self.examples["cutidx"].max() + 1

        print(f"\tMax length: {self.max_len}")

        # Find mean and std if using standard scalar
        # NOTE: ignoring 0s as they are usually indicative of missing values
        # TODO: training scaler needs to be shared with validation dataset
        self.standard_scale = standard_scale
        if standard_scale:
            print("Standard scalar turned ON, loading LR standard scaler")
            with open("cache/models/singleLr/whole_model.pkl", "rb") as f:
                self.scaler = pickle.load(f).named_steps["standardscaler"]

        print(f"[{type(self).__name__}] Dataset initialization complete")

    def dataloader_factory(self, batch_size: int):
        # TODO: need to be able to specify collate_fn
        dl = torch.utils.data.DataLoader(
            self,
            collate_fn=self.maxlen_padmask_collate_skorch,
            num_workers=config.cores_available,
            batch_size=batch_size,
            pin_memory=True,
        )

        return dl

    def maxlen_padmask_collate(self, batch):
        """
        Pad and return third value (the pad mask)
        Returns X, y, padmask, stay_ids
        """
        for idx, (X, y, stay_id) in enumerate(batch):
            actual_len = X.shape[1]

            assert (
                actual_len <= self.max_len
            ), f"Actual: {actual_len}, Max: {self.max_len}"

            pad_mask = torch.ones(actual_len)
            X_mod = pad(X, (0, self.max_len - actual_len), mode="constant", value=0.0)

            pad_mask = pad(
                pad_mask, (0, self.max_len - actual_len), mode="constant", value=0.0
            )

            batch[idx] = (X_mod.T, y, pad_mask, stay_id)

        X = torch.stack([X for X, _, _, _ in batch], dim=0)
        y = torch.stack([Y for _, Y, _, _ in batch], dim=0)
        pad_mask = torch.stack([pad_mask for _, _, pad_mask, _ in batch], dim=0)
        IDs = torch.tensor([stay_id for _, _, _, stay_id in batch])

        return X, y, pad_mask.to(self.pm_type), IDs

    def maxlen_padmask_collate_skorch(self, batch):
        """
        Skorch expects kwargs output
        """
        X, y, pad_mask, _ = self.maxlen_padmask_collate(batch)
        return dict(X=X, padding_masks=pad_mask), y

    def timewarp_collate(self, batch):
        raise NotImplemented

    def maxlen_padmask_collate_combined(self, batch):
        """
        For compatibility with scikit learn, add the padmask as the last feature in X
        * If using this method, remember to remove the padmask from X in model
        """
        raise NotImplementedError(
            "Need to account for the fact that __getitem__() returns ID now"
        )
        X, y, pad_mask = self.maxlen_padmask_collate(batch)
        X_and_pad = torch.cat((X, torch.unsqueeze(pad_mask, dim=-1)), dim=-1)
        return X_and_pad, y

    def maxlen_collate(self, batch):
        """
        Pad, but don't include padmask at all (either as separate return value or part of X)
        """
        raise NotImplementedError(
            "Need to account for the fact that __getitem__() returns ID now"
        )
        X, y, _ = self.maxlen_padmask_collate(batch)
        return X, y

    def last_nonzero_collate(self, batch):
        """
        Just return the last nonzero value in X
        """

        if self.standard_scale:
            raise RuntimeError(
                "Collate fn not valid for datasets that implement standard scaling"
            )

        x_out = list()
        for idx, (X, y, ID) in enumerate(batch):
            # Shape will be 1 dim (# of features)
            last_nonzero_indices = (X.shape[1] - 1) - torch.argmax(
                torch.flip(X, dims=(1,)).ne(0.0).int(), dim=1
            )
            x_out.append(X[torch.arange(X.shape[0]), last_nonzero_indices])

        y = torch.stack([Y for _, Y, _ in batch], dim=0)
        X = torch.stack(x_out, dim=0)

        return X.float(), y.float()

    def all_24_collate(self, batch):
        """
        - Truncate timeseries to just most recent 24 timesteps
        - Carry forward all nonzero values older than 24 timesteps prior to cut
        NOTE: this only really makes sense if the dataset is hour-by-hour; this will
        probably break if there are timeseries that are less than 24 steps in length
        """
        # TODO
        x_out = list()
        for idx, (X, y) in enumerate(batch):
            most_recent_X = X[:, -24:, :]  # Values that will be passed thru
            summarizable_X = X[:, :-24, :]  # Values that will be carried fwd
            last_nonzero_indices = (summarizable_X.shape[1] - 1) - torch.argmax(
                torch.flip(summarizable_X, dims=(1,)).ne(0.0).int(), dim=1
            )
            summarized_old_vals = summarizable_X[
                torch.arange(summarizable_X.shape[0]), last_nonzero_indices
            ]

            final_X = torch.cat((summarized_old_vals, most_recent_X), dim=1)
            x_out.append(final_X)

        raise NotImplementedError

    def get_num_features(self) -> int:
        return len(self.feature_ids)

    def get_labels(self) -> torch.Tensor:
        Y = torch.tensor(self.examples["label"].to_list())
        # Y = torch.unsqueeze(Y, 1)
        return Y

    def __getitem_X__(self, index: int):
        """
        Outsource so that it can be called separately in unsupervised datasets
        """
        stay_id = self.examples["stay_id"].iloc[index]
        cutidx = self.examples["cutidx"].iloc[index]

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
                    usecols=list(range(0, cutidx + 1)),
                    index_col="feature_id",
                )

                combined_features = pd.concat([combined_features, curr_features])

        # Make sure all itemids are represented in order, add 0-tensors where missing
        combined_features = combined_features.reindex(
            self.feature_ids
        )  # Need to add any itemids that are missing
        combined_features = combined_features.fillna(0.0)
        X = torch.tensor(combined_features.values).float()

        if self.standard_scale:
            X = (X - self.feature_means[:, None]) / self.feature_stds[:, None]
            X = torch.nan_to_num(X)

        return X

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        stay_id = self.examples["stay_id"].iloc[index]
        Y = torch.tensor(self.examples["label"].iloc[index]).float()
        # Y = torch.unsqueeze(Y, 0)

        X = self.__getitem_X__(index)

        assert not torch.isnan(X).any()
        assert not torch.isnan(Y).any()

        return X, Y, stay_id


def demo(dl):
    print("Printing first few batches:")
    for batchnum, (X, Y) in enumerate(dl):
        print(f"Batch number: {batchnum}")
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
    ds = FileBasedDataset(examples="cache/test_examples.csv", standard_scale=True)

    dl = torch.utils.data.DataLoader(
        ds,
        collate_fn=ds.maxlen_padmask_collate_skorch,
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
