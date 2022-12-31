"""
Dataset for unsupervised pretraining
"""

import torch
from mvtst.datasets.dataset import collate_unsuperv, noise_mask

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset


class FileBasedImputationDataset(FileBasedDataset):
    def __init__(
        self,
        examples_path: str,
        shuffle: bool = True,
        processed_mimic_path: str = "./mimicts",
        mean_mask_length: int = 3,
        masking_ratio: float = 0.15,
        mode: str = "separate",
        distribution: str = "geometric",
        exclude_feats=None,
        pm_type=torch.bool,
    ):
        super().__init__(examples_path, shuffle, processed_mimic_path, pm_type)

        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats

    def __getitem__(self, index: int):
        stay_id = self.examples["stay_id"].iloc[index]

        # Need to transpose b/c downstream functions expect
        # (seq_length, feat_dim) shape for some reason
        X = super().__getitem_X__(index).T
        mask = noise_mask(
            X,
            self.masking_ratio,
            self.mean_mask_length,
            self.mode,
            self.distribution,
            self.exclude_feats,
        )  # (seq_length, feat_dim) boolean array

        return X, torch.from_numpy(mask), stay_id

    def update(self):
        """
        Present in the gzerveas version of Imputation dataset
        Want to make sure it's not used by mistake
        """
        raise NotImplemented


def demo(dl):
    print("Printing first few batches:")
    for batchnum, (X, targets, target_masks, padding_masks, IDs) in enumerate(dl):
        print(f"Batch number: {batchnum}")
        print(IDs)
        print(f"X shape: {X.shape}")
        print(X)
        print(f"Targets shape: {targets.shape}")
        print(targets)
        print(f"Target Masks shape: {target_masks.shape}")
        print(target_masks)
        print(f"Padding Masks shape: {padding_masks.shape}")
        print(padding_masks)

        print("=" * 10)

        if batchnum == 5:
            break


if __name__ == "__main__":
    ds = FileBasedImputationDataset("cache/pretrain_examples.csv")

    dl = torch.utils.data.DataLoader(
        ds,
        collate_fn=collate_unsuperv,
        # num_workers=config.cores_available,
        num_workers=1,
        batch_size=4,
        pin_memory=True,
    )

    demo(dl)
