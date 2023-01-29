"""
Dataset for unsupervised pretraining
"""

import random

import torch
from mvtst.datasets.dataset import collate_unsuperv, transduct_mask

from tabsep import config
from tabsep.dataProcessing.fileBasedImputationDataset import FileBasedImputationDataset


class FileBasedTransductionDataset(FileBasedImputationDataset):
    # TODO: should probably create an abstract unsupervised dataset that
    # both imputation and and transduction inherit from but will do this
    # for now
    def __init__(
        self,
        examples_path: str,
        mask_feats: list,
        start_hint: int = 0.0,
        end_hint: int = 0.0,
        shuffle: bool = True,
        processed_mimic_path: str = "./mimicts",
        pm_type=torch.bool,
    ):
        super().__init__(examples_path, shuffle, processed_mimic_path, pm_type)

        random.seed(42)

        self.mask_feats = mask_feats
        self.start_hint = start_hint
        self.end_hint = end_hint

        if mask_feats is None:
            # TODO: think more about this but for now just use most important
            mask_feats = []
            raise NotImplemented(
                "Need to think more about how mask feats should be generated"
            )

        print(f"\tMask features: {self.mask_feats}")

    def collate_unsuperv_skorch(self, batch):
        """
        Return values in a format that can automatically be passed to forward() / loss()
        i.e. via skorch

        Forward requires kwargs X and padding_masks
        Loss requires targets and target_masks
        """
        X, targets, target_masks, padding_masks, IDs = collate_unsuperv(
            batch, max_len=self.max_len
        )
        return (
            dict(X=X, padding_masks=padding_masks),
            dict(y_true=targets, mask=target_masks),
        )

    def __getitem__(self, index: int):
        stay_id = self.examples["stay_id"].iloc[index]

        # For now, just selecting one feature to mask at a time
        this_mask_feats = random.choices(self.mask_feats, k=1)

        # Need to transpose b/c downstream functions expect
        # (seq_length, feat_dim) shape for some reason
        X = super().__getitem_X__(index).T
        mask = transduct_mask(X, this_mask_feats, self.start_hint, self.end_hint)

        return X, torch.from_numpy(mask), stay_id


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
    ds = FileBasedTransductionDataset("cache/pretrain_examples.csv")

    dl = torch.utils.data.DataLoader(
        ds,
        collate_fn=ds.collate_unsuperv_skorch,
        # num_workers=config.cores_available,
        num_workers=1,
        batch_size=4,
        pin_memory=True,
    )

    demo(dl)
