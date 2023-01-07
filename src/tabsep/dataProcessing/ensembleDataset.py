"""
Labels are LR distance-from-true, not actual labels
"""

import torch

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset, demo


class EnsembleDataset(FileBasedDataset):
    def __init__(
        self,
        examples_path: str,
        shuffle: bool = True,
        processed_mimic_path: str = "./mimicts",
        pm_type=torch.bool,
    ):
        super().__init__(examples_path, shuffle, processed_mimic_path, pm_type)

        if "lr_pred" not in self.examples.columns:
            raise ValueError(
                "Examples dataframe needs to be populated with LR predictions first"
            )

        self.examples["ensemble_label"] = (
            self.examples["label"] - self.examples["lr_pred"]
        )

    def maxlen_padmask_collate_skorch(self, batch):
        """
        Need new collate fn b/c __getitem__ now returns 4
        """
        newbatch = [
            (X, lr_correction, stay_id) for X, lr_correction, Y, stay_id in batch
        ]
        X, y, pad_mask, _ = super().maxlen_padmask_collate(newbatch)
        return dict(X=X, padding_masks=pad_mask), y

    def __getitem__(self, index: int):
        stay_id = self.examples["stay_id"].iloc[index]
        lr_correction = torch.tensor(
            self.examples["ensemble_label"].iloc[index]
        ).float()
        Y = torch.tensor(self.examples["label"].iloc[index]).float()

        X = self.__getitem_X__(index)

        return X, lr_correction, Y, stay_id


if __name__ == "__main__":
    ds = EnsembleDataset(examples_path="cache/test_examples.csv")

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
