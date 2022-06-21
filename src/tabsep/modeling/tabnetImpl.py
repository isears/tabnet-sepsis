from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import torch
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset

if __name__ == "__main__":
    all_stay_ids = (
        pd.read_csv("cache/included_stayids.csv").squeeze("columns").sample(1000)
    )
    train_sids = all_stay_ids.iloc[0:900].to_list()
    validation_sids = all_stay_ids.iloc[900:1000].to_list()

    train_ds = FileBasedDataset("cache/mimicts", stay_ids=train_sids)
    valid_ds = FileBasedDataset("cahce/mimicts", stay_ids=validation_sids)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=len(train_ds),
        collate_fn=train_ds.padding_collate,
        num_workers=16,
        pin_memory=True,
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=len(valid_ds),
        collate_fn=valid_ds.padding_collate,
        num_workers=16,
        pin_memory=True,
    )

    X_train, Y_train = next(iter(train_dl))
    X_valid, Y_valid = next(iter(valid_dl))

    clf = TabNetClassifier()
    clf.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)])

    print(clf)
