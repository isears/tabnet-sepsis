"""
Experiment with tsai version of TST
"""
from tsai.all import (
    TST,
    BCEWithLogitsLossFlat,
    Learner,
    RocAucMulti,
    TSDataLoader,
    TSDataLoaders,
    computer_setup,
)

from tabsep import config
from tabsep.dataProcessing.tsaiDataset import TsaiDataset

if __name__ == "__main__":
    computer_setup()

    train_ds = TsaiDataset("cache/train_examples.csv")
    valid_ds = TsaiDataset("cache/test_examples.csv")
    valid_ds.max_len = train_ds.max_len
    dl = TSDataLoaders.from_dsets(
        train_ds, valid_ds, bs=32, num_workers=config.cores_available
    )

    model = TST(dl.get_num_features(), dl.c, dl.max_len)
    learn = Learner(
        dl, model, loss_func=BCEWithLogitsLossFlat(), metrics=[RocAucMulti()]
    )
    suggested_lr = learn.lr_find().valley

    out = learn.fit_one_cycle(5, lr_max=suggested_lr)

    print(out)
