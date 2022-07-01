from xgboost import XGBClassifier
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import multiprocessing
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    cut_sample = cut_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    train, test = train_test_split(cut_sample, test_size=0.1, random_state=42)
    train_ds = FileBasedDataset("cache/mimicts", cut_sample=train)
    test_ds = FileBasedDataset("cache/mimicts", cut_sample=test)

    test_ds.max_len = train_ds.max_len

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=len(train_ds),
        collate_fn=train_ds.maxlen_padmask_collate,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=len(test_ds),
        collate_fn=test_ds.maxlen_padmask_collate,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )

    # TODO: fix test_dl
    X_train, y_train, pm_train = next(iter(train_dl))

    print(X_train.shape)
    print(y_train.shape)
    print(pm_train.shape)

    model = XGBClassifier()
    model.fit(
        X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2])),
        y_train,
    )

    print(model)

    X_test, y_test, pm_test = next(iter(test_dl))

    y_pred = model.predict(
        X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    )

    print(roc_auc_score(y_test, y_pred))
