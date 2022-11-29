"""
Train single transformer model for downstream analysis
"""

import datetime
import os

import pandas as pd
import skorch
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from skorch import NeuralNet
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling.tstEstimator import AdamW, TstWrapper
from tabsep.modeling.tstImpl import TSTransformerEncoderClassiregressor

if __name__ == "__main__":

    cut_sample = pd.read_csv("cache/sample_cuts.csv")
    sids_train, sids_test = train_test_split(
        cut_sample["stay_id"].to_list(), test_size=0.1, random_state=42
    )

    save_path = f"cache/models/singleTst"
    os.mkdir(save_path)

    pd.DataFrame(data={"stay_id": sids_train}).to_csv(
        f"{save_path}/train_stayids.csv", index=False
    )

    pd.DataFrame(data={"stay_id": sids_test}).to_csv(
        f"{save_path}/test_stayids.csv", index=False
    )

    train_ds = FileBasedDataset(sids_train)
    test_ds = FileBasedDataset(sids_test)

    print("[+] Data loaded, training...")

    tst = NeuralNet(
        TSTransformerEncoderClassiregressor,
        criterion=torch.nn.BCELoss,
        optimizer=AdamW,
        iterator_train__collate_fn=train_ds.maxlen_padmask_collate,
        iterator_valid__collate_fn=train_ds.maxlen_padmask_collate,
        iterator_train__num_workers=config.cores_available,
        iterator_valid__num_workers=config.cores_available,
        iterator_train__pin_memory=True,
        iterator_valid__pin_memory=True,
        device="cuda",
        max_epochs=100,  # Large for early stopping
        train_split=skorch.dataset.ValidSplit(0.1),
        callbacks=[
            EarlyStopping(patience=3),
            Checkpoint(
                load_best=True, fn_prefix=f"{save_path}/", f_pickle="whole_model.pkl"
            ),
        ],
        # TST params
        module__feat_dim=train_ds.get_num_features(),
        module__d_model=128,
        module__dim_feedforward=256,
        module__max_len=train_ds.max_len,
        module__n_heads=16,
        module__num_classes=1,
        module__num_layers=3,
    )

    tst.fit(train_ds)

    print(f"[+] Training complete, saving to {save_path}")

    tst.save_params(f_params=f"{save_path}/model.pkl")

    preds = tst.predict_proba(test_ds)
    torch.save(preds, f"{save_path}/preds.pt")
    score = roc_auc_score(FileBasedDataset(sids_test).get_labels(), preds)
    print(f"Validation score: {score}")

    with open(f"{save_path}/roc_auc_score.txt", "w") as f:
        f.write(str(score))
        f.write("\n")
