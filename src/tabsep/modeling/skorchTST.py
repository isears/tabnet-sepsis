import os
import pickle
import sys

import skorch
import torch
import torch.utils.data
from mvtst.models.ts_transformer import (
    TSTransformerEncoder,
    TSTransformerEncoderClassiregressor,
)
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from skorch import NeuralNet, NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
    LRScheduler,
)

from tabsep import config
from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import (
    BaseModelRunner,
    TabsepModelFactory,
    TSTConfig,
    my_auprc,
    my_auroc,
    my_f1,
)
from tabsep.modeling.commonCaptum import captum_runner
from tabsep.modeling.commonCV import cv_runner

BEST_PARAMS = {
    "lr": 3.954336616242573e-05,
    "dropout": 0.2553653431379216,
    "d_model_multiplier": 1,
    "num_layers": 11,
    "n_heads": 32,
    "dim_feedforward": 325,
    "batch_size": 66,
    "pos_encoding": "fixed",
    "activation": "relu",
    "norm": "BatchNorm",
    "optimizer_name": "PlainRAdam",
    "weight_decay": 0.1,
}


class AutoPadmaskingTST(TSTransformerEncoderClassiregressor):
    def forward(self, X):
        # # examples x # feats
        squeeze_feats = torch.sum(X != -1, dim=2) > 0

        max_valid_idx = (X.shape[1] - 1) - (
            torch.argmax(
                (torch.flip(squeeze_feats, dims=(1,))).int(), dim=1, keepdim=False
            )
        )

        pm = torch.zeros((X.shape[0], X.shape[1])).to(X.get_device())

        # TODO: more efficient way to do this?
        for bidx in range(0, X.shape[0]):
            pm[bidx, 0 : max_valid_idx[bidx]] = 1

        return super().forward(X, pm.bool())


def tst_factory(tst_config: TSTConfig):
    """
    Generate TSTs wrapped in standard skorch wrapper
    """
    os.makedirs(tst_config.save_path, exist_ok=True)

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=3),
        Checkpoint(
            load_best=True,
            fn_prefix=f"{tst_config.save_path}/",
            f_pickle="whole_model.pkl",
        ),
        EpochScoring(my_auroc, name="auroc", lower_is_better=False),
        EpochScoring(my_auprc, name="auprc", lower_is_better=False),
        EpochScoring(my_f1, name="f1", lower_is_better=False),
        # LRScheduler("StepLR", step_size=2),
    ]

    tst = NeuralNetBinaryClassifier(
        AutoPadmaskingTST,
        criterion=torch.nn.BCEWithLogitsLoss,
        # criterion__pos_weight=torch.FloatTensor([10]),
        device="cuda",
        callbacks=tst_callbacks,
        train_split=skorch.dataset.ValidSplit(0.1),
        # train_split=None,
        # TST params
        module__feat_dim=88,
        module__max_len=120,
        max_epochs=50,
        **tst_config.generate_skorch_full_params(),
    )

    return tst


def do_cv():
    d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
    X = d.get_dense_normalized()
    y = d.get_labels()

    conf = TSTConfig(save_path="cache/models/skorchCvTst", **BEST_PARAMS)
    cv_runner(lambda: tst_factory(conf), X, y)


def do_captum():
    with open("cache/AutoPadmaskingTST/model.pkl", "rb") as f:
        model = pickle.load(f)

    X = torch.load(f"cache/AutoPadmaskingTST/X_test.pt")
    captum_runner(model.module_, X)


def train_one():
    d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
    X = d.get_dense_normalized()
    y = d.get_labels()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    m = tst_factory(TSTConfig(save_path="cache/models/skorchCvTst", **BEST_PARAMS))

    m.fit(X_train, y_train)
    preds = m.predict_proba(X_test)[:, 1]

    print(f"Final AUROC: {roc_auc_score(y_test, preds)}")
    print(f"Final AUPRC: {average_precision_score(y_test, preds)}")

    save_dir = f"cache/{m.module_.__class__.__name__}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/model.pkl", "wb") as f:
        m.module_ = m.module_.to("cpu")
        pickle.dump(m, f)

    torch.save(X_train, f"{save_dir}/X_train.pt")
    torch.save(X_test, f"{save_dir}/X_test.pt")

    print(f"[+] Saved to {save_dir}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        do_captum()
    else:
        cmd = sys.argv[1]
        if cmd == "cv":
            do_cv()
        elif cmd == "captum":
            do_captum()
        elif cmd == "train":
            train_one()
        else:
            print(f"[-] Invalid cmd: {cmd}")
