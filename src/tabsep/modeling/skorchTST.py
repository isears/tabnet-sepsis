import os
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
    import pickle

    with open("cache/models/skorchCvTst/whole_model.pkl", "rb") as f:
        model = pickle.load(f)

    d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
    X = d.get_dense_normalized()
    captum_runner(model.module_, X)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        do_captum()
    else:
        if sys.argv[1] == "cv":
            do_cv()
        elif sys.argv[1] == "captum":
            do_captum()
        else:
            print(f"[-] Invalid cmd: {sys.argv[1]}")
