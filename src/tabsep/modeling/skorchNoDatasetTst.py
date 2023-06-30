import os
import pickle

import pandas as pd
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
from tabsep.dataProcessing.derivedDataset import DerivedDataset
from tabsep.modeling import TSTConfig, my_auprc, my_auroc, my_f1
from tabsep.modeling.singleLR import load_to_mem
from tabsep.modeling.skorchPretrainEncoder import (
    MaskedMSELoss,
    MaskedMSELossSkorchConnector,
)


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


TUNING_PARAMS = {
    # "lr": 0.010573607193088362,
    # "dropout": 0.17431075675709043,
    # "d_model_multiplier": 4,
    # "num_layers": 3,
    # "n_heads": 8,
    # "dim_feedforward": 141,
    # "batch_size": 171,
    # "pos_encoding": "fixed",
    # "activation": "gelu",
    # "norm": "LayerNorm",
    # "optimizer_name": "PlainRAdam",
    # "weight_decay": 0.001,
}


def skorch_tst_factory(tst_config: TSTConfig, pruner=None):
    """
    Generate TSTs wrapped in standard skorch wrapper
    """
    os.makedirs(tst_config.save_path, exist_ok=True)

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=10),
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

    if pruner is not None:
        tst_callbacks.append(pruner)

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


# TODO: maybe move this into dataprocessing

if __name__ == "__main__":
    d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
    X = d.get_dense_normalized()
    y = d.get_labels()
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    tst_config = TSTConfig(save_path="cache/models/skorchTst", **TUNING_PARAMS)
    tst = skorch_tst_factory(tst_config)

    tst.fit(X_train, y_train)

    preds = tst.predict_proba(X_test)[:, 1]
    print(roc_auc_score(y_test, preds))
