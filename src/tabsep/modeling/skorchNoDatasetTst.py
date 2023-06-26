import os
import pickle

import pandas as pd
import skorch
import torch
import torch.utils.data
from mvtst.models.ts_transformer import (TSTransformerEncoder,
                                         TSTransformerEncoderClassiregressor)
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from skorch import NeuralNet, NeuralNetBinaryClassifier
from skorch.callbacks import (Checkpoint, EarlyStopping, EpochScoring,
                              GradientNormClipping, LRScheduler)

from tabsep import config
from tabsep.dataProcessing import load_data_labeled_sparse
from tabsep.dataProcessing.derivedDataset import DerivedDataset
from tabsep.modeling import TSTConfig, my_auprc, my_auroc, my_f1
from tabsep.modeling.singleLR import load_to_mem
from tabsep.modeling.skorchPretrainEncoder import (
    MaskedMSELoss, MaskedMSELossSkorchConnector)

TUNING_PARAMS = {
    "lr": 0.010573607193088362,
    "dropout": 0.17431075675709043,
    "d_model_multiplier": 4,
    "num_layers": 3,
    "n_heads": 8,
    "dim_feedforward": 141,
    "batch_size": 171,
    "pos_encoding": "fixed",
    "activation": "gelu",
    "norm": "LayerNorm",
    "optimizer_name": "PlainRAdam",
    "weight_decay": 0.001,
}


def skorch_tst_factory(tst_config: TSTConfig, ds: DerivedDataset, pruner=None):
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
        TSTransformerEncoderClassiregressor,
        criterion=torch.nn.BCEWithLogitsLoss,
        # criterion__pos_weight=torch.FloatTensor([10]),
        device="cuda",
        callbacks=tst_callbacks,
        train_split=skorch.dataset.ValidSplit(0.1),
        # train_split=None,
        # TST params
        module__feat_dim=len(ds.features),
        module__max_len=ds.max_len,
        max_epochs=50,
        **tst_config.generate_skorch_full_params(),
    )

    return tst


if __name__ == "__main__":
    d = load_data_labeled_sparse("cache/sparse_labeled.pkl")
    X = d.X_sparse.to_dense()
    print(X.shape)
