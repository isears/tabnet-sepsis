"""
Run Skorch implementation of TST w/specific hyperparams
"""

import os
import pickle

import skorch
import torch
import torch.utils.data
from mvtst.models.ts_transformer import (
    TSTransformerEncoder,
    TSTransformerEncoderClassiregressor,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from skorch import NeuralNet, NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import TSTConfig, my_auprc, my_auroc
from tabsep.modeling.skorchPretrainEncoder import (
    MaskedMSELoss,
    MaskedMSELossSkorchConnector,
)

TUNING_PARAMS = {
    "lr": 8.640938824421861e-06,
    "dropout": 0.3283181195131914,
    "d_model_multiplier": 64,
    "num_layers": 8,
    "n_heads": 4,
    "dim_feedforward": 460,
    "batch_size": 138,
    "pos_encoding": "learnable",
    "activation": "relu",
    "norm": "LayerNorm",
    "optimizer_name": "RAdam",
    "weight_decay": 0.01,
}


def skorch_tst_factory(
    tst_config: TSTConfig, ds: FileBasedDataset, pruner=None, pretrained_encoder=False
):
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
    ]

    if pruner is not None:
        tst_callbacks.append(pruner)

    tst = NeuralNetBinaryClassifier(
        TSTransformerEncoderClassiregressor,
        criterion=torch.nn.BCEWithLogitsLoss,
        iterator_train__collate_fn=ds.maxlen_padmask_collate_skorch,
        iterator_valid__collate_fn=ds.maxlen_padmask_collate_skorch,
        iterator_train__num_workers=config.cores_available,
        iterator_valid__num_workers=config.cores_available,
        iterator_train__pin_memory=True,
        iterator_valid__pin_memory=True,
        device="cuda",
        callbacks=tst_callbacks,
        train_split=skorch.dataset.ValidSplit(0.1),
        # TST params
        module__feat_dim=ds.get_num_features(),
        module__max_len=ds.max_len,
        max_epochs=50,
        **tst_config.generate_skorch_full_params(),
    )

    if pretrained_encoder:
        with open(
            "cache/models/skorchPretrainingTst/pretrained_encoder.pkl", "rb"
        ) as f:
            skorch_encoder = pickle.load(f)

        tst.initialize()
        tst.module_.transformer_encoder.load_state_dict(
            skorch_encoder.module_.transformer_encoder.state_dict()
        )

    return tst


if __name__ == "__main__":

    pretraining_ds = FileBasedDataset("cache/train_examples.csv", standard_scale=True)
    tst_config = TSTConfig(save_path="cache/models/skorchTst", **TUNING_PARAMS)

    tst = skorch_tst_factory(tst_config, pretraining_ds, pretrained_encoder=False)

    tst.fit(pretraining_ds, y=None)
