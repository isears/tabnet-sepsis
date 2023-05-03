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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from skorch import NeuralNet, NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
    LRScheduler,
)

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import TSTConfig, my_auprc, my_auroc, my_f1
from tabsep.modeling.singleLR import load_to_mem
from tabsep.modeling.skorchPretrainEncoder import (
    MaskedMSELoss,
    MaskedMSELossSkorchConnector,
)

TUNING_PARAMS = {
    "lr": 0.0006902481260359048,
    "dropout": 0.46203984390654546,
    "d_model_multiplier": 8,
    "num_layers": 3,
    "n_heads": 32,
    "dim_feedforward": 331,
    "batch_size": 18,
    "pos_encoding": "fixed",
    "activation": "gelu",
    "norm": "BatchNorm",
    "optimizer_name": "PlainRAdam",
    "weight_decay": 0.1,
    # "top_n_features": 100,
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
        EarlyStopping(patience=5),
        Checkpoint(
            load_best=True,
            fn_prefix=f"{tst_config.save_path}/",
            f_pickle="whole_model.pkl",
        ),
        EpochScoring(my_auroc, name="auroc", lower_is_better=False),
        EpochScoring(my_auprc, name="auprc", lower_is_better=False),
        EpochScoring(my_f1, name="f1", lower_is_better=False),
        LRScheduler("StepLR", step_size=2),
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
        # train_split=None,
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
    pretraining_ds = FileBasedDataset(
        "cache/train_examples.csv", standard_scale=True, top_n_features=None
    )
    tst_config = TSTConfig(save_path="cache/models/skorchTst", **TUNING_PARAMS)

    tst = skorch_tst_factory(tst_config, pretraining_ds, pretrained_encoder=False)

    tst.fit(pretraining_ds, y=None)

    test_ds = FileBasedDataset("cache/test_examples.csv", standard_scale=True)
    X, y = load_to_mem(test_ds)
    y_pred = tst.predict_proba(test_ds)[:, 1]

    final_auroc = roc_auc_score(y, y_pred)
    final_auprc = average_precision_score(y, y_pred)
    final_f1 = f1_score(y, y_pred.round())

    print("Final score:")
    print(f"\tAUROC: {final_auroc}")
    print(f"\tAverage precision: {final_auprc}")
    print(f"\tF1: {final_f1}")
