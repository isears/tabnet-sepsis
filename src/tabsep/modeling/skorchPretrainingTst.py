"""
Run Skorch implementation of TST and Pretrainer w/specific hyperparams
"""

import os

import skorch
import torch
import torch.utils.data
from mvtst.datasets.dataset import collate_unsuperv
from mvtst.models.loss import MaskedMSELoss, NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import TSTransformerEncoder
from mvtst.optimizers import AdamW
from sklearn.metrics import average_precision_score, roc_auc_score
from skorch import NeuralNet, NeuralNetRegressor
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.dataProcessing.fileBasedImputationDataset import FileBasedImputationDataset
from tabsep.modeling import my_auprc, my_auroc


def translate_optuna_params(params: dict):
    """
    Optuna cannot directly set d_model, must instead set d_model_multiplier

    This functions translates into skorch-consumable parameters
    """
    params["module__d_model"] = params["module__n_heads"] * params["d_model_multiplier"]
    del params["d_model_multiplier"]
    return params


class MaskedMSELossSkorchConnector(MaskedMSELoss):
    """
    Need a connector b/c skorch expects all loss functions to take just two args
    """

    def forward(self, y_pred, target_packed):
        """
        target_packed should be dict such that:
        y_true = target_packed['y_true']
        mask = target_packed['mask']
        """
        return super().forward(y_pred, **target_packed)


PARAMS = dict(
    optimizer=AdamW,
    optimizer__lr=1e-4,
    module__dropout=0.1,
    d_model_multiplier=8,
    module__num_layers=3,
    module__n_heads=16,
    module__dim_feedforward=256,
    module__pos_encoding="learnable",  # or fixed? Does that make sense for pretraining?
    module__activation="gelu",
    module__norm="BatchNorm",
    module__freeze=False,
    iterator_train__batch_size=128,  # Should be 128
)


def skorch_pretraining_encoder_factory(
    params: dict, ds: FileBasedImputationDataset, save_path: str, pruner=None
):
    """
    Generate TSTs wrapped in standard skorch wrapper
    """
    os.makedirs(save_path, exist_ok=True)

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=3),
        Checkpoint(
            load_best=True,
            fn_prefix=f"{save_path}/",
            f_pickle="pretrained_encoder.pkl",
        ),
    ]

    # TODO: convert to NeuralNetRegressor?
    pretraining_encoder = NeuralNetRegressor(
        TSTransformerEncoder,
        iterator_train__collate_fn=ds.collate_unsuperv_skorch,
        iterator_valid__collate_fn=ds.collate_unsuperv_skorch,
        iterator_train__num_workers=config.cores_available,
        iterator_valid__num_workers=config.cores_available,
        iterator_train__pin_memory=True,
        iterator_valid__pin_memory=True,
        criterion=MaskedMSELossSkorchConnector,
        criterion__reduction="mean",
        device="cuda",
        callbacks=tst_callbacks,
        train_split=skorch.dataset.ValidSplit(0.1),
        # TST params
        module__feat_dim=ds.get_num_features(),
        module__max_len=ds.max_len,
        max_epochs=25,
        **translate_optuna_params(params),
    )

    return pretraining_encoder


if __name__ == "__main__":
    pretraining_ds = FileBasedImputationDataset("cache/pretrain_examples.csv")

    pretraining_encoder = skorch_pretraining_encoder_factory(
        PARAMS, pretraining_ds, "cache/models/skorchPretrainingTst"
    )

    pretraining_encoder.fit(pretraining_ds)
