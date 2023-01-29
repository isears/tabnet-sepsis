"""
Run Skorch implementation of TST and Pretrainer w/specific hyperparams
"""

import os

import skorch
import torch
import torch.utils.data
from mvtst.datasets.dataset import collate_unsuperv
from mvtst.models.loss import MaskedMSELoss, NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import (
    TransformerBatchNormEncoderLayer,
    TSTransformerEncoder,
)
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
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset, get_feature_labels
from tabsep.dataProcessing.fileBasedImputationDataset import FileBasedImputationDataset
from tabsep.dataProcessing.fileBasedTransductionDataset import (
    FileBasedTransductionDataset,
)
from tabsep.modeling import TSTConfig, my_auprc, my_auroc


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


def skorch_pretraining_encoder_factory(
    tst_config: TSTConfig, ds: FileBasedImputationDataset, pruner=None
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
            f_pickle="pretrained_encoder.pkl",
        ),
    ]

    pretraining_encoder = NeuralNet(
        TSTransformerEncoder,  # TODO: also try batchnorm encoder?
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
        **tst_config.generate_skorch_pretraining_params(),
    )

    return pretraining_encoder


if __name__ == "__main__":
    features_to_mask_names = [
        "Heart Rate",
        # "Arterial Blood Pressure systolic",
        # "Arterial Blood Pressure diastolic",
        # "Arterial Blood Pressure mean",
        # "Respiratory Rate",
        # "O2 saturation pulseoxymetry",
        # "Chloride (serum)",
        # "Potassium (serum)",
        # "PTT",
        # "INR",
        # "Platelet Count",
    ]

    feature_names = get_feature_labels()

    features_to_mask_indices = [
        feature_names.index(fname) for fname in features_to_mask_names
    ]

    pretraining_ds = FileBasedTransductionDataset(
        "cache/pretrain_examples.csv", mask_feats=features_to_mask_indices
    )
    tst_config = TSTConfig(save_path="cache/models/skorchPretrainingTst")

    pretraining_encoder = skorch_pretraining_encoder_factory(tst_config, pretraining_ds)

    pretraining_encoder.fit(pretraining_ds)
