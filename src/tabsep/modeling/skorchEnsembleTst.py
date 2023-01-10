import os

import numpy as np
import pandas as pd
import skorch
import torch
import torch.utils.data
from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor
from sklearn.metrics import average_precision_score, roc_auc_score
from skorch import NeuralNet, NeuralNetRegressor
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from tabsep import config
from tabsep.dataProcessing.ensembleDataset import EnsembleDataset
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import TSTConfig


class EnsembleScorer:
    def __init__(self) -> None:
        pass

    def _correct_preds(self, net, X):
        # TODO: this is slow, return to fix if time allows
        y_actual = torch.stack([i[3] for i in X])
        lr_preds = torch.stack([i[2] for i in X])
        corrected_preds = lr_preds + np.squeeze(net.predict(X))
        return y_actual, corrected_preds

    def auroc(self, net, X, y):
        return roc_auc_score(*self._correct_preds(net, X))

    def auprc(self, net, X, y):
        return average_precision_score(*self._correct_preds(net, X))


# Need to correct shape of predictions coming out of skorch model for some reason
class MyMSELoss(torch.nn.MSELoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.squeeze(input)
        return super().forward(input, target)


def ensemble_tst_factory(tst_config: TSTConfig, ds: FileBasedDataset, pruner=None):
    """
    Generate TSTs wrapped in standard skorch wrapper
    """
    os.makedirs(tst_config.save_path, exist_ok=True)

    ensemble_scorer = EnsembleScorer()

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=3),
        Checkpoint(
            load_best=True,
            fn_prefix=f"{tst_config.save_path}/",
            f_pickle="whole_model.pkl",
        ),
        EpochScoring(ensemble_scorer.auroc, name="auroc", lower_is_better=False),
        EpochScoring(ensemble_scorer.auprc, name="auprc", lower_is_better=False),
    ]

    if pruner is not None:
        tst_callbacks.append(pruner)

    tst = NeuralNetRegressor(
        TSTransformerEncoderClassiregressor,
        criterion=MyMSELoss,
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
        max_epochs=1,
        **tst_config.generate_skorch_full_params(),
    )

    return tst


if __name__ == "__main__":
    ds = EnsembleDataset("cache/train_examples.csv")
    tst_config = TSTConfig(
        save_path="cache/models/ensembleSkorchTst",
        lr=7.729784380014021e-05,
        dropout=0.6594354080067655,
        d_model_multiplier=4,
        num_layers=2,
        n_heads=8,
        dim_feedforward=485,
        batch_size=94,
        pos_encoding="fixed",
        activation="relu",
        norm="BatchNorm",
        optimizer_name="RAdam",
        weight_decay=0.1,
    )

    tst = ensemble_tst_factory(tst_config, ds)

    tst.fit(ds, y=None)

    test_ds = EnsembleDataset("cache/test_examples.csv")
    # Need to get these after ds shuffling
    lr_preds = test_ds.examples["lr_pred"].to_numpy()

    corrections = tst.predict(test_ds)

    corrected_preds = lr_preds + np.squeeze(corrections)

    auroc = roc_auc_score(test_ds.get_labels(), corrected_preds)
    auprc = average_precision_score(test_ds.get_labels(), corrected_preds)

    print("Final test metrics:")
    print(f"\tAUROC: {auroc}")
    print(f"\tAverage precision: {auprc}")
