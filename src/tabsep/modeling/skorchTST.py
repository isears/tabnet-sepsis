import os

import skorch
import torch
import torch.utils.data
from mvtst.models.ts_transformer import (
    TSTransformerEncoder,
    TSTransformerEncoderClassiregressor,
)
from skorch import NeuralNet, NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
    LRScheduler,
)

from tabsep import config
from tabsep.modeling import TSTConfig, my_auprc, my_auroc, my_f1

BEST_PARAMS = {
    "lr": 1.030930107062219e-05,
    "dropout": 0.11112006454363599,
    "d_model_multiplier": 16,
    "num_layers": 5,
    "n_heads": 32,
    "dim_feedforward": 429,
    "batch_size": 149,
    "pos_encoding": "fixed",
    "activation": "relu",
    "norm": "LayerNorm",
    "weight_decay": 0.1,
}


class AutoPadmaskingTST(TSTransformerEncoderClassiregressor):
    @staticmethod
    def autopadmask(X) -> torch.Tensor:
        # # examples x # feats
        squeeze_feats = torch.sum(X != -1, dim=2) > 0

        max_valid_idx = X.shape[1] - (
            torch.argmax(
                (torch.flip(squeeze_feats, dims=(1,))).int(), dim=1, keepdim=False
            )
        )

        pm = torch.zeros((X.shape[0], X.shape[1])).to(X.device)

        # TODO: more efficient way to do this?
        for bidx in range(0, X.shape[0]):
            pm[bidx, 0 : max_valid_idx[bidx]] = 1

        # TODO: if there are examples w/no data, the transformer encoder will freak out and return nans
        assert torch.sum(pm, dim=1).min() > 0.0
        return pm

    def forward(self, X):
        pm = self.autopadmask(X)

        out = super(AutoPadmaskingTST, self).forward(X, pm.bool())

        return out


def tst_factory(tst_config: TSTConfig, patience=10):
    """
    Generate TSTs wrapped in standard skorch wrapper
    """
    os.makedirs(tst_config.save_path, exist_ok=True)

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=patience),
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
