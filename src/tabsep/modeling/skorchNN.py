import tempfile

import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from skorch import NeuralNet, NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import (
    my_auprc,
    my_auroc,
    my_f1,
)


class SimpleFFNN(nn.Module):
    def __init__(
        self,
        n_hidden: int,
        width: int,
        n_features: int,
        activation_fn: str,
        dropout: float,
    ):
        super(SimpleFFNN, self).__init__()
        self.input_layer = nn.Linear(n_features, width)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(width, width) for i in range(n_hidden)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.output_layer = nn.Linear(width, 1)
        self.activation_fn = F.relu

        if activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "gelu":
            self.activation_fn == F.gelu
        else:
            raise ValueError(
                f"{activation_fn} not among available activation functions"
            )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_fn(x)

        for l in self.hidden_layers:
            x = self.dropout(x)
            x = l(x)
            x = self.activation_fn(x)

        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def nn_factory(
    lr: float = 0.01,
    n_hidden: int = 1,
    width: int = 10,
    activation_fn: str = "relu",
    dropout=0.01,
    patience=5,
):
    checkpoint_dir = tempfile.TemporaryDirectory()
    cbs = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=patience),
        Checkpoint(
            load_best=True,
            fn_prefix=checkpoint_dir.name,
            f_pickle="whole_model.pkl",
        ),
        EpochScoring(my_auprc, name="auprc", lower_is_better=False),
    ]

    sffnn = NeuralNetBinaryClassifier(
        SimpleFFNN,
        module__n_hidden=n_hidden,
        module__width=width,
        module__n_features=86,
        module__activation_fn=activation_fn,
        module__dropout=dropout,
        criterion=torch.nn.BCEWithLogitsLoss,
        # criterion__pos_weight=torch.FloatTensor([10]),
        device="cuda",
        callbacks=cbs,
        train_split=skorch.dataset.ValidSplit(0.1),
        max_epochs=50,
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
    )

    return sffnn
