import tempfile

import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from skorch import NeuralNet, NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
    LRScheduler,
)

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import (
    BaseModelRunner,
    TabsepModelFactory,
    my_auprc,
    my_auroc,
    my_f1,
)
from tabsep.modeling.commonCV import cv_runner


class SimpleFFNN(nn.Module):
    def __init__(self, n_hidden: int, width: int, n_features: int):
        super(SimpleFFNN, self).__init__()
        self.input_layer = nn.Linear(n_features, width)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(width, width) for i in range(n_hidden)]
        )
        self.output_layer = nn.Linear(width, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for l in self.hidden_layers:
            x = l(x)
            x = F.relu(x)

        x = self.output_layer(x)
        return x


class SkorchNNFactory(TabsepModelFactory):
    def __init__(self, lr: float = 0.01, n_hidden: int = 1, width=44) -> None:
        self.lr = lr
        self.n_hidden = n_hidden
        self.width = width
        self.checkpoint_dir = tempfile.TemporaryDirectory()
        super().__init__()

    def __call__(self):
        cbs = [
            GradientNormClipping(gradient_clip_value=4.0),
            EarlyStopping(patience=10),
            Checkpoint(
                load_best=True,
                fn_prefix=self.checkpoint_dir.name,
                f_pickle="whole_model.pkl",
            ),
            EpochScoring(my_auroc, name="auroc", lower_is_better=False),
            EpochScoring(my_auprc, name="auprc", lower_is_better=False),
            EpochScoring(my_f1, name="f1", lower_is_better=False),
        ]

        sffnn = NeuralNetBinaryClassifier(
            SimpleFFNN,
            module__n_hidden=self.n_hidden,
            module__width=self.width,
            module__n_features=88,
            criterion=torch.nn.BCEWithLogitsLoss,
            # criterion__pos_weight=torch.FloatTensor([10]),
            device="cuda",
            callbacks=cbs,
            train_split=skorch.dataset.ValidSplit(0.1),
            max_epochs=50,
            optimizer=torch.optim.Adam,
            optimizer__lr=self.lr,
        )

        return sffnn


class NNRunner(BaseModelRunner):
    def cv(self):
        d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
        X = d.get_snapshot()
        y = d.get_labels()
        cv_runner(SkorchNNFactory(), X, y)

    def hparams(self):
        raise NotImplementedError()

    def importance(self):
        raise NotImplementedError()


if __name__ == "__main__":
    NNRunner().parse_cmdline()
