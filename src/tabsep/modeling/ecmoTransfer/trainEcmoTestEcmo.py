import pandas as pd
import skorch
import torch
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
)
from skorch.dataset import ValidSplit

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset, TruncatedFBD
from tabsep.modeling import TSTConfig, my_auprc, my_auroc

PARAMS = {
    "lr": 1e-4,
    "dropout": 0.1,
    "d_model_multiplier": 8,
    "num_layers": 3,
    "n_heads": 16,
    "dim_feedforward": 256,
    "batch_size": 64,
    "pos_encoding": "learnable",
    "activation": "relu",
    "norm": "LayerNorm",
    "optimizer_name": "RAdam",
    "weight_decay": 0.01,
}


def get_ecmo_examples():
    ecmo_stayids = pd.read_csv("cache/ecmo_stayids.csv").squeeze("columns").to_list()

    train_examples = pd.read_csv("cache/train_examples.csv")
    test_examples = pd.read_csv("cache/test_examples.csv")

    all_examples = pd.concat([train_examples, test_examples])

    ecmo_examples = all_examples[all_examples["stay_id"].isin(ecmo_stayids)]
    return ecmo_examples


def get_nonecmo_examples():
    ecmo_stayids = pd.read_csv("cache/ecmo_stayids.csv").squeeze("columns").to_list()

    train_examples = pd.read_csv("cache/train_examples.csv")
    test_examples = pd.read_csv("cache/test_examples.csv")

    all_examples = pd.concat([train_examples, test_examples])

    nonecmo_examples = all_examples[~all_examples["stay_id"].isin(ecmo_stayids)]
    return nonecmo_examples


if __name__ == "__main__":
    tst_config = TSTConfig(save_path="cache/models/ecmoTst", **PARAMS)
    examples = get_ecmo_examples()
    ds = TruncatedFBD(examples)

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=3),
        Checkpoint(
            load_best=True,
            fn_prefix=f"{tst_config.save_path}/",
            f_pickle="trainEcmo.pkl",
        ),
        EpochScoring(my_auroc, name="auroc", lower_is_better=False),
        EpochScoring(my_auprc, name="auprc", lower_is_better=False),
    ]

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
        train_split=ValidSplit(cv=5),
        # TST params
        module__feat_dim=ds.get_num_features(),
        module__max_len=ds.max_len,
        max_epochs=50,
        **tst_config.generate_skorch_full_params(),
    )

    tst.fit(ds, y=None)
