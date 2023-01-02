"""
Run Skorch implementation of TST w/specific hyperparams
"""

import os

import skorch
import torch
import torch.utils.data
from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor
from sklearn.metrics import average_precision_score, roc_auc_score
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import (
    TSTCombinedConfig,
    TSTModelConfig,
    TSTRunConfig,
    my_auprc,
    my_auroc,
)

PARAMS = dict(
    optimizer__lr=1e-4,
    module__dropout=0.1,
    d_model_multiplier=8,
    module__num_layers=3,
    module__n_heads=16,
    module__dim_feedforward=256,
    iterator_train__batch_size=128,  # Should be 128
)


def skorch_tst_factory(
    tst_config: TSTCombinedConfig, ds: torch.utils.data.Dataset, pruner=None
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

    tst = NeuralNetBinaryClassifier(
        TSTransformerEncoderClassiregressor,
        criterion=torch.nn.BCEWithLogitsLoss,
        iterator_train__collate_fn=ds.maxlen_padmask_collate,
        iterator_valid__collate_fn=ds.maxlen_padmask_collate,
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
        max_epochs=15,
        **tst_config.generate_optuna_params(),
    )

    return tst


if __name__ == "__main__":
    sids_train, sids_test = split_data_consistently()

    train_ds = FileBasedDataset(sids_train)
    test_ds = FileBasedDataset(sids_test)

    model_config = TSTModelConfig()
    run_config = TSTRunConfig()
    tst_config = TSTCombinedConfig(
        save_path="cache/models/singleTst",
        model_config=model_config,
        run_config=run_config,
    )

    tst = skorch_tst_factory(tst_config, train_ds)

    tst.fit(train_ds, train_ds.get_labels())

    final_auroc = roc_auc_score(test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1])

    final_auprc = average_precision_score(
        test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1]
    )

    print("Final score:")
    print(f"\tAUROC: {final_auroc}")
    print(f"\tAverage precision: {final_auprc}")
