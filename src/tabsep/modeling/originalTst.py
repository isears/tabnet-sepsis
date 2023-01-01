"""
Use all the original TST code
"""

import torch.utils.data
from mvtst.datasets.dataset import collate_unsuperv
from mvtst.models.loss import MaskedMSELoss, NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import (
    TSTransformerEncoder,
    TSTransformerEncoderClassiregressor,
)
from mvtst.optimizers import AdamW
from mvtst.running import SupervisedRunner, UnsupervisedRunner

from tabsep import config
from tabsep.dataProcessing.fileBasedImputationDataset import (
    FileBasedDataset,
    FileBasedImputationDataset,
)
from tabsep.modeling import TSTCombinedConfig, TSTModelConfig, TSTRunConfig

if __name__ == "__main__":
    pretraining_ds = FileBasedImputationDataset("cache/pretrain_examples.csv")
    training_ds = FileBasedDataset("cache/train_examples.csv")
    testing_ds = FileBasedDataset("cache/test_examples.csv")

    pretraining_encoder = TSTransformerEncoder(
        feat_dim=pretraining_ds.get_num_features(),
        # NOTE: this will fail if pretraining ds has longer maxlen than training ds
        max_len=training_ds.max_len,
        d_model=128,
        n_heads=16,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
        freeze=False,
    ).to("cuda")

    pretraining_dl = torch.utils.data.DataLoader(
        pretraining_ds,
        collate_fn=lambda x: collate_unsuperv(x, max_len=pretraining_ds.max_len),
        num_workers=config.cores_available,
        batch_size=128,
        pin_memory=True,
    )

    runner = UnsupervisedRunner(
        model=pretraining_encoder,
        dataloader=pretraining_dl,
        device="cuda",
        loss_module=MaskedMSELoss(reduction="none"),
        optimizer=AdamW(pretraining_encoder.parameters(), lr=1e-3),
    )

    for idx in range(0, 7):
        run_metrics = runner.train_epoch(epoch_num=idx)
        print(f"\n{run_metrics}")

    tst = TSTransformerEncoderClassiregressor(
        feat_dim=pretraining_ds.get_num_features(),
        max_len=pretraining_ds.max_len,
        d_model=128,
        n_heads=16,
        num_layers=3,
        dim_feedforward=256,
        num_classes=2,
        dropout=0.1,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
        freeze=False,
    ).to("cuda")

    tst.transformer_encoder = pretraining_encoder

    training_dl = torch.utils.data.DataLoader(
        training_ds,
        collate_fn=training_ds.maxlen_padmask_collate,
        num_workers=config.cores_available,
        batch_size=128,
        pin_memory=True,
    )

    # TODO: should probably do this both ways
    testing_ds.max_len = training_ds.max_len
    testing_dl = torch.utils.data.DataLoader(
        testing_ds,
        collate_fn=testing_ds.maxlen_padmask_collate,
        num_workers=config.cores_available,
        batch_size=128,
        pin_memory=True,
    )

    training_runner = SupervisedRunner(
        model=tst,
        dataloader=training_dl,
        device="cuda",
        loss_module=NoFussCrossEntropyLoss(reduction="none"),
        optimizer=AdamW(tst.parameters(), lr=1e-3, weight_decay=0),
    )

    testing_runner = SupervisedRunner(
        model=tst,
        dataloader=testing_dl,
        device="cuda",
        loss_module=NoFussCrossEntropyLoss(reduction="none"),
        optimizer=None,  # Shouldn't need this
    )

    for idx in range(0, 20):
        train_metrics = training_runner.train_epoch(epoch_num=idx)
        test_metrics, _ = testing_runner.evaluate(epoch_num=idx)
        print(f"\n{train_metrics}")
        print(test_metrics)

