"""
Use all the original TST code
"""

import torch.utils.data
from mvtst.datasets.dataset import collate_unsuperv
from mvtst.models.loss import MaskedMSELoss
from mvtst.models.ts_transformer import (TSTransformerEncoder,
                                         TSTransformerEncoderClassiregressor)
from mvtst.optimizers import AdamW
from mvtst.running import UnsupervisedRunner

from tabsep import config
from tabsep.dataProcessing.fileBasedImputationDataset import (
    FileBasedDataset, FileBasedImputationDataset)
from tabsep.modeling import TSTCombinedConfig, TSTModelConfig, TSTRunConfig

if __name__ == "__main__":
    pretraining_ds = FileBasedImputationDataset("cache/pretrain_examples.csv")
    training_ds = FileBasedDataset("cache/train_examples.csv")
    testing_ds = FileBasedDataset("cache/test_examples.csv")

    # Default params
    model_config = TSTModelConfig()
    run_config = TSTRunConfig()
    combined_config = TSTCombinedConfig("cache/originalTst", model_config, run_config)

    pretraining_encoder = TSTransformerEncoder(
        feat_dim=pretraining_ds.get_num_features(),
        max_len=pretraining_ds.max_len,
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

    for idx in range(0, 3):
        metrics = runner.train_epoch(epoch_num=idx)
        print(metrics)

