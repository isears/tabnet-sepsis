import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from mvtst.models.loss import NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import TSTConfig


class LitTst(pl.LightningModule):
    def __init__(
        self, tst: TSTransformerEncoderClassiregressor, tst_config: TSTConfig
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tst = tst

        self.tst_config = tst_config
        # self.loss_fn = NoFussCrossEntropyLoss()

        self.auroc_metric = torchmetrics.AUROC(task="binary")
        self.average_precision_metric = torchmetrics.AveragePrecision(task="binary")

    def training_step(self, batch, batch_idx):
        X, y, pm, IDs = batch
        preds = torch.squeeze(self.tst(X, pm))

        # loss = self.loss_fn(preds, y)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, pm, IDs = batch
        preds = torch.squeeze(self.tst(X, pm))

        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y)
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": val_loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):
        self.auroc_metric(outputs["preds"], outputs["target"])
        self.log("AUC", self.auroc_metric)

        self.average_precision_metric(outputs["preds"], outputs["target"])
        self.log("Average Precision", self.average_precision_metric)

    def test_step(self, batch, batch_idx):
        X, y, pm, IDx = batch
        preds = torch.squeeze(self.tst(X, pm))
        test_loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y)
        self.log("test_loss", test_loss)

        return {"loss": test_loss, "preds": preds, "target": y}

    def test_step_end(self, outputs):
        self.auroc_metric(outputs["preds"], outputs["target"])
        self.log("AUC", self.auroc_metric)

        self.average_precision_metric(outputs["preds"], outputs["target"])
        self.log("Average Precision", self.average_precision_metric)

    def configure_optimizers(self):
        return self.tst_config.generate_optimizer(self.parameters())


def lightning_tst_factory(tst_config: TSTConfig, ds: FileBasedDataset):
    tst = TSTransformerEncoderClassiregressor(
        **tst_config.generate_model_params(),
        feat_dim=ds.get_num_features(),
        max_len=ds.max_len
    )

    lightning_wrapper = LitTst(tst, tst_config)

    return lightning_wrapper


def generate_dataloaders(batch_size: int):
    examples = pd.read_csv("cache/train_examples.csv")
    train_examples, valid_examples = train_test_split(
        examples, test_size=0.1, random_state=42
    )
    training_ds = FileBasedDataset(train_examples)
    test_ds = FileBasedDataset("cache/test_examples.csv")
    valid_ds = FileBasedDataset(valid_examples)
    valid_ds.max_len = training_ds.max_len
    test_ds.max_len = training_ds.max_len

    training_dl = torch.utils.data.DataLoader(
        training_ds,
        collate_fn=training_ds.maxlen_padmask_collate,
        num_workers=config.cores_available,
        batch_size=batch_size,
        pin_memory=True,
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        collate_fn=valid_ds.maxlen_padmask_collate,
        num_workers=config.cores_available,
        batch_size=batch_size,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        collate_fn=test_ds.maxlen_padmask_collate,
        num_workers=config.cores_available,
        batch_size=batch_size,
    )

    return training_dl, valid_dl, test_dl


if __name__ == "__main__":
    problem_params = {
        "lr": 0.0006800060968099381,
        "dropout": 0.6260221855538255,
        "d_model_multiplier": 4,
        "num_layers": 3,
        "n_heads": 64,
        "dim_feedforward": 251,
        "batch_size": 86,
        "pos_encoding": "fixed",
        "activation": "relu",
        "norm": "LayerNorm",
        "optimizer_name": "RAdam",
        "weight_decay": 0.001,
    }
    tst_config = TSTConfig(save_path="lightningTst", **problem_params)
    training_dl, valid_dl, test_dl = generate_dataloaders(tst_config.batch_size)

    model = lightning_tst_factory(tst_config, training_dl.dataset)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=150,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        devices=config.gpus_available,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True),
            checkpoint_callback,
        ],
        enable_checkpointing=True,
    )

    trainer.fit(
        model=model,
        train_dataloaders=training_dl,
        val_dataloaders=valid_dl,
    )

    best_model = LitTst.load_from_checkpoint(checkpoint_callback.best_model_path)
    results = trainer.test(model=best_model, dataloaders=valid_dl)

    print(type(results))
    print(results)
