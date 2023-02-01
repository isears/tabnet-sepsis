"""
Find optimal hyperparameters for TST
"""
from dataclasses import fields

import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from tabsep import config
from tabsep.modeling import TSTConfig
from tabsep.modeling.lightningTst import (
    LitTst,
    generate_dataloaders,
    lightning_tst_factory,
)


def objective(trial: optuna.Trial) -> float:
    # Parameters to tune:
    trial.suggest_float("lr", 1e-10, 0.1, log=True)
    trial.suggest_float("dropout", 0.01, 0.7)
    trial.suggest_categorical("d_model_multiplier", [1, 2, 4, 8, 16, 32, 64])
    trial.suggest_int("num_layers", 1, 15)
    trial.suggest_categorical("n_heads", [4, 8, 16, 32, 64])
    trial.suggest_int("dim_feedforward", 128, 512)
    trial.suggest_int("batch_size", 8, 128)
    trial.suggest_categorical("pos_encoding", ["fixed", "learnable"])
    trial.suggest_categorical("activation", ["gelu", "relu"])
    trial.suggest_categorical("norm", ["BatchNorm", "LayerNorm"])
    trial.suggest_categorical("optimizer_name", ["AdamW", "PlainRAdam", "RAdam"])
    trial.suggest_categorical("weight_decay", [1e-3, 1e-2, 1e-1, 0])

    tst_config = TSTConfig(save_path="cache/models/lightningTuning", **trial.params)

    train_dl, valid_dl = generate_dataloaders(tst_config.batch_size)

    model = lightning_tst_factory(
        tst_config,
        train_dl.dataset,
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=150,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        devices=8,  # TODO: will eventually need to modify this
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            checkpoint_callback,
        ],
        enable_checkpointing=True,
    )

    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=valid_dl,
        )
    except RuntimeError as e:
        del trainer
        del model
        del train_dl
        del valid_dl
        torch.cuda.empty_cache()
        if "PYTORCH_CUDA_ALLOC_CONF" in str(e):
            print(f"Caught OOM, skipping this trial")
            return None
        else:
            raise e

    best_model = LitTst.load_from_checkpoint(checkpoint_callback.best_model_path)
    test_results = trainer.test(model=best_model, dataloaders=valid_dl)

    return test_results[0]["Average Precision"]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    pruner = None
    # pruner = optuna.pruners.PercentilePruner(25.0)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=1000)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
