import os

import torch
import torch.utils.data
import pandas as pd
import numpy as np
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import datetime
import multiprocessing
import dask.dataframe as dd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from typing import Tuple


def do_scoring(dl: torch.utils.data.DataLoader, scorable_model) -> Tuple[float, float]:
    with torch.inference_mode(mode=True):
        scorable_model.eval()
        y_pred = torch.Tensor().to("cuda")
        y_actual = torch.Tensor().to("cuda")

        for X, Y, padding_mask in dl:
            this_y_pred = scorable_model(X.to("cuda"), padding_mask.to("cuda"))
            y_pred = torch.cat((y_pred, this_y_pred))
            y_actual = torch.cat((y_actual, Y.to("cuda")))

        ret = dict()

        ret["auc_avg"] = roc_auc_score(
            y_actual.to("cpu"), y_pred.to("cpu")
        )  # scikit learn needs to do ops on CPU

        ret["loss"] = log_loss(y_actual.to("cpu"), y_pred.to("cpu"))

        return ret


def do_cv(
    model_factory,
    batch_size=64,
    num_epochs=100,
    criterion=None,
    early_stopping=None,
    name="model",
):
    torch.manual_seed(42)

    # TODO: stratify?
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cut_sample = pd.read_csv("cache/sample_cuts.csv")

    all_folds_scores = list()

    run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    this_run_save_path = f"cache/models/{name}_{run_time}"
    os.mkdir(this_run_save_path)

    for cv_idx, (train, test) in enumerate(cv.split(cut_sample)):
        print(f"===== Starting CV fold {cv_idx} =====")

        train_ds = FileBasedDataset("cache/mimicts", cut_sample=cut_sample.iloc[train])
        test_ds = FileBasedDataset("cache/mimicts", cut_sample=cut_sample.iloc[test])

        # TODO: do better at this (overriding max_length in test dataset to match train dataset)
        test_ds.max_len = train_ds.max_len

        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=train_ds.maxlen_padmask_collate,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        test_dl = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            collate_fn=test_ds.maxlen_padmask_collate,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        update_interval = 0.25
        model, optimizer = model_factory(train_ds.max_len)
        model.to("cuda")

        print(f"Starting training with max {num_epochs} epochs")
        for epoch_idx in range(0, num_epochs):
            model.train()

            last_update = datetime.datetime.now()
            epoch_loss = 0
            for batch_num, (X, Y, padding_mask) in enumerate(train_dl):
                optimizer.zero_grad()
                outputs = model.forward(X.to("cuda"), padding_mask.to("cuda"))

                loss = criterion(outputs, Y.to("cuda"))
                epoch_loss += loss.item()
                loss.backward()
                # TODO: what's the significance of this?
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()

                if (
                    datetime.datetime.now() - last_update
                ).total_seconds() > update_interval:
                    last_update = datetime.datetime.now()
                    print(f"Batch #{batch_num} loss: {loss.item():.5f}  ", end="\r")

            training_score = do_scoring(train_dl, model)
            validation_score = do_scoring(test_dl, model)

            print(
                f"Epoch: {epoch_idx}, loss: {training_score['loss']:1.5f} validation: {validation_score['auc_avg']:1.5f} AUC, {validation_score['loss']:1.5f} Loss"
            )

            # TODO: are best weights actually being restored?
            early_stopping(validation_score["loss"], model.state_dict())

            if early_stopping.early_stop:
                break

        if early_stopping.saved_best_weights:
            print(
                f"Loading saved best weights (loss {early_stopping.best_loss:1.5f} from early stopper)"
            )
            model.load_state_dict(early_stopping.saved_best_weights)

            torch.save(model.state_dict(), f"{this_run_save_path}/{name}_{cv_idx}.pt")

        scores = do_scoring(test_dl, model)

        all_folds_scores.append(scores)

        if early_stopping:
            early_stopping.reset()

    print(f"Cross validation complete")

    # TODO: revert to single label
    # for label_name in all_folds_scores[0].keys():
    #     this_label_scores = [x[label_name] for x in all_folds_scores]

    #     avg = np.average(this_label_scores)
    #     ci_lower = avg - 1.96 * np.std(this_label_scores, ddof=1) / np.sqrt(
    #         len(this_label_scores)
    #     )
    #     ci_upper = avg + 1.96 * np.std(this_label_scores, ddof=1) / np.sqrt(
    #         len(this_label_scores)
    #     )

    #     print(
    #         f"\t{label_name} average AUC (95% ci): {avg:.3f} ({ci_lower:.3f} - {ci_upper:.3f})"
    #     )
