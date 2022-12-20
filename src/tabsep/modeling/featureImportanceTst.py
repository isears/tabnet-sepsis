import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import (
    GuidedBackprop,
    GuidedGradCam,
    InputXGradient,
    IntegratedGradients,
    ShapleyValueSampling,
)
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabsep import config
from tabsep.dataProcessing.labelGeneratingDataset import CoagulopathyDataset
from tabsep.modeling import split_data_consistently

torch.manual_seed(42)
np.random.seed(42)
# torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


if __name__ == "__main__":

    with open("cache/models/singleTst/whole_model.pkl", "rb") as f:
        tst = pickle.load(f)

    _, test_sids = split_data_consistently()

    attributor_batch_size = 16

    ds = CoagulopathyDataset(test_sids)
    dl = DataLoader(
        ds,
        batch_size=attributor_batch_size,
        num_workers=config.cores_available,
        collate_fn=ds.maxlen_padmask_collate,
        drop_last=False,
    )

    y_pred, y_actual, attributions = (
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
    )
    tst.module_.eval()

    for batchnum, (batch_X, batch_y) in tqdm(
        enumerate(dl), total=int(len(ds) / attributor_batch_size)
    ):
        with torch.no_grad():
            X = batch_X["X"].to("cuda")
            padding_masks = batch_X["padding_masks"].to("cuda")
            batch_preds = tst.module_.forward(X, padding_masks)

            X.requires_grad = True
            attributor = InputXGradient(tst.module_)
            batch_attributions = attributor.attribute(
                X,
                additional_forward_args=padding_masks,
                target=0,
            )

            y_pred = torch.cat((y_pred, batch_preds.to("cpu")))
            y_actual = torch.cat((y_actual, batch_y))
            attributions = torch.cat((attributions, batch_attributions.to("cpu")))

            del X
            del padding_masks
            del batch_X
            torch.cuda.empty_cache()

    auroc = roc_auc_score(y_actual, y_pred)
    auprc = average_precision_score(y_actual, y_pred)

    print(f"Baseline performance - AUROC {auroc:.3f}; AUPRC {auprc:.3f}")

    torch.save(attributions, "cache/models/singleTst/attributions.pt")
