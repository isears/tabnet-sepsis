import pickle
from dataclasses import dataclass

import torch


def load_data_labeled_sparse(path: str):
    with open(path, "rb") as f:
        sparse_data = pickle.load(f)

    return sparse_data


@dataclass
class LabeledSparseTensor:
    stay_ids: list
    features: list
    X_sparse: torch.Tensor
    y: torch.Tensor
