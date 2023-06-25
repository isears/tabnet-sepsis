from dataclasses import dataclass

import torch


def load_labeld_sparse_tensor(path: str):
    raise NotImplementedError
    return X, y


@dataclass
class LabeledSparseTensor:
    stay_ids: list
    features: list
    X_sparse: torch.Tensor
    y: torch.Tensor
