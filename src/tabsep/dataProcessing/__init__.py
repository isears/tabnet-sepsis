from dataclasses import dataclass

import torch


@dataclass
class LabeledSparseTensor:
    stay_ids: list
    features: list
    sparse_data: torch.Tensor
