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

    def get_dense_normalized(self):
        X_dense = self.X_sparse.to_dense()

        n = torch.count_nonzero(X_dense, dim=(0, -1))
        s = torch.sum(X_dense, dim=(0, -1))
        mu = s / n

        mu = mu.unsqueeze(0).unsqueeze(-1)
        n = n.unsqueeze(0).unsqueeze(-1)

        mu_diff_2 = torch.where(
            X_dense != 0.0,
            (X_dense - mu) ** 2,
            0.0,
        )
        std = torch.sqrt(
            torch.sum(mu_diff_2, dim=(0, -1)).unsqueeze(0).unsqueeze(-1) / (n - 1)
        )

        return torch.where(X_dense != 0.0, (X_dense - mu) / std, 0.0)
