from __future__ import annotations

import pickle
from dataclasses import dataclass

import torch

from tabsep.modeling.skorchTST import AutoPadmaskingTST


@dataclass
class LabeledSparseTensor:
    stay_ids: list
    features: list
    X_sparse: torch.Tensor
    y: torch.Tensor

    @classmethod
    def load_from_pickle(cls, path: str) -> LabeledSparseTensor:
        with open(path, "rb") as f:
            sparse_data = pickle.load(f)

        return sparse_data

    def get_dense_standardized(self):
        """
        (x - mu) / sigma
        filling 0.0 where no data
        """
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

        X_dense = torch.where(X_dense != 0.0, (X_dense - mu) / std, 0.0)

        # shape (n_examples, seq_len, feat_dim)
        return X_dense.permute(0, 2, 1).float()

    def get_dense_normalized(self):
        """
        Min / max normalization filling -1s where no data
        """

        # Need to manually densify to preserve nans
        X_dense = torch.full(self.X_sparse.shape, float("nan"))
        indices = self.X_sparse.coalesce().indices()
        values = self.X_sparse.coalesce().values()
        X_dense[indices[0, :], indices[1, :], indices[2, :]] = values.float()

        nanmask = torch.isnan(X_dense)
        # TODO: this process will "delete" variables that only take on one value (e.g. meds that are only ever 1)
        # Right now this is the reason we can't use the antibiotics table
        X_dense[nanmask] = float("-inf")
        maxvals = torch.amax(X_dense, dim=(0, -1)).unsqueeze(0).unsqueeze(-1)
        X_dense[nanmask] = float("inf")
        minvals = torch.amin(X_dense, dim=(0, -1)).unsqueeze(0).unsqueeze(-1)
        X_dense[nanmask] = float("nan")

        intervals = maxvals - minvals

        X_dense = (X_dense - minvals) / intervals
        X_dense = torch.nan_to_num(X_dense, -1.0)

        return X_dense.permute(0, 2, 1).float()

    def get_snapshot(self):
        X_dense = self.get_dense_normalized()

        # TODO: could write tests to verify this
        max_valid_idx = (X_dense.shape[1] - 1) - (
            torch.argmax(
                ((torch.flip(X_dense, dims=(1,))) != -1).int(), dim=1, keepdim=True
            )
        )

        X_most_recent = torch.gather(X_dense, dim=1, index=max_valid_idx).squeeze()

        return X_most_recent

    def get_snapshot_los(self):
        X_dense = self.get_dense_normalized()
        # Divide by 5 b/c max LOS is 5 days
        los = (AutoPadmaskingTST.autopadmask(X_dense).sum(dim=1) - 1) / (5.0 * 24)

        # TODO: could write tests to verify this
        max_valid_idx = (X_dense.shape[1] - 1) - (
            torch.argmax(
                ((torch.flip(X_dense, dims=(1,))) != -1).int(), dim=1, keepdim=True
            )
        )

        X_most_recent = torch.gather(X_dense, dim=1, index=max_valid_idx).squeeze()
        X_with_los = torch.concat([X_most_recent, los.unsqueeze(-1)], dim=1)

        return X_with_los

    def get_labels(self):
        return self.y.float()
