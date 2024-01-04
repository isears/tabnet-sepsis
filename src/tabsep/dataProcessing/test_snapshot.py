import sys

import numpy as np
from tqdm import tqdm

from tabsep.dataProcessing import LabeledSparseTensor

if __name__ == "__main__":
    path = "cache/sparse_labeled_sav.pkl"

    if len(sys.argv) == 2:
        path = sys.argv[1]

    print(f"[*] Loading data from {path}")
    lst = LabeledSparseTensor.load_from_pickle(path)
    X_ts = lst.get_dense_normalized()
    X_snapshot = lst.get_snapshot()

    np.random.seed(42)

    number_to_test = 100000
    example_indices = np.random.choice(
        range(0, X_ts.shape[0]), size=number_to_test, replace=True
    )
    feature_indices = np.random.choice(
        range(0, X_ts.shape[-1]), size=number_to_test, replace=True
    )

    for example_idx, feature_idx in tqdm(list(zip(example_indices, feature_indices))):
        sample_ts = X_ts[example_idx, :, feature_idx]
        sample_snapshot = X_snapshot[example_idx, feature_idx]

        # Get last valid element in ts the stupid way
        for idx in reversed(range(0, len(sample_ts))):
            if sample_ts[idx] != -1:
                assert (
                    sample_ts[idx].item() == sample_snapshot.item()
                ), f"[-] Failed to verify at sample idx {example_idx} and feature idx {feature_idx} ({sample_ts[idx].item()} != {sample_snapshot.item()})"
                break

    print("[+] Tests passed")
