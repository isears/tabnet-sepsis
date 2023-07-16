import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from tabsep.dataProcessing import LabeledSparseTensor

if __name__ == "__main__":
    path = "cache/sparse_labeled_6.pkl"

    if len(sys.argv) == 2:
        path = sys.argv[1]

    print(f"[*] Loading data from {path}")
    lst = LabeledSparseTensor.load_from_pickle(path)

    sepsis_df = pd.read_parquet("mimiciv_derived/sepsis3.parquet").set_index("stay_id")

    for idx, stay_id in enumerate(lst.stay_ids):
        if lst.y[idx] == 1:
            assert (
                int(stay_id) in sepsis_df.index
            ), f"{stay_id} septic in dataset but not found in sepsis3"
        elif lst.y[idx] == 0:
            assert (
                int(stay_id) not in sepsis_df.index
            ), f"{stay_id} non-septic in dataset but found in sepsis3"
        else:
            assert True, f"Invalid y: {lst.y[idx]}"

    print("[+] Tests passed")
