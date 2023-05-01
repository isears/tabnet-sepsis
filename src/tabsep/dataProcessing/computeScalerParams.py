"""
For every feature, compute mean and std
"""

import dask.dataframe as dd
import pandas as pd

if __name__ == "__main__":

    ce = dd.read_csv(
        "mimiciv/icu/chartevents.csv",
        usecols=["itemid", "valuenum"],
        dtype={"itemid": "int", "valuenum": "float"},
    )

    scaling_params = (
        ce.groupby("itemid").agg({"valuenum": ["mean", "std", "max", "min"]}).compute()
    )
    scaling_params.columns = scaling_params.columns.droplevel()

    print(scaling_params)

    scaling_params.reset_index().to_csv("cache/scaling_params.csv")
    print("[+] Done")
