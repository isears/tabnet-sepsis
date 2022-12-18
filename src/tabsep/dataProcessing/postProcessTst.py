import datetime
from typing import Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from tabsep import config
from tabsep.dataProcessing.util import all_inclusive_dtypes

pd.options.mode.chained_assignment = None  # default='warn'


def get_mimicts_timestep():
    with open("cache/mimicts/readme.txt", "r") as f:
        for line in f.readlines():
            if "timestep=" in line:
                return datetime.timedelta(seconds=int(line.split("=")[-1]))


if __name__ == "__main__":
    stay_ids = pd.read_csv("cache/included_stayids.csv").squeeze("columns")
    icustays = pd.read_csv("mimiciv/icu/icustays.csv")

    ce = dd.read_csv("mimiciv/icu/chartevents.csv", dtype=all_inclusive_dtypes)
    ptt_event_id = 227466

    print("[*] Running initial filter")
    ProgressBar().register()
    ptts_only = ce[ce["itemid"] == ptt_event_id].compute(scheduler="processes")

    ptts_only = ptts_only[ptts_only["stay_id"].isin(stay_ids)]

    ptts_only = ptts_only.merge(
        icustays[["stay_id", "intime"]], how="left", on="stay_id"
    )

    # Drop anything that's too close to intime
    for time_column in ["charttime", "intime"]:
        ptts_only[time_column] = pd.to_datetime(ptts_only[time_column])

    ptts_only = ptts_only[
        ptts_only["charttime"] > (ptts_only["intime"] + datetime.timedelta(hours=24))
    ]

    # Compute where the dataset should truncate the data for a specific value
    ptts_only["difftime"] = ptts_only["charttime"] - ptts_only["intime"]
    ptts_only["cutidx"] = np.ceil(
        ptts_only["difftime"] / get_mimicts_timestep()
    ).astype("int")

    ptts_only = ptts_only.rename(columns={"valuenum": "label"})

    ptts_only[["stay_id", "cutidx", "label"]].to_csv(
        "cache/sample_cuts.csv", index=False
    )

    print(f"Mean PTT: {ptts_only['label'].mean()}")
    print(f"Total count: {len(ptts_only)}")

    print("[+] Done")
