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
        icustays[["stay_id", "intime", "outtime"]], how="left", on="stay_id"
    )

    # Drop anything that's too close to intime
    for time_column in ["charttime", "intime", "outtime"]:
        ptts_only[time_column] = pd.to_datetime(ptts_only[time_column])

    ptts_only = ptts_only[
        ptts_only["charttime"] > (ptts_only["intime"] + datetime.timedelta(hours=24))
    ]

    # Also drop anything that's past outtime (this happens sometimes for some reason)
    ptts_only = ptts_only[ptts_only["charttime"] < ptts_only["outtime"]]

    # Compute where the dataset should truncate the data for a specific value
    ptts_only["difftime"] = ptts_only["charttime"] - ptts_only["intime"]
    ptts_only["cutidx"] = np.floor(  # Earliest possible: TODO: off-by-one errors
        ptts_only["difftime"] / get_mimicts_timestep()
    ).astype("int")

    # ptts_only = ptts_only.rename(columns={"valuenum": "label"})
    ptts_only["label"] = (ptts_only["valuenum"] > 35).astype("int")

    ptts_only[["stay_id", "cutidx", "label"]].to_csv(
        "cache/sample_cuts.csv", index=False
    )

    pos_count = len(ptts_only[ptts_only["label"] == 1])
    print(f"Pos count: {pos_count} ({pos_count / len(ptts_only) * 100} %)")
    print(f"Total count: {len(ptts_only)}")
    print(f"Max cut: {ptts_only['cutidx'].max()}")

    print("[+] Done")
