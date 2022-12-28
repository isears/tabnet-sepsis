"""
Associate each microbiology event (from the hosp module) with a specific icustay id
...or drop the event if it's from he wards
"""

import datetime

import numpy as np
import pandas as pd

from tabsep import config

if __name__ == "__main__":
    micro_events = pd.read_csv("mimiciv/hosp/microbiologyevents.csv", low_memory=False)
    micro_events["charttime"] = pd.to_datetime(micro_events["charttime"])

    # Drop data that we're never going to use
    # For now, just isolating blood cultures
    micro_events = micro_events[micro_events["spec_type_desc"] == "BLOOD CULTURE"]
    micro_events = micro_events[~micro_events["charttime"].isna()]
    micro_events = micro_events[~(micro_events["org_name"] == "CANCELLED")]
    # For now, just want subject_id, charttime, and what organism was detected (if any)
    micro_events = micro_events[["subject_id", "charttime", "org_name"]]

    # Match microbioevents with corresponding icustay id
    # This is non-trivial b/c microbioevents only reliably contains subject_id
    def group_icustays(group):
        return group.to_dict(orient="records")

    icustays = pd.read_csv("mimiciv/icu/icustays.csv")
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    aggregated_icustays = (
        icustays[["subject_id", "stay_id", "intime", "outtime"]]
        .groupby("subject_id")
        .apply(group_icustays)
    )

    aggregated_icustays.name = "possible_icustays"  # Required by merge

    micro_events = micro_events.merge(
        aggregated_icustays, how="left", left_on="subject_id", right_index=True
    )

    micro_events = micro_events[~micro_events["possible_icustays"].isna()]

    def get_icustay(row):
        charttime = row["charttime"]

        for possible_stay in row["possible_icustays"]:
            if (
                possible_stay["intime"] < charttime
                and possible_stay["outtime"] > charttime
            ):
                return possible_stay["stay_id"]
        else:
            return pd.NA

    micro_events["stay_id"] = micro_events.apply(get_icustay, axis=1)
    micro_events = micro_events.drop(columns=["possible_icustays"])
    micro_events = micro_events[~micro_events["stay_id"].isna()]
    micro_events = micro_events.merge(
        icustays[["stay_id", "intime", "outtime"]], how="left", on="stay_id"
    )
    micro_events["difftime"] = micro_events["charttime"] - micro_events["intime"]
    # Basically just doing inclusion criteria here now too
    micro_events = micro_events[micro_events["difftime"] > datetime.timedelta(hours=24)]
    micro_events = micro_events[micro_events["difftime"] < datetime.timedelta(days=24)]

    # Transform list of microbio events to pos / neg labels w/timestamp
    positive_micro_events = micro_events[~micro_events["org_name"].isna()]
    positive_sample = positive_micro_events.groupby("stay_id", as_index=False).apply(
        lambda x: x.sample(n=1, random_state=42)
    )
    positive_sample["label"] = 1

    negative_micro_events = micro_events[micro_events["org_name"].isna()]
    # Make sure that there's no overlap between stayids with positive examples and stayids with neg examples
    negative_micro_events = negative_micro_events[
        ~negative_micro_events["stay_id"].isin(positive_sample["stay_id"])
    ]
    negative_sample = negative_micro_events.groupby("stay_id", as_index=False).apply(
        lambda x: x.sample(n=1, random_state=42)
    )
    negative_sample["label"] = 0

    combined_sample = pd.concat([positive_sample, negative_sample])

    # Now need to compute the cut index
    assert (combined_sample["difftime"] > datetime.timedelta(hours=24)).all()
    combined_sample["cutidx"] = (
        np.floor(combined_sample["difftime"] / config.timestep)
    ).astype(int)
    assert (
        combined_sample["cutidx"] >= datetime.timedelta(hours=24) / config.timestep
    ).all()

    combined_sample[["stay_id", "label", "cutidx"]].to_csv(
        "cache/sample_cuts.csv", index=False
    )

    print("Done")
