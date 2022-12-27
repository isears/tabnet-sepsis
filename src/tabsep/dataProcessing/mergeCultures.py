"""
Associate each microbiology event (from the hosp module) with a specific icustay id
...or drop the event if it's from he wards
"""

import pandas as pd

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
        aggregated_icustays, left_on="subject_id", right_index=True
    )

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

    micro_events.to_csv("cache/usable_micro_events.csv", index=False)

    print("Done")
