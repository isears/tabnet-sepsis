import os
import random
import sys
from typing import Tuple

import pandas as pd

from tabsep import config

pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "balanced":
            balanced = True
        elif sys.argv[1] == "unbalanced":
            balanced = False
    else:
        balanced = False

    random.seed(42)
    icustays = pd.read_csv("mimiciv/icu/icustays.csv")
    sepsis3 = pd.read_csv("mimiciv/derived/sepsis3.csv")

    icustays = icustays.merge(sepsis3[["stay_id", "sepsis3"]], how="left", on="stay_id")
    icustays["sepsis3"] = icustays["sepsis3"].fillna(False)

    inclusion_sids = (
        pd.read_csv("cache/included_stayids.csv").squeeze("columns").to_list()
    )

    icustays = icustays[icustays["stay_id"].isin(inclusion_sids)]

    septic_icustays = icustays[icustays["sepsis3"]]
    nonseptic_icustays = icustays[~icustays["sepsis3"]]

    assert len(septic_icustays) + len(nonseptic_icustays) == len(inclusion_sids)

    print("[*] Postprocessing started with:")
    print(f"\t{len(septic_icustays)} septic stays")
    print(f"\t{len(nonseptic_icustays)} non-septic stays")
    print(f"\t{len(inclusion_sids)} total stays in inclusion criteria")

    if balanced:
        print(
            f"[*] Sampling {len(septic_icustays)} non-septic stays for balanced data..."
        )

        # Sample from the longest icu stays so we have enough data
        nonseptic_icustays = nonseptic_icustays[
            nonseptic_icustays["los"] > nonseptic_icustays["los"].mean()
        ]

        nonseptic_sample = nonseptic_icustays.sample(
            len(septic_icustays), random_state=42
        )
    else:
        nonseptic_sample = nonseptic_icustays

    print(f"[*] Applying just-before-sepsis cut to septic sample")

    def do_sepsis_pos_cut(stay_id: int) -> int:
        sepsis_df = pd.read_csv(
            f"mimicts/{stay_id}/sepsis3_features.csv", index_col="feature_id"
        )
        row = sepsis_df.iloc[0]
        sepsis_idx = int(row[row == 1].index[0])
        # This should be guaranteed by inclusion criteria
        assert sepsis_idx >= 24
        # Predict ahead by a certain window
        cutidx = sepsis_idx - config.prediction_timesteps

        assert cutidx >= 2

        return cutidx

    pos_septic_sample = septic_icustays[["stay_id"]]
    pos_septic_sample["cutidx"] = pos_septic_sample["stay_id"].apply(do_sepsis_pos_cut)
    pos_septic_sample["label"] = 1.0

    print(f"[*] Applying random cut to non-septic sample")
    pos_cut_distribution = pos_septic_sample["cutidx"].to_list()
    pos_cut_oversample = random.choices(pos_cut_distribution, k=len(nonseptic_icustays))

    class DistributionCutter:  # Building class so that we can maintain state
        def __init__(self, sample) -> None:
            self.sample = sample
            self.idx = 0
            self.total_length = len(sample)

        def do_dist_cut(self, stay_id: int) -> int:
            """
            Attempts to create a distribution similar to self.sample
            Will fail if not possible
            """
            ce_df = pd.read_csv(
                f"mimicts/{stay_id}/chartevents_features.csv",
                nrows=1,
                index_col="feature_id",
            )

            stay_len = len(ce_df.columns)

            # Get the longest stays out of the way first, when possible
            valid_cuts = [c for c in self.sample if c <= (stay_len)]
            assert (
                len(valid_cuts) > 0
            ), f"[-] Couldn't find a valid cut idx for stay length {stay_len} ({self.idx} / {self.total_length})\n{self.sample}"
            cutidx = max(valid_cuts)
            self.sample.remove(cutidx)

            self.idx += 1
            return cutidx

        def do_random_cut(self, stay_id: int) -> int:
            ce_df = pd.read_csv(
                f"mimicts/{stay_id}/chartevents_features.csv",
                nrows=1,
                index_col="feature_id",
            )

            stay_len = len(ce_df.columns)
            assert stay_len > 24  # Should be guaranteed by inclusion criteria

            return random.randrange(12, stay_len)

    dc = DistributionCutter(pos_cut_oversample)

    nonseptic_sample["cutidx"] = nonseptic_sample["stay_id"].apply(
        lambda s: dc.do_random_cut(s)
    )
    nonseptic_sample["label"] = 0.0

    print(f"[+] Done, saving to disk")
    final_df = pd.concat([nonseptic_sample, pos_septic_sample])[
        ["stay_id", "cutidx", "label"]
    ]

    print("Cut distribution test:")
    print(f"\tPositives: {final_df[final_df['label'] == 1]['cutidx'].median()}")
    print(f"\tNegatives: {final_df[final_df['label'] == 0]['cutidx'].median()}")
    final_df.to_csv("cache/sample_cuts.csv", index=False)
