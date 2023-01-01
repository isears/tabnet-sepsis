import random
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

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

    def do_random_cut(stay_id: int) -> int:
        ce_df = pd.read_csv(
            f"mimicts/{stay_id}/chartevents_features.csv",
            nrows=1,
            index_col="feature_id",
        )

        stay_len = len(ce_df.columns)
        assert stay_len > 24  # Should be guaranteed by inclusion criteria

        return random.randrange(12, stay_len)

    nonseptic_sample["cutidx"] = nonseptic_sample["stay_id"].apply(
        lambda s: do_random_cut(s)
    )
    nonseptic_sample["label"] = 0.0

    print(f"[+] Done, saving to disk")
    final_df = pd.concat([nonseptic_sample, pos_septic_sample])[
        ["stay_id", "cutidx", "label"]
    ]

    print("Cut distribution test:")
    print(f"\tPositives: {final_df[final_df['label'] == 1]['cutidx'].median()}")
    print(f"\tNegatives: {final_df[final_df['label'] == 0]['cutidx'].median()}")
    train_df, test_df = train_test_split(final_df, test_size=0.1, random_state=42)
    train_df.to_csv("cache/train_examples.csv", index=False)
    test_df.to_csv("cache/test_examples.csv", index=False)

    # Pretraining should include everyone except the test set
    print("[*] Generating pre-trainable dataset")
    icustays = pd.read_csv("mimiciv/icu/icustays.csv")
    icustays = icustays[~icustays["stay_id"].isin(test_df["stay_id"])]

    def random_cut_mark_if_invalid(stay_id: int):
        try:
            return do_random_cut(stay_id)
        except AssertionError:
            return -1

    pretrain_df = icustays[["stay_id"]]
    pretrain_df["cutidx"] = pretrain_df["stay_id"].apply(random_cut_mark_if_invalid)
    pretrain_df = pretrain_df[pretrain_df["cutidx"] > 0]
    # Prevent pretraining from taking up significantly more memory than training
    pretrain_df = pretrain_df[pretrain_df["cutidx"] < train_df["cutidx"].max()]
    print(f"Pretraining examples: {len(pretrain_df)}")
    pretrain_df.to_csv("cache/pretrain_examples.csv", index=False)
