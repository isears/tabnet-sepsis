"""
Apply inclusion criteria to generate a list of included stay ids
"""
import pandas as pd


if __name__ == "__main__":
    all_stay_ids = pd.read_csv("mimiciv/icu/icustays.csv")
    all_stay_ids["stay_id"].to_csv("cache/included_stayids.csv", index=False)
