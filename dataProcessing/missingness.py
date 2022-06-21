import dask.dataframe as dd
import pandas as pd


def generate_missingness(threshold: float) -> pd.Series:
    events_dd = dd.read_csv(
        "mimiciv/icu/chartevents.csv",
        assume_missing=True,
        blocksize=100e6,
        parse_dates=["charttime", "storetime"],
        dtype={
            "subject_id": "int",
            "hadm_id": "int",
            "stay_id": "int",
            "charttime": "object",
            "storetime": "object",
            "itemid": "int",
            "value": "object",
            "valueuom": "object",
            "warning": "object",
            "valuenum": "float",
        },
    )

    total_hadms = events_dd["hadm_id"].nunique()
    hadm_counts_by_itemid = (
        events_dd.groupby("itemid")["hadm_id"].nunique().compute(scheduler="processes")
    )

    hadm_counts_by_itemid = hadm_counts_by_itemid.reset_index()
    hadm_counts_by_itemid = hadm_counts_by_itemid.rename(
        columns={"hadm_id": "hadm_count"}
    )
    included_itemids = hadm_counts_by_itemid[
        (hadm_counts_by_itemid["hadm_count"] / total_hadms) > threshold
    ]

    print(
        f"Dropped {len(hadm_counts_by_itemid) - len(included_itemids)} features that did not meet the missingness threshold ({threshold})"
    )
    included_itemids["itemid"].to_csv("cache/included_features.csv", index=False)


if __name__ == "__main__":
    generate_missingness(0.1)
