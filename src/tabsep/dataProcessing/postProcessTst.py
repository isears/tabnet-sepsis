import pandas as pd


if __name__ == "__main__":
    icustays = pd.read_csv("mimiciv/icu/icustays.csv")
    admissions = pd.read_csv("mimiciv/core/admissions.csv")
    inclusion_sids = (
        pd.read_csv("cache/included_stayids.csv").squeeze("columns").to_list()
    )

    icustays = icustays[icustays["stay_id"].isin(inclusion_sids)]
    # TODO: this is a little inaccurate b/c it assumes all admissions only had one ICU stay
    # Either have to exclude multiple ICU stays or identify admissions w/multiple ICU stays and
    # assign the mortality label to the final ICU stay
    icustays = icustays.merge(
        admissions[["hadm_id", "hospital_expire_flag"]], how="left", on="hadm_id"
    )

    icustays = icustays.rename(columns={"hospital_expire_flag": "label"})
    icustays[["stay_id", "label"]].to_csv("cache/ihm_sample.csv", index=False)
