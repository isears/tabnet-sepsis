import pandas as pd

from tabsep.dataProcessing.derivedDataset import DerivedDataset
from tabsep.dataProcessing.tstransform import standard_tables, zero_info_columns

features = [
    "AbsoluteBasophilCount",
    "AbsoluteEosinophilCount",
    "AbsoluteLymphocyteCount",
    "AbsoluteMonocyteCount",
    "AbsoluteNeutrophilCount",
    "AtypicalLymphocytes",
    "Bands",
    "Basophils",
    "Blasts",
    "EosinophilCount",
    "Eosinophils",
    "GranulocyteCount",
    "HypersegmentedNeutrophils",
    "ImmatureGranulocytes",
    "Lymphocytes",
    "LymphocytesPercent",
    "Metamyelocytes",
    "MonocyteCount",
    "Monocytes",
    "Myelocytes",
    "Neutrophils",
    "NucleatedRedCells",
    "OtherCells",
    "PlateletCount",
    "Promyelocytes",
    "RedBloodCells",
    "ReticulocyteCountAbsolute",
    "ReticulocyteCountAutomated",
    "WBCCount",
    "WhiteBloodCells",
    "aado2",
    "aado2_calc",
    "albumin",
    "alp",
    "alt",
    "amylase",
    "aniongap",
    "ast",
    "baseexcess",
    "bicarbonate",
    "bilirubin_direct",
    "bilirubin_indirect",
    "bilirubin_total",
    "bun",
    "calcium",
    "carboxyhemoglobin",
    "chloride",
    "ck_cpk",
    "ck_mb",
    "creatinine",
    "crp",
    "d_dimer",
    "diastolic_bp",
    "dobutamine",
    "epinephrine",
    "fibrinogen",
    "fio2",
    "ggt",
    "globulin",
    "glucose",
    "heart_rate",
    "hematocrit",
    "hemoglobin",
    "inr",
    "invasive_line",
    "lactate",
    "ld_ldh",
    "methemoglobin",
    "milrinone",
    "norepinephrine",
    "pao2fio2ratio",
    "pco2",
    "ph",
    "phenylephrine",
    "po2",
    "potassium",
    "pt",
    "ptt",
    "resp_rate",
    "so2",
    "sodium",
    "spo2",
    "systolic_bp",
    "temperature",
    "thrombin",
    "total_protein",
    "totalco2",
    "vasopressin",
    "ventilation",
]


if __name__ == "__main__":
    stats = list()

    # Standard tables
    for table_name in standard_tables + ["bg", "vitalsign"]:
        df = pd.read_parquet(f"mimiciv_derived/{table_name}.parquet")

        if "temperature" in df.columns:
            df["temperature"] = df["temperature"].astype("float")

        aggable_columns = [
            c
            for c in df.columns
            if c
            not in [
                "charttime",
                "subject_id",
                "stay_id",
                "hadm_id",
                "specimen_id",
                "specimen",
                "temperature_site",
            ]
            + zero_info_columns
        ]

        stats.append(df[aggable_columns].agg(["mean", "median", "std", "max", "min"]))

    all_stats = pd.concat(stats, axis="columns")

    # One-offs
    all_stats["fio2"] = all_stats[["fio2", "fio2_chartevents"]].mean(axis=1)
    all_stats["systolic_bp"] = all_stats[["sbp", "sbp_ni"]].mean(axis=1)
    all_stats["diastolic_bp"] = all_stats[["dbp", "dbp_ni"]].mean(axis=1)

    all_stats = all_stats.groupby(by=all_stats.columns, axis=1).mean()
    # TODO: some columns have no values ever; should stop using these entirely
    all_stats = all_stats.fillna(0.0)

    all_stats.to_parquet("processed/stats.parquet")
    all_stats = all_stats.drop(
        columns=[c for c in all_stats.columns if c not in features]
    )

    print("done")
