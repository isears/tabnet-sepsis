"""
- Annotate every stay id as "true positive", "false positive", "false negative", "true negative"
- Generate odds ratios (w/pval) of correct vs incorrect labeling for static categorical variables
- Calculate distribution and pval of correct vs incorrect labeling for static continuous variables

"""

import pandas as pd
import torch
import scipy.stats as stats
from tabsep import config
from tabsep.reporting.table1 import Table1Generator
import statsmodels.api as sm
import numpy as np


pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == "__main__":
    X_test = torch.load(f"{config.model_path}/X_test.pt")
    y_test = torch.load(f"{config.model_path}/y_test.pt")
    preds = torch.load(f"{config.model_path}/preds.pt")
    attributions = torch.load(f"{config.model_path}/attributions.pt")
    sids = pd.read_csv(f"{config.model_path}/test_stayids.csv").squeeze("columns")

    df = pd.DataFrame(
        data={"stay_id": sids.to_list(), "prediction": preds, "actual": y_test}
    )

    def annotate(row):
        if row["actual"] == 0 and row["prediction"] < 0.5:
            return "tn"  # True negative
        elif row["actual"] == 0 and row["prediction"] > 0.5:
            return "fp"  # False positive
        elif row["actual"] == 1 and row["prediction"] < 0.5:
            return "fn"  # False negative
        elif row["actual"] == 1 and row["prediction"] > 0.5:
            return "tp"  # True positive

    df["annotation"] = df.apply(annotate, axis=1)

    # Create separate tables for each group
    # print("[*] Creating tp / tn / fp / fn tables")
    # for gname, g in df.groupby("annotation"):
    #     t1generator = Table1Generator(g["stay_id"].to_list())
    #     t1 = t1generator.populate()

    #     t1.to_csv(f"results/table1_{gname}.csv", index=False)

    print("[*] Analyzing significance of correct vs incorrect groups")
    correct_t1g = Table1Generator(
        df[(df["annotation"] == "tn") | (df["annotation"] == "tp")]["stay_id"].to_list()
    )
    correct_t1g.populate()
    incorrect_t1g = Table1Generator(
        df[(df["annotation"] == "fn") | (df["annotation"] == "fp")]["stay_id"].to_list()
    )
    incorrect_t1g.populate()

    correct_df = correct_t1g.all_df
    incorrect_df = incorrect_t1g.all_df

    def pprint_continuous_var_results(name: str):
        print(f"{name}:")
        print(f"\tCorrectly-labeled mean: {correct_df[name].mean():.3f}")
        print(f"\tIncorrectly-labeled mean: {incorrect_df[name].mean():.3f}")
        _, pval = stats.ttest_ind(
            correct_df[name].to_list(), incorrect_df[name].to_list()
        )
        print(f"\tP-value: {pval:.3f}")

    pprint_continuous_var_results("age_at_intime")
    pprint_continuous_var_results("cci_score")

    def pprint_categorical_var_results(name: str):
        # Should have same groups for a given category in each df
        correct_groups = correct_df[name].value_counts().index.to_list()
        incorrect_groups = correct_df[name].value_counts().index.to_list()

        for c in correct_groups:
            assert c in incorrect_groups

        for c in incorrect_groups:
            assert c in correct_groups

        groups = correct_groups

        print(f"{name}:")
        for g in groups:
            """
            Crostab format:
                        IncorrectLabel  CorrectLabel
            Exposure
            No Exposure
            """
            crosstab = np.zeros((2, 2))
            # Incorrectly labeled, in the group
            crosstab[0, 0] = len(incorrect_df[incorrect_df[name] == g])
            # Correctly labeled, in the group
            crosstab[0, 1] = len(correct_df[correct_df[name] == g])
            # Incorrectly labeled, not in the group
            crosstab[1, 0] = len(incorrect_df[incorrect_df[name] != g])
            # Correctly labeled, in the group
            crosstab[1, 1] = len(correct_df[correct_df[name] != g])

            res = sm.stats.Table2x2(crosstab)

            print(
                f"\t{g} odds of incorrect label: {res.oddsratio:.3f} (p-value: {res.oddsratio_pvalue():.3f})"
            )

    for categorical_column in [
        "gender",
        "ethnicity",
        "language",
        "hospital_expire_flag",
    ]:
        pprint_categorical_var_results(categorical_column)

    print("[+] Done")
