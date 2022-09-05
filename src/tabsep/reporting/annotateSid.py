"""
Annotate every stay id as "true positive", "false positive", "false negative", "true negative"
"""

import pandas as pd
import torch
from tabsep import config
from tabsep.reporting.table1 import Table1Generator


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

    for gname, g in df.groupby("annotation"):
        t1generator = Table1Generator(g["stay_id"].to_list())
        t1 = t1generator.populate()

        t1.to_csv(f"results/table1_{gname}.csv", index=False)
