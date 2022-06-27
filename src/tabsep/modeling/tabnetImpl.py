from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tabsep.dataProcessing.tabularDataset import TabularDataset

if __name__ == "__main__":

    just_cols = pd.read_csv("cache/combined_tailored.csv", nrows=1).columns
    data_types = {col: "float" for col in just_cols}
    data_types["stay_id"] = "int"
    combined_data = pd.read_csv("cache/combined_tailored.csv", low_memory=False)
    combined_data = combined_data.set_index("stay_id")
    X = combined_data[[col for col in combined_data.columns if col != "label"]].values
    y = combined_data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    print(f"Training size: {X_train.shape}")
    print(f"Validation size: {X_test.shape}")
    clf = TabNetClassifier()
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    print(clf)
