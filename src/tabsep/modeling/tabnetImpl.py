from pyparsing import col
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tabsep.dataProcessing.loadAllData import load_from_disk

if __name__ == "__main__":
    combined_data = load_from_disk()

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
