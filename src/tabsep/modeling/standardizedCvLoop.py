from matplotlib.pyplot import vlines
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from tabsep.dataProcessing.loadAllData import load_from_disk
import sys


class PretrainingTabNetClf(TabNetClassifier):
    def __init__(self):
        return super().__init__()

    def fit(self, X, y):
        print("Running unsupervised pretraining...")
        unsupervised = TabNetPretrainer()
        unsupervised.fit(X)
        print("Pretraining complete, moving onto supervised training")
        super().fit(X, y, from_unsupervised=unsupervised)


class ValidatingTabNetClf(TabNetClassifier):
    def __init__(self):
        return super().__init__()

    def fit(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.10, random_state=42
        )

        super().fit(X_train, y_train, eval_set=[(X_valid, y_valid)])


if __name__ == "__main__":
    models = {
        "tabnet": ValidatingTabNetClf(),
        "lr": LogisticRegression(max_iter=1e9),
        "rf": RandomForestClassifier(),
    }

    model_name = sys.argv[1]

    if model_name in models:
        clf = models[model_name]
    else:
        raise ValueError(f"No model named {sys.argv[1]}. Pick from {models.keys()}")

    combined_data = load_from_disk()

    X = combined_data[[col for col in combined_data.columns if col != "label"]].values
    y = combined_data["label"].values

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv_splitter, scoring="roc_auc", n_jobs=-1)
    print(scores)
    print(scores.mean())
