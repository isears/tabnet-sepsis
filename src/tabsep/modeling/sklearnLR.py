from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabsep.dataProcessing import LabeledSparseTensor

if __name__ == "__main__":
    d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
    X = d.get_snapshot()
    y = d.get_labels()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lr = LogisticRegression(max_iter=10000)

    lr.fit(X_train, y_train)

    preds = lr.predict_proba(X_test)[:, 1]

    print(roc_auc_score(y_test, preds))

    print("Done")
