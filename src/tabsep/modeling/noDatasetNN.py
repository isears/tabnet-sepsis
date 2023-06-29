from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from tabsep.dataProcessing import LabeledSparseTensor

if __name__ == "__main__":
    d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
    X = d.get_snapshot()
    y = d.get_labels()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    nn = MLPClassifier(
        solver="lbfgs",
        alpha=1e-5,
        hidden_layer_sizes=(5, 2),
        random_state=1,
        max_iter=10000,
    )

    nn.fit(X_train, y_train)

    preds = nn.predict_proba(X_test)[:, 1]

    print(roc_auc_score(y_test, preds))

    print("Done")
