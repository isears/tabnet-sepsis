import torch
from sklearn.model_selection import StratifiedKFold

from tabsep.modeling import CVResults


def cv_runner(model_factory: callable, X: torch.Tensor, y: torch.Tensor):
    skf = StratifiedKFold(n_splits=10)
    tmp_model = model_factory()
    res = CVResults(tmp_model.__class__.__name__)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        model = model_factory()

        print(
            f"[CrossValidation] Starting fold {fold_idx} for {model.__class__.__name__}"
        )

        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[test_idx])[:, 1]

        res.add_result(y[test_idx], preds)

    res.print_report()
