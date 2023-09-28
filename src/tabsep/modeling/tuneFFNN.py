import sys

import optuna
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import TSTConfig, my_auprc, my_auroc, my_f1
from tabsep.modeling.skorchNN import nn_factory


def objective(trial: optuna.Trial, X, y) -> float:
    # Parameters to tune:
    trial.suggest_float("lr", 1e-5, 0.1, log=True)
    trial.suggest_int("n_hidden", 1, 15)
    trial.suggest_int("width", 5, 100)
    # trial.suggest_int("batch_size", 8, 256)
    trial.suggest_categorical("activation_fn", ["gelu", "relu"])
    trial.suggest_float("dropout", 0.01, 0.5)

    skf = StratifiedKFold(n_splits=3)

    cv_scores = list()

    try:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            model = nn_factory(**trial.params, patience=3)

            print(
                f"[CrossValidation] Starting fold {fold_idx} for {model.__class__.__name__}"
            )

            model.fit(X[train_idx], y[train_idx])
            preds = model.predict_proba(X[test_idx])[:, 1]

            cv_scores.append(average_precision_score(y[test_idx], preds))
    except RuntimeError as e:
        print(f"Warning, assumed memory error: {e}")
        del model
        torch.cuda.empty_cache()
        # return float("nan")
        return 0.0

    return sum(cv_scores) / len(cv_scores)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    if len(sys.argv) < 2:
        datapath = "cache/sparse_labeled_12.pkl"
    else:
        datapath = sys.argv[1]

    d = LabeledSparseTensor.load_from_pickle(datapath)
    X = d.get_snapshot_los()
    y = d.get_labels()

    # Saving a hold-out set for testing
    X_tune, X_test, y_tune, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    study.optimize(lambda trial: objective(trial, X_tune, y_tune), n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
