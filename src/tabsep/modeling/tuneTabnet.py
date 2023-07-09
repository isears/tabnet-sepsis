import optuna
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabsep.dataProcessing import LabeledSparseTensor


def objective(trial: optuna.Trial) -> float:
    # Parameters to tune:
    trial.suggest_int("n_d", 8, 64)
    trial.suggest_int("n_steps", 3, 10)
    trial.suggest_float("gamma", 1.0, 2.0)
    trial.suggest_int("n_independent", 1, 5)
    trial.suggest_float("momentum", 0.01, 0.4, log=True)
    trial.suggest_categorical("mask_type", ["sparsemax", "entmax"])
    trial.suggest_float("optimizer_lr", 1e-5, 0.1, log=True)
    trial.suggest_int("fit_batch_size", 16, 2048, log=True)

    skf = StratifiedKFold(n_splits=3)

    d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
    X = d.get_snapshot().numpy()
    y = d.get_labels().numpy()

    # Need to sub-dict the optimizer params
    tabnet_args = {
        k: v
        for k, v in trial.params.items()
        if not k.startswith("optimizer_") and not k.startswith("fit_")
    }
    tabnet_args["n_a"] = tabnet_args["n_d"]

    optimizer_params = {
        k[len("optimizer_") :]: v
        for k, v in tabnet_args.items()
        if k.startswith("optimizer_")
    }
    tabnet_args["optimizer_params"] = optimizer_params

    fit_params = {
        d[len("fit_") :]: v for k, v in tabnet_args.items() if k.startswith("fit_")
    }

    cv_scores = list()

    try:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            model = TabNetClassifier(**tabnet_args)

            print(
                f"[CrossValidation] Starting fold {fold_idx} for {model.__class__.__name__}"
            )

            X_train, X_valid, y_train, y_valid = train_test_split(
                X[train_idx], y[train_idx], test_size=0.1
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                patience=3,
                eval_metric=["logloss"],
                **fit_params,
            )
            preds = model.predict(X[test_idx])

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
    study.optimize(objective, n_trials=10000)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
