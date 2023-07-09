import optuna
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import TSTConfig, my_auprc, my_auroc, my_f1
from tabsep.modeling.skorchTST import AutoPadmaskingTST, tst_factory


def objective(trial: optuna.Trial) -> float:
    # Parameters to tune:
    trial.suggest_float("lr", 1e-5, 0.1, log=True)
    trial.suggest_float("dropout", 0.1, 0.7)
    trial.suggest_categorical("d_model_multiplier", [1, 2, 4, 8, 16, 32, 64])
    trial.suggest_int("num_layers", 1, 15)
    trial.suggest_categorical("n_heads", [4, 8, 16, 32, 64])
    trial.suggest_int("dim_feedforward", 128, 512)
    trial.suggest_int("batch_size", 8, 256)
    trial.suggest_categorical("pos_encoding", ["fixed", "learnable"])
    trial.suggest_categorical("activation", ["gelu", "relu"])
    trial.suggest_categorical("norm", ["BatchNorm", "LayerNorm"])
    trial.suggest_categorical("weight_decay", [1e-3, 1e-2, 1e-1, 0])

    skf = StratifiedKFold(n_splits=3)
    tst_config = TSTConfig(
        save_path="cache/models/skorchCvTst", optimizer_name="AdamW", **trial.params
    )

    d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
    X = d.get_dense_normalized()
    y = d.get_labels()

    cv_scores = list()

    try:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            model = tst_factory(tst_config)

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
    study.optimize(objective, n_trials=10000)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
