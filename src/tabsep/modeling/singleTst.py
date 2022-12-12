"""
Run TST w/specific hyperparameters
"""

from sklearn.metrics import average_precision_score, roc_auc_score

from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling.tstTuning import split_data_consistently, tunable_tst_factory

PARAMS = dict(
    optimizer__lr=0.002656205687468148,
    module__dropout=0.13910274697124775,
    d_model_multiplier=4,
    module__num_layers=1,
    module__n_heads=8,
    module__dim_feedforward=268,
    iterator_train__batch_size=83,
)

if __name__ == "__main__":
    sids_train, sids_test = split_data_consistently()

    train_ds = FileBasedDataset(sids_train)
    test_ds = FileBasedDataset(sids_test)

    tst = tunable_tst_factory(PARAMS, train_ds)

    tst.fit(train_ds, train_ds.get_labels())

    final_auroc = roc_auc_score(test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1])

    final_auprc = average_precision_score(
        test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1]
    )

    print("Final score:")
    print(f"\tAUROC: {final_auroc}")
    print(f"\tAverage precision: {final_auprc}")
