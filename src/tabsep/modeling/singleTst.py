"""
Run TST w/specific hyperparameters
"""

from sklearn.metrics import average_precision_score, roc_auc_score

from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.dataProcessing.labelGeneratingDataset import CoagulopathyDataset
from tabsep.modeling.tstTuning import split_data_consistently, tunable_tst_factory

PARAMS = dict(
    optimizer__lr=1e-4,
    module__dropout=0.1,
    d_model_multiplier=8,
    module__num_layers=3,
    module__n_heads=16,
    module__dim_feedforward=256,
    iterator_train__batch_size=16,  # Should be 128
)

if __name__ == "__main__":
    train_sids, test_sids = split_data_consistently()

    train_ds = CoagulopathyDataset(train_sids)
    test_ds = CoagulopathyDataset(test_sids)

    tst = tunable_tst_factory(PARAMS, train_ds, save_path="cache/models/singleTst")

    tst.fit(train_ds, y=None)

    # TODO: dont' have get_labels() anymore need to update this
    # final_auroc = roc_auc_score(test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1])

    # final_auprc = average_precision_score(
    #     test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1]
    # )

    # print("Final score:")
    # print(f"\tAUROC: {final_auroc}")
    # print(f"\tAverage precision: {final_auprc}")
