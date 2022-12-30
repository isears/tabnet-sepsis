"""
Run TST w/specific hyperparameters
"""

from sklearn.metrics import average_precision_score, roc_auc_score

from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import (
    TSTCombinedConfig,
    TSTModelConfig,
    TSTRunConfig,
    tst_skorch_factory,
)
from tabsep.modeling.tstTuning import split_data_consistently

PARAMS = dict(
    optimizer__lr=1e-4,
    module__dropout=0.1,
    d_model_multiplier=8,
    module__num_layers=3,
    module__n_heads=16,
    module__dim_feedforward=256,
    iterator_train__batch_size=128,  # Should be 128
)

if __name__ == "__main__":
    sids_train, sids_test = split_data_consistently()

    train_ds = FileBasedDataset(sids_train)
    test_ds = FileBasedDataset(sids_test)

    model_config = TSTModelConfig()
    run_config = TSTRunConfig()
    tst_config = TSTCombinedConfig(
        save_path="cache/models/singleTst",
        model_config=model_config,
        run_config=run_config,
    )

    tst = tst_skorch_factory(tst_config, train_ds)

    tst.fit(train_ds, train_ds.get_labels())

    final_auroc = roc_auc_score(test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1])

    final_auprc = average_precision_score(
        test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1]
    )

    print("Final score:")
    print(f"\tAUROC: {final_auroc}")
    print(f"\tAverage precision: {final_auprc}")
