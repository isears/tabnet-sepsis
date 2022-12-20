"""
Run TST w/specific hyperparameters
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


class TimeSeriesScaler(StandardScaler):
    """
    Scale the features one at a time
    Assumes shape of data is (n_samples, seq_len, n_features)
    """

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None, sample_weight=None):
        self.feature_scalers = list()

        for feature_idx in range(0, X.shape[-1]):  # Assuming feature_dim is last dim
            feature_scaler = StandardScaler(
                copy=self.copy, with_mean=self.with_mean, with_std=self.with_std
            )
            feature_scaler.fit(X[:, :, feature_idx])
            self.feature_scalers.append(feature_scaler)

        return self

    def transform(self, X, copy=None):
        return np.stack(
            [f.transform(X[:, :, idx]) for idx, f in enumerate(self.feature_scalers)],
            axis=-1,
        )

    def inverse_transform(self, X, copy=None):
        return np.stack(
            [
                f.reverse_tranform(X[:, :, idx])
                for idx, f in enumerate(self.feature_scalers)
            ],
            axis=1,
        )


if __name__ == "__main__":
    train_sids, test_sids = split_data_consistently()

    train_ds = CoagulopathyDataset(train_sids)
    test_ds = CoagulopathyDataset(test_sids)

    tst = tunable_tst_factory(PARAMS, train_ds, save_path="cache/models/singleTst")
    scaled_tst = make_pipeline([TimeSeriesScaler(), tst])

    scaled_tst.fit(train_ds, y=None)

    # TODO: dont' have get_labels() anymore need to update this
    # final_auroc = roc_auc_score(test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1])

    # final_auprc = average_precision_score(
    #     test_ds.get_labels(), tst.predict_proba(test_ds)[:, 1]
    # )

    # print("Final score:")
    # print(f"\tAUROC: {final_auroc}")
    # print(f"\tAverage precision: {final_auprc}")
