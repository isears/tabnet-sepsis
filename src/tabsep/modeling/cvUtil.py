import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset


# Generate datasets for cross validation folds
def cv_generator(n_splits=5):
    combined_examples = pd.concat(
        [
            pd.read_csv("cache/train_examples.csv"),
            pd.read_csv("cache/test_examples.csv"),
        ]
    )

    # Shuffle
    combined_examples = combined_examples.sample(
        len(combined_examples), random_state=42
    )

    # Shuffle again to prevent intra-fold label grouping
    cv_split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_indices, test_indices in cv_split.split(
        combined_examples, combined_examples["label"]
    ):
        train_ds = FileBasedDataset(
            combined_examples.iloc[train_indices], standard_scale=True
        )
        test_ds = FileBasedDataset(
            combined_examples.iloc[test_indices], standard_scale=True
        )
        yield train_ds, test_ds


if __name__ == "__main__":
    for fold_idx, (train, test) in enumerate(cv_generator()):
        print(fold_idx)
