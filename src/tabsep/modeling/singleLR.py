import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling.tstTuning import split_data_consistently


def load_to_mem(sids: list):
    all_X, all_y = torch.tensor([]), torch.tensor([])

    train_ds = FileBasedDataset(sids)
    memory_loader = DataLoader(
        train_ds,
        batch_size=16,  # Batch size only important for tuning # workers to load to mem
        num_workers=config.cores_available,
        collate_fn=train_ds.last_nonzero_collate,
        drop_last=False,
    )

    for batch_X, batch_y in memory_loader:
        all_X = torch.cat((all_X, batch_X))
        all_y = torch.cat((all_y, batch_y))

    return all_X, all_y


if __name__ == "__main__":

    sids_train, sids_test = split_data_consistently()
    train_X, train_y = load_to_mem(sids_train)
    print("[*] Training data loaded data to memory")

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(train_X, train_y)

    test_X, test_y = load_to_mem(sids_test)
    print("[*] Testing data loaded to memory")

    final_auroc = roc_auc_score(test_y, lr.predict_proba(test_X)[:, 1])
    final_auprc = average_precision_score(test_y, lr.predict_proba(test_X)[:, 1])

    print("Final score:")
    print(f"\tAUROC: {final_auroc}")
    print(f"\tAverage precision: {final_auprc}")
