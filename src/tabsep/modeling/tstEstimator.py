import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from tabsep import config
from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset
from tabsep.modeling import EarlyStopping
from tabsep.modeling.tstImpl import AdamW, TSTransformerEncoderClassiregressor


class TstWrapper(BaseEstimator, ClassifierMixin):
    # TODO: max_len must be set dynamically based on cache metadata
    def __init__(
        # From TST paper: hyperparameters that perform generally well
        self,
        # Fit params
        max_epochs=7,  # This is not specified by paper, depends on dataset size
        batch_size=128,  # Should be 128, but gpu can't handle it
        optimizer_cls=AdamW,
        # TST params
        d_model=128,
        dim_feedforward=256,
        max_len=120,
        n_heads=16,
        num_classes=1,
        num_layers=3,
        use_early_stopping=False,
    ) -> None:

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        self.n_heads = n_heads
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_early_stopping = use_early_stopping

        self.loader_args = dict(
            num_workers=config.cores_available,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def fit(self, stay_ids, y=None):
        # scikit boilerplate
        self.classes_ = np.array([0.0, 1.0])
        # original_y_shape = y.shape
        # self.classes_, y = np.unique(y, return_inverse=True)
        # y = torch.reshape(torch.tensor(y), original_y_shape).float()  # re-torch y
        es = EarlyStopping(patience=5)

        if self.use_early_stopping:
            X_train, X_valid = train_test_split(
                stay_ids, test_size=0.1, random_state=42
            )

            train_ds = FileBasedDataset(X_train)
            valid_ds = FileBasedDataset(X_valid)

            train_dl = torch.utils.data.DataLoader(
                train_ds, collate_fn=train_ds.maxlen_padmask_collate, **self.loader_args
            )

            valid_dl = torch.utils.data.DataLoader(
                valid_ds, collate_fn=valid_ds.maxlen_padmask_collate, **self.loader_args
            )

        else:
            train_ds = FileBasedDataset(stay_ids)
            train_dl = torch.utils.data.DataLoader(
                train_ds, collate_fn=train_ds.maxlen_padmask_collate, **self.loader_args
            )

        model = TSTransformerEncoderClassiregressor(
            feat_dim=train_ds.get_num_features(),
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            max_len=self.max_len,
            n_heads=self.n_heads,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
        ).to("cuda")

        optimizer = self.optimizer_cls(model.parameters())
        criterion = torch.nn.BCELoss()

        for epoch_idx in range(0, self.max_epochs):
            model.train()
            optimizer.zero_grad()

            for batch_idx, (batch_X, batch_y, batch_pm) in enumerate(train_dl):
                outputs = model.forward(batch_X.to("cuda"), batch_pm.to("cuda"))
                loss = criterion(outputs, batch_y.to("cuda"))
                loss.backward()
                # TODO: what's the significance of this?
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()

            # Do early stopping
            if self.use_early_stopping:
                model.eval()
                y_pred = torch.Tensor().to("cuda")
                y_actual = torch.Tensor().to("cuda")

                for bXv, byv, pmv in valid_dl:
                    this_y_pred = model(bXv.to("cuda"), pmv.to("cuda"))
                    y_pred = torch.cat((y_pred, this_y_pred))
                    y_actual = torch.cat((y_actual, byv.to("cuda")))

                validation_loss = log_loss(
                    y_actual.detach().to("cpu"), y_pred.detach().to("cpu")
                )

                es(validation_loss, model.state_dict())

                if es.early_stop:
                    print(f"Stopped training @ epoch {epoch_idx}")
                    break

        if es.saved_best_weights:
            model.load_state_dict(es.saved_best_weights)

        self.model = model

        return self

    # This seems to be the function used by scikit cv_loop, which is all we really care about right now
    def decision_function(self, stay_ids):
        test_ds = FileBasedDataset(stay_ids)
        test_dl = torch.utils.data.DataLoader(
            test_ds, collate_fn=test_ds.maxlen_padmask_collate, **self.loader_args
        )

        with torch.no_grad():
            all_batch_y = list()

            for batch_idx, (batch_X, batch_y, batch_pm) in enumerate(test_dl):
                all_batch_y.append(self.model(batch_X.to("cuda"), batch_pm.to("cuda")))

            # send model to cpu at end so that it's not taking up GPU space
            print("[*] Fold done, sending model to CPU")
            self.model.to("cpu")

        y_pred = torch.cat(all_batch_y)

        return y_pred.to("cpu")  # sklearn needs to do cpu ops

    def predict(self, X):
        return self.decision_function(X)

    def predict_proba(self, X):
        return self.decision_function(X)
