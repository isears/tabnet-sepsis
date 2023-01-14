from tabsep.dataProcessing.fileBasedDataset import FileBasedDataset


class TsaiDataset(FileBasedDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def collate_tsai(self, batch):
        X, y, pad_mask, _ = self.maxlen_padmask_collate(batch)
        X = X.transpose(1, 2)  # TSai expects (batch_dim, feat_dim, seq_len)
        return X, pad_mask, y.unsqueeze(dim=1)

    def __getitem__(self, indices: slice | list):
        """
        Basically, must combine collate_fn and __getitem__ into one
        """

        if type(indices) == slice:
            # TODO: should probably just modify original __getitem__ to
            # handle slices better...
            step = 1 if indices.step is None else indices.step

            batch = [
                super(TsaiDataset, self).__getitem__(idx)
                for idx in range(indices.start, indices.stop, step)
            ]
        elif type(indices) == list:
            batch = [super(TsaiDataset, self).__getitem__(idx) for idx in indices]
        else:
            raise ValueError(f"Unsupported data type: {type(indices)} for indexer")

        return self.collate_tsai(batch)

