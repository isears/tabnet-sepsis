import datetime

import pandas as pd


def get_feature_labels():
    """
    Returns feature labels in-order of their appearance in X
    """
    feature_ids = (
        pd.read_csv("cache/included_features.csv").squeeze("columns").to_list()
    )
    d_items = pd.read_csv("mimiciv/icu/d_items.csv", index_col="itemid")
    d_items = d_items.reindex(feature_ids)

    assert len(d_items) == len(feature_ids)

    return d_items["label"].to_list()


min_seq_len = datetime.timedelta(hours=24)
