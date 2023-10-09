from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling.baseRunner import BaseModelRunner


class LogisticRegressionRunner(BaseModelRunner):
    save_dir = "cache/LR"
    name = "LR"

    def __init__(self, default_cmd="cv") -> None:
        self.configured_model_factory = lambda: LogisticRegression(max_iter=1000)
        super().__init__(default_cmd)

    def _load_data(self):
        d = LabeledSparseTensor.load_from_pickle(self.data_src)
        X = d.get_snapshot_los()
        y = d.get_labels()

        return X, y


if __name__ == "__main__":
    LogisticRegressionRunner().parse_cmdline()
