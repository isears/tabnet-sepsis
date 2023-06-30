import sys

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import BaseModelRunner, TabsepModelFactory
from tabsep.modeling.cvCommon import cv_runner


class LogisticRegressionFactory(TabsepModelFactory):
    def __call__(self) -> BaseEstimator:
        return LogisticRegression(max_iter=10000)


class LogisticRegressionRunner(BaseModelRunner):
    def cv(self):
        d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
        X = d.get_snapshot()
        y = d.get_labels()
        cv_runner(LogisticRegressionFactory(), X, y)

    def hparams(self):
        raise NotImplementedError()

    def importance(self):
        raise NotImplementedError()


if __name__ == "__main__":
    LogisticRegressionRunner().parse_cmdline()
