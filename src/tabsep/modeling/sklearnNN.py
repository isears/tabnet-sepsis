from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier

from tabsep.dataProcessing import LabeledSparseTensor
from tabsep.modeling import BaseModelRunner, TabsepModelFactory
from tabsep.modeling.cvCommon import cv_runner


class NNFactory(TabsepModelFactory):
    def __call__(self) -> BaseEstimator:
        return MLPClassifier(
            solver="lbfgs",
            alpha=1e-5,
            hidden_layer_sizes=(5, 2),
            random_state=1,
            max_iter=10000,
        )


class NNRunner(BaseModelRunner):
    def cv(self):
        d = LabeledSparseTensor.load_from_pickle("cache/sparse_labeled.pkl")
        X = d.get_snapshot()
        y = d.get_labels()
        cv_runner(NNFactory(), X, y)

    def hparams(self):
        raise NotImplementedError()

    def importance(self):
        raise NotImplementedError()


if __name__ == "__main__":
    NNRunner().parse_cmdline()
