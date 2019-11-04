import numpy as np

from sklearn import tree

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils import check_array

__all__ = ["treeClassifier"]


class treeClassifier(BaseEstimator, ClassifierMixin):
    """A tree classifier."""

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 metric='euclidean',
                 metric_params=None,
                 force_dim=None,
                 random_state=None):
        """A decision tree
        :param max_depth: The maximum depth of the tree. If `None` the
           tree is expanded until all leafs are pure or until all
           leafs contain less than `min_samples_split` samples
           (default: None).
        :param min_samples_split: The minimum number of samples to
           split an internal node (default: 2).
        :param metric: Distance metric used to identify the best
           match. (default: `'euclidean'`)
        :param metric_params: Paramters to the distace measure
        :param force_dim: Force the number of dimensions (default:
           None). If `int`, `force_dim` reshapes the input to the
           shape `[n_samples, force_dim, -1]` to support the
           `BaggingClassifier` interface.
        :param random_state: If `int`, `random_state` is the seed used
           by the random number generator; If `RandomState` instance,
           `random_state` is the random number generator; If `None`,
           the random number generator is the `RandomState` instance
           used by `np.random`.
        """

        self.max_depth = max_depth
        self.max_depth = max_depth or 2**31
        self.min_samples_split = min_samples_split
        self.random_state = check_random_state(random_state)
        self.metric = metric
        self.metric_params = metric_params
        self.force_dim = force_dim

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a tree classifier from the training set (X, y)
        :param X: array-like, shape `[n_samples, n_timesteps]` or
           `[n_samples, n_dimensions, n_timesteps]`. The training time
           series.
        :param y: array-like, shape `[n_samples, n_classes]` or
           `[n_classes]`. Target values (class labels) as integers or
           strings.
        :param sample_weight: If `None`, then samples are equally
            weighted. Splits that would create child nodes with net
            zero or negative weight are ignored while searching for a
            split in each node. Splits are also ignored if they would
            result in any single class carrying a negative weight in
            either child node.
        :param check_input: Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        :returns: `self`
        """
        random_state = check_random_state(self.random_state)

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions")

        n_samples = X.shape[0]
        if isinstance(self.force_dim, int):
            X = np.reshape(X, [n_samples, self.force_dim, -1])

        n_timesteps = X.shape[-1]

        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        metric_params = self.metric_params
        if self.metric_params is None:
            metric_params = {}

        # min_sample_split = self.min_samples_split
        # self.n_classes_ = len(self.classes_)
        # self.n_timestep_ = n_timesteps
        # self.n_dims_ = n_dims

        # tree_builder = ClassificationShapeletTreeBuilder(
        #     self.max_depth,
        #     min_sample_split,
        #     distance_measure,
        #     X,
        #     y,
        #     sample_weight,
        #     random_state,
        #     self.n_classes_,
        # )
        #
        # self.root_node_ = tree_builder.build_tree()

        self.clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=random_state)

        # # compute catch22 features
        # num_insts = X.shape[0]
        # X_catch22 = []
        # for i in range(num_insts):
        #     series = X[i, :]
        #     c22_dict = catch22_all(series)
        #     X_catch22.append(c22_dict['values'])

        # fit classifier based on catch22 feature values
        self.clf.fit(X, y, sample_weight)

        return self

    def predict(self, X, check_input=True):
        """Predict the class for X
        :param X: array-like, shape `[n_samples, n_timesteps]` or
            `[n_samples, n_dimensions, n_timesteps]`. The input time
            series.
        :param check_input: Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        :returns: array of `shape = [n_samples]`. The predicted
            classes
        """
        return self.classes_[np.argmax(
            self.predict_proba(X, check_input=check_input), axis=1)]

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.  The predicted
        class probability is the fraction of samples of the same class
        in a leaf.
        :param X: array-like, shape `[n_samples, n_timesteps]` or
           `[n_samples, n_dimensions, n_timesteps]`. The input time
           series.
        :param check_input: Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        :returns: array of `shape = [n_samples, n_classes]`. The
            class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(
                X.ndim))

        if isinstance(self.force_dim, int):
            X = np.reshape(X, [X.shape[0], self.force_dim, -1])

        # if X.shape[-1] != self.n_timestep_:
        #     raise ValueError("illegal input shape ({} != {})".format(
        #         X.shape[-1], self.n_timestep_))
        #
        # if X.ndim > 2 and X.shape[1] != self.n_dims_:
        #     raise ValueError("illegal input shape ({} != {}".format(
        #         X.shape[1], self.n_dims))

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        metric_params = self.metric_params
        if self.metric_params is None:
            metric_params = {}


        # predictor = ClassificationShapeletTreePredictor(
        #     X, distance_measure, len(self.classes_))
        # return predictor.predict(self.root_node_)

        # # compute catch22 features
        # num_insts = X.shape[0]
        # X_catch22 = []
        # for i in range(num_insts):
        #     series = X[i, :]
        #     c22_dict = catch22_all(series)
        #     X_catch22.append(c22_dict['values'])

        return self.clf.predict_proba(X)