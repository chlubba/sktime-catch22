import numpy as np

from catch22Forest.tree import treeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor

from sktime.classifiers.base import BaseClassifier

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

from sklearn.utils import check_random_state
from sklearn.utils import check_array

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from catch22 import catch22_all


class catch22ForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 metric='euclidean',
                 metric_params=None,
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None):
        """A forest classifier based on catch22 features
        """
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.metric = metric
        self.metric_params = metric_params
        self.random_state = random_state

    def predict(self, X, check_input=True):
        return self.classes_[np.argmax(
            self.predict_proba(X, check_input=check_input), axis=1)]

    def predict_proba(self, X, check_input=True):
        # Correct formating of x
        if len(X.iloc[0]) == 1:  # UNI
            X = [np.array(X.iloc[i].iloc[0]).tolist() for i in range(0, len(X))]
        else:  # MULTI
            X = [[np.array(X.iloc[i].iloc[j]).tolist() for j in range(0, len(X.iloc[i]))] for i in range(0, len(X))]

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(
                X.ndim))

        if self.n_dims_ > 1 and X.ndim != 3:
            raise ValueError("illegal input dimensions X.ndim != 3")

        if X.shape[-1] != self.n_timestep_:
            raise ValueError("illegal input shape ({} != {})".format(
                X.shape[-1], self.n_timestep_))

        if X.ndim > 2 and X.shape[1] != self.n_dims_:
            raise ValueError("illegal input shape ({} != {}".format(
                X.shape[1], self.n_dims))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        X = X.reshape(X.shape[0], self.n_dims_ * self.n_timestep_)

        # compute catch22 features
        num_insts = X.shape[0]
        X_catch22 = []
        for i in range(num_insts):
            series = X[i, :]
            c22_dict = catch22_all(series)
            X_catch22.append(c22_dict['values'])

        return self.bagging_classifier_.predict_proba(X_catch22)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a random catch22 feature forest classifier
        """
        # Correct formating of x
        if len(X.iloc[0]) == 1:  # UNI
            X2 = [np.array(X.iloc[i].iloc[0]).tolist() for i in range(0, len(X))]
        else:  # MULTI
            X2 = [[np.array(X.iloc[i].iloc[j]).tolist() for j in range(0, len(X.iloc[i]))] for i in range(0, len(X))]

        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X2, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

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

        tree_classifier = treeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=random_state,
        )

        if n_dims > 1:
            tree_classifier.force_dim = n_dims

        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=tree_classifier,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)

        # compute catch22 features
        num_insts = X.shape[0]
        X_catch22 = []
        for i in range(num_insts):
            series = X[i, :]
            c22_dict = catch22_all(series)
            X_catch22.append(c22_dict['values'])

        self.bagging_classifier_.fit(X_catch22, y, sample_weight=sample_weight)
        return self
