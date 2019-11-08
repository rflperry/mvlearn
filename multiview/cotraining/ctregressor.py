# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implements multi-view (2-view) or single view co-training regressor.

from .base import BaseCoTrainEstimator
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from ..utils.utils import check_Xs, check_Xs_y_nan_allowed


class CTRegressor(BaseCoTrainEstimator):
    def __init__(
                 self,
                 k_neighbors=3,
                 p=[2, 5],
                 regressor_weights=[.5, .5],
                 random_state=0
                 ):
        """
        Co-Training Regressor

        This class implements the co-training regressor with the framework as
        described in [1]. If using with multi-view data, this should ideally
        be used on 2 views of the input data which satisfy the 3 conditions
        for multi-view co-training (sufficiency, compatibility, conditional
        independence) as in [2]. Extends BaseCoTrainEstimator.

        Parameters
        ----------
        k_neighbors : int
            The number of neighbors for KNeighborsRegressor estimators and for
            estimating the change in MSE of unlabeled samples.
        p : List of int, length = n_views
            Distance orders used for KNeighborsRegressors.
        regressor_weights : List of real numbers, length = n_views,
                            default [.5, .5]
            The weights to assign to each view's regressor's estimates when
            computing the fully regression predictions. Final predictions are
            computed as regressor_weights[0] * estimator1_predictions +
            regressor_weights[1] * estimator2_predictions.
        random_state : int
            The starting random seed for fit() and other class operations.

        Attributes
        ----------
        estimator1 : sklearn KNeighborsRegressor object
            The regressor used on view 1.
        estimator2 : sklearn KNeighborsRegressor object
            The regressor used on view 2.
        class_name: string
            The name of the class.
        n_views_ : int
            The number of views supported by the multi-view classifier
        p_ : List of int, length = n_views
            Distance orders used for KNeighborsRegressors.
        num_iter_ : int
            Maximum number of training iterations to run.
        regressor_weights_ : List of real numbers, length = n_views,
                             default [.5, .5]
            The weights to assign to each view's regressor's estimates when
            computing the fully regression predictions. Final predictions
            are computed as regressor_weights[0] * estimator1_predictions +
            regressor_weights[1] * estimator2_predictions.
        random_state : int
            The starting random seed for fit() and other class operations.

        References
        ----------
        [1] Zhou, Z. H., & Li, M. (2005, July). Semi-Supervised Regression
        with Co-Training. In IJCAI (Vol. 5, pp. 908-913).
        
        [2] Blum, A., & Mitchell, T. (1998, July). Combining labeled and
        unlabeled_pool data with co-training. In Proceedings of the eleventh
        annual conference on Computational learning theory (pp. 92-100). ACM.
        """

        # initialize a BaseCTEstimator object
        super().__init__(KNeighborsRegressor(n_neighbors=k_neighbors, p=p[0]),
                         KNeighborsRegressor(n_neighbors=k_neighbors, p=p[1]),
                         random_state)

        self.p_ = [p[0], p[1]]

        self.n_views_ = 2  # only 2 view learning supported currently

        self.class_name = "CTRegressor"

        self.k_neighbors_ = k_neighbors

        self.regressor_weights_ = regressor_weights


    def fit(
            self,
            Xs,
            y,
            X_test,
            y_test,
            unlabeled_pool_size=100,
            num_iter=50,
            ):
        """
        Fit the regressor object to the data in Xs, y.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
            A list of the different views of data to train on.
        y : array, shape (n_samples,)
            The labels of the training data. Unlabeled_pool examples should
            have label np.nan.
        unlabeled_pool_size : int, optional (default=75)
            The number of unlabeled_pool samples which will be kept in a
            separate pool for classification and selection by the updated
            classifier at each training iteration.
        num_iter : int, optional (default=50)
            The maximum number of training iterations to run.
        """

        Xs, y = check_Xs_y_nan_allowed(Xs,
                                       y,
                                       multiview=True,
                                       enforce_views=self.n_views_)

        y = np.array(y)

        np.random.seed(self.random_state)

        self.unlabeled_pool_size_ = unlabeled_pool_size
        self.num_iter_ = num_iter

        # extract the multiple views given
        X1 = Xs[0]
        X2 = Xs[1]

        # the full set of unlabeled samples
        U = [i for i, y_i in enumerate(y) if np.isnan(y_i)]

        # shuffle unlabeled_pool data for easy random access
        np.random.shuffle(U)

        # the small pool of unlabled samples to draw from in training
        unlabeled_pool = U[-min(len(U), self.unlabeled_pool_size_):]

        # the labeled sample indices
        L = [i for i, y_i in enumerate(y) if ~np.isnan(y_i)]

        # remove the pool from overall unlabeled data
        U = U[:-len(unlabeled_pool)]

        it = 0

        iter_errors = []
        iter_errors1 = []
        iter_errors2 = []

        while it < self.num_iter_ and U:
            it += 1

            # fit each model to its respective view
            self.estimator1.fit(X1[L], y[L])
            self.estimator2.fit(X2[L], y[L])

            if X_test is not None:
                y_hat_test = self.predict(X_test)
                iter_errors.append(((np.linalg.norm(y_hat_test - y_test))**2)/X_test[0].shape[0])
                y_hat_test1 = self.estimator1.predict(X_test[0])
                iter_errors1.append(((np.linalg.norm(y_hat_test1 - y_test))**2)/X_test[0].shape[0])
                y_hat_test2 = self.estimator2.predict(X_test[1])
                iter_errors2.append(((np.linalg.norm(y_hat_test2 - y_test))**2)/X_test[0].shape[0])

            y_hat1 = self.estimator1.predict(X1[unlabeled_pool])
            y_hat2 = self.estimator2.predict(X2[unlabeled_pool])

            neighbors1 = (self.estimator1.kneighbors(X1[unlabeled_pool], n_neighbors=self.k_neighbors_))[1]
            neighbors2 = (self.estimator1.kneighbors(X2[unlabeled_pool], n_neighbors=self.k_neighbors_))[1]

            # find sample in each view which lowers the MSE the most
            delta_MSE1 = []
            for sample, (u, neigh) in enumerate(zip(unlabeled_pool, neighbors1)):
                new_L = L.copy()
                new_L.append(u)
                new_y = np.concatenate((y[L].copy(), np.array(y_hat1[sample]).reshape(1,)))
                new_estimator = KNeighborsRegressor(n_neighbors=self.k_neighbors_, p=self.p_[0])
                new_estimator.fit(X1[new_L], new_y)
                delta_MSE1.append(self.estimate_delta_MSE_(self.estimator1, new_estimator, (X1[L])[neigh], (y[L])[neigh]))

            best_delta_idx = np.argmin(delta_MSE1)
            now_labeled = []
            add_labels = []
            if delta_MSE1[best_delta_idx] > 0:
                now_labeled.append(unlabeled_pool[best_delta_idx])
                add_labels.append(y_hat1[best_delta_idx])

            delta_MSE2 = []
            for sample, (u, neigh) in enumerate(zip(unlabeled_pool, neighbors2)):
                new_L = L.copy()
                new_L.append(u)
                new_y = np.concatenate((y[L].copy(), np.array(y_hat2[sample]).reshape(1,)))
                new_estimator = KNeighborsRegressor(n_neighbors=self.k_neighbors_, p=self.p_[1])
                new_estimator.fit(X2[new_L], new_y)
                delta_MSE2.append(self.estimate_delta_MSE_(self.estimator2, new_estimator, (X2[L])[neigh], (y[L])[neigh])) 

            # find top 2 in case overlap with view 1 selection
            best_delta_idx = np.argsort(delta_MSE2)[-2:][::-1]
            if delta_MSE2[best_delta_idx[0]] > 0 and unlabeled_pool[best_delta_idx[0]] not in now_labeled:
                now_labeled.append(unlabeled_pool[best_delta_idx[0]])
                add_labels.append(y_hat1[best_delta_idx[0]])
            elif delta_MSE2[best_delta_idx[1]] > 0:
                now_labeled.append(unlabeled_pool[best_delta_idx[1]])
                add_labels.append(y_hat1[best_delta_idx[1]])

            # create new labels for new additions to the labeled group
            for x, y_hat in zip(now_labeled, add_labels):
                y[x] = y_hat
                #L.extend([x])
                L.append(x)

            # remove newly labeled samples from unlabeled_pool
            unlabeled_pool = [elem for elem in unlabeled_pool
                              if not (elem in now_labeled)]

            # add new elements to unlabeled_pool
            add_counter = 0
            while add_counter != len(now_labeled) and U:
                add_counter += 1
                unlabeled_pool.append(U.pop())

        # fit the overall model on fully "labeled" data
        self.estimator1.fit(X1[L], y[L])
        self.estimator2.fit(X2[L], y[L])

        return iter_errors1, iter_errors2, iter_errors


    def estimate_delta_MSE_(self, old_estimator, new_estimator, X, y):
        """
        Estimate the decrease in MSE of the new estimator based on a small
        sample of neighbors.

        Parameters
        ----------
        old_estimator: estimator object
            The current estimator trained on less data.

        new_estimator: estimator object
            The new estimator trained with additional data.

        X : array-like, shape (n_samples, n_features)
            The truly labeled data that the old estimator was trained on.

        y : array-like, shape (n_samples,)
            The labels for the samples in X.

        Returns
        -------
        delta_MSE : float
            The estimated change in MSE (old MSE - new MSE) when using
            the new_estimator.
        """

        # estimate the change in MSE
        y_hat_old = old_estimator.predict(X)
        y_hat_new = new_estimator.predict(X)

        return (np.sum((y-y_hat_old)**2 - (y-y_hat_new)**2))/X[0].shape[0]


    def predict(self, Xs):
        """
        Predict the regression output of the examples in the two input views.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
            A list of the different views of data to predict.

        Returns
        -------
        y : array-like (n_samples,)
            The estimated values.
        """

        Xs = check_Xs(Xs,
                      multiview=True,
                      enforce_views=self.n_views_)

        return self.regressor_weights_[0] * self.estimator1.predict(Xs[0])
                + self.regressor_weights_[1] * self.estimator2.predict(Xs[1])
