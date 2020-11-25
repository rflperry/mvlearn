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
# Omnibus embedding for multiview dimensionality reduction.
# Code from the https://github.com/neurodata/graspy package,
# reproduced and shared with permission.

from graspy.embed import OmnibusEmbed
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from .base import BaseEmbed
from ..utils.utils import check_Xs


class Omnibus(BaseEmbed):
    """
    Omnibus computes the pairwise distances for each view. Each
    of these matrices is a n x n dissimilarity matrix where n is the number
    of rows in each view. Omnibus embedding [#1Omni]_ is then performed
    over the dissimilarity matrices and the computed embeddings are returned.

    Parameters
    ----------
    n_components : strictly positive int (default = 2)
        Desired dimensionality of output embeddings. See graspy docs for
        additional details.

    distance_metric : string (default = 'euclidean')
        Distance metric used to compute pairwise distances. Metrics must
        be found in sklearn.neighbors.DistanceMetric.

    normalize : string or None (default = 'l1')
        Normalize function to use on views before computing
        pairwise distances. Must be 'l2', 'l1', 'max'
        or None. If None, the distance matrices will not be normalized.

    algorithm : string (default = 'randomized')
        SVD solver to use. Must be 'full', 'randomized', or 'truncated'.
        See graspy docs for details.

    n_iter : positive int (default = 5)
        Number of iterations for randomized SVD solver. See graspy docs for
        details.

    Attributes
    ----------
    embeddings_: list of arrays (default = None)
        List of Omnibus embeddings. One embedding matrix is provided
        per view. If fit() has not been called, embeddings_ is set to
        None.

    Notes
    -----
    From an implementation perspective, omnibus embedding is performed
    using the GrasPy package's implementation graspy.embed.OmnibusEmbed
    for dissimilarity matrices.

    References
    ----------
    .. [#1Omni] https://graspy.neurodata.io/tutorials/embedding/omnibus

    Examples
    --------
    >>> from mvlearn.embed import omnibus
    >>> import numpy as np
    >>> # Create 2 random data views with feature sizes 50 and 100
    >>> view1 = np.random.rand(1000, 50)
    >>> view2 = np.random.rand(1000, 100)
    >>> embedder = omnibus.Omnibus(n_components=3)
    >>> embeddings = embedder.fit_transform([view1, view2])
    >>> view1_hat, view2_hat = embeddings
    >>> print(view1_hat.shape, view2_hat.shape)
    (1000, 3) (1000, 3)
    """

    def __init__(self, n_components=2, distance_metric="euclidean",
                 normalize="l1",
                 algorithm="randomized",
                 n_iter=5):

        super().__init__()
        self.n_components = n_components
        self.normalize = normalize
        self.distance_metric = distance_metric
        self.algorithm = algorithm
        self.n_iter = n_iter
        self._check_params()
        self.embeddings_ = None

    def _check_params(self):
        """
        Checks that Omnibus arguments are valid. A ValueError
        is thrown if any are not. The checks performed are:
            - distance_metric is valid
            - algorithm is valid
            - normalize is valid
            - n_components is positive int
            - n_iter is positive int
        """

        valid_metrics = ['braycurtis', 'canberra',
                         'chebyshev', 'cityblock', 'correlation',
                         'cosine', 'dice', 'euclidean', 'hamming',
                         'jaccard', 'jensenshannon', 'kulsinski',
                         'mahalanobis', 'matching', 'minkowski',
                         'rogerstanimoto', 'russellrao', 'seuclidean',
                         'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

        valid_algorithms = ["randomized", "full", "truncated"]

        valid_normalize = ["l1", "l2", "max"]

        if self.distance_metric not in valid_metrics:
            raise ValueError("distance_metric must be in \
                             scipy.spatial.distances.pdist.")

        if self.algorithm not in valid_algorithms:
            raise ValueError("algorithm must be 'randomized', \
                            'full', or 'truncated'.")

        if self.normalize is not None and \
           self.normalize not in valid_normalize:
            raise ValueError("normalize must be 'l2', 'l1', or 'max'.")

        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError("n_components must be positive int.")

        if not isinstance(self.n_iter, int) or self.n_iter <= 0:
            raise ValueError("n_iter must be positive int.")

    def fit(self, Xs, y=None):
        """
        Fit the model with Xs and apply the embedding on Xs.
        The embeddings are saved as a class attribute.

        Parameters
        ==========
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to embed based on the prior fit function. Each
            X in Xs will receive its own embedding.
        y : ignored
            Included for API compliance.
        """
        Xs = check_Xs(Xs)
        dissimilarities = []
        for X in Xs:
            if self.normalize is not None:
                X = normalize(X, norm=self.normalize)
            dissimilarity = pairwise_distances(X, metric=self.distance_metric)

            dissimilarities.append(dissimilarity)

        embedder = OmnibusEmbed(n_components=self.n_components,
                                algorithm=self.algorithm,
                                n_iter=self.n_iter)

        self.embeddings_ = embedder.fit_transform(dissimilarities)

    def fit_transform(self, Xs, y=None):
        """
        Fit the model with Xs and apply the embedding on Xs using
        the fit() function. The resulting embeddings are returned.

        Parameters
        ==========
         Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to embed based on the prior fit function. Each
            X in Xs will receive its own embedding.
        y : ignored
            Included for API compliance.

        Returns
        =======
        embeddings : list of arrays
            list of (n_samples, n_components) matrices for each X in Xs.
        """

        self.fit(Xs)
        return self.embeddings_
