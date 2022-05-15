# Original version from https://github.com/gaussalgo/anlearn
# anlearn Copyright (C) 2020 Gauss Algorithmic a.s.
# GNU Lesser General Public License v3 or later (LGPLv3+)
# 
# Modified to support incremental learning (online historams)
# Copyright (C) 2020 Vili Ketonen.

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from typing import TypeVar

ArrayLike = TypeVar("ArrayLike")

# There is no DensityEstimationMixin in scikit-learn
class Histogram(BaseEstimator):
    """Histogram model

    Histogram model based on :obj:`scipy.stats.rv_histogram`.

    Parameters
    ----------
    bins : Union[int, str], optional
        * :obj:`int` - number of equal-width bins in the given range.
        * :obj:`str` - method used to calculate bin width (:obj:`numpy.histogram_bin_edges`).

        See :obj:`numpy.histogram_bin_edges` bins for more details, by default "auto"
    return_min : bool, optional
        Return minimal float value instead of 0, by default True

    Attributes
    ----------
    hist : numpy.ndarray
        Value of histogram
    bin_edges : numpy.ndarray
        Edges of histogram
    pdf : numpy.ndarray
        Probability density function
    """

    def __init__(self, bins: Union[int, str] = "auto") -> None:
        self.bins = bins

    def fit(self, X: np.ndarray) -> "Histogram":
        """Fit estimator

        Parameters
        ----------
        X : numpy.ndarray
            Input data, shape (n_samples,)

        Returns
        -------
        Histogram
            Fitted estimator
        """

        if self.bins == 'auto':
            self.bins = self.__hist_optimal(X)

        self.bin_width = np.ptp(X) / self.bins
        self.b = np.min(X)
        idxs = np.floor((X - self.b) / self.bin_width)

        self.hist, self.bin_edges = np.histogram(idxs, bins=np.arange(0, self.bins + 1))
        widths = self.bin_edges[1:] - self.bin_edges[:-1]

        self.count = np.sum(self.hist)
        
        start =  self.bin_edges[0]
        end = self.bin_edges[-2]
        hist_dict = {}
        i = 0
        for bin_idx in range(start, end + 1):
            hist_dict[bin_idx] = self.hist[i]
            i += 1

        self.hist = hist_dict

        return self

    def __hist_optimal(self, X):
        """
        Performs density estimation using penalized regular histogram
        - outputs how many bins should be put in a regular histogram
        
        References:
        Birgé and Rozenholc (2006)
        
        Args:
        X: the observations, n samples of an unknown density f
        
        Returns:
        int: optimal number of bins
        """
        xx = np.ravel(X)
        xmin = np.min(xx)
        xmax = np.max(xx)
        if xmin < 0 or xmax > 1:
            xx = (xx - xmin) / (xmax - xmin)
            
        def penalty(nbin):
            return nbin - 1 + np.log(nbin)** 2.5
        def likelihood(nbin):
            hist, bins = np.histogram(xx, bins=nbin)
            return (hist * np.log(nbin * np.maximum(hist, 1) / float(len(xx)))).sum()
        
        nbins = np.arange(1, np.floor(len(xx) / np.log(len(xx))) + 1, dtype='i')
        nbin = nbins[np.argmax([likelihood(n) - penalty(n) for n in nbins])]
        h = (xmax - xmin) / nbin
        return nbin

    def insert_value(self, x):
        idx = int(np.floor((x - self.b) / self.bin_width))
        if not idx in self.hist:
            self.hist[idx] = 0
        self.hist[idx] = self.hist[idx] + 1
        self.count += 1

    def remove_value(self, x):
        idx = int(np.floor((x - self.b) / self.bin_width))
        if not idx in self.hist:
            raise ValueError('No such value exists in histogram')
        self.hist[idx] = self.hist[idx] - 1
        self.count -= 1

    def remove_from_bin(self, idx, count):
        if not idx in self.hist:
            raise ValueError('No such bin index exists in histogram')
        self.hist[idx] = self.hist[idx] - count
        self.count -= count

    def predict_proba(self, X: Union[np.ndarray, float]) -> np.ndarray:
        """Predict probability

        Predict probability of input data X.

        Parameters
        ----------
        X : numpy.ndarray
            Input data, shape (n_samples,)

        Returns
        -------
        numpy.ndarray
            Probability estimated from histogram, shape (n_samples,)
        """
        if isinstance(X, np.ndarray):
            idxs = np.floor((X - self.b) / self.bin_width).astype(int)

            indexer = np.array([self.hist.get(i, 0) for i in range(idxs.min(), idxs.max() + 1)])
            bins = indexer[(idxs - idxs.min())]

            prob_x = bins / (self.count * self.bin_width)
            prob_x[prob_x == 0] = np.finfo(np.float).eps

            return prob_x
        else:
            idx = int(np.floor((X - self.b) / self.bin_width))
            
            if not idx in self.hist:
                return np.finfo(np.float).eps

            prob_x = self.hist[idx] / (self.count * self.bin_width)

            if prob_x == 0:
                prob_x = np.finfo(np.float).eps

            return prob_x


class LODA(BaseEstimator, OutlierMixin):
    """LODA: Lightweight on-line detector of anomalies [1]_

    LODA is an ensemble of histograms on random projections.
    See Pevný, T. Loda [1]_ for more details.

    Parameters
    ----------
    n_estimators : int, optional
        number of histograms, by default 1000
    bins : Union[int, str], optional
        * :obj:`int` - number of equal-width bins in the given range.
        * :obj:`str` - method used to calculate bin width (:obj:`numpy.histogram_bin_edges`).

        See :obj:`numpy.histogram_bin_edges` bins for more details, by default "auto"
    q : float, optional
        Quantile for compution threshold from training data scores.
        This threshold is used for `predict` method, by default 0.05
    random_state : Optional[int], optional
        Random seed used for stochastic parts., by default None
    n_jobs : Optional[int], optional
        Not implemented yet, by default None
    verbose : int, optional
        Verbosity of logging, by default 0

    Attributes
    ----------
    projections_ : numpy.ndarray
        Random projections, shape (n_estimators, n_features)
    hists_ : List[Histogram]
        Histograms on random projections, shape (n_estimators,)
    anomaly_threshold_ : float
        Treshold for :meth:`predict` function

    Examples
    --------
    >>> import numpy as np
    >>> from anlearn.loda import LODA
    >>> X = np.array([[0, 0], [0.1, -0.2], [0.3, 0.2], [0.2, 0.2], [-5, -5], [0.6, 0.7]])
    >>> loda = LODA(n_estimators=10, bins=10, random_state=42)
    >>> loda.fit(X)
    LODA(bins=10, n_estimators=10, random_state=42)
    >>> loda.predict(X)
    array([ 1,  1,  1,  1, -1,  1])

    References
    ----------
    .. [1] Pevný, T. Loda: Lightweight on-line detector of anomalies. Mach Learn 102, 275–304 (2016).
           <https://doi.org/10.1007/s10994-015-5521-0>
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        bins: Union[int, str] = "auto",
        q: float = 0.05,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ) -> None:

        self.n_estimators = n_estimators
        self.bins = bins
        self.q = q
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs  # TODO

        self._validate()

    def _validate(self) -> None:
        if not (isinstance(self.n_estimators, int) or isinstance(self.n_estimators, float)) or self.n_estimators < 0:
            raise ValueError("LODA: n_estimators must be > 0")

        if self.q < 0 or self.q > 1:
            raise ValueError("LODA: q must be in [0; 1]")

    def _init_projections(self) -> None:
        self.projections_ = np.zeros((self.n_estimators, self._shape[1]))

        non_zero_w = np.rint(self._shape[1] * (self._shape[1] ** (-1 / 2))).astype(int)

        rnd = check_random_state(self.random_state)

        indexes = rnd.rand(self.n_estimators, self._shape[1]).argpartition(
            non_zero_w, axis=1
        )[:, :non_zero_w]

        rand_values = rnd.normal(size=indexes.shape)

        for projection, chosen_d, values in zip(
            self.projections_, indexes, rand_values
        ):
            projection[chosen_d] = values

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "LODA":
        """Fit estimator

        Parameters
        ----------
        X : ArrayLike
            Input data, shape (n_samples, n_features)
        y : Optional[ArrayLike], optional
            Present for API consistency by convention, by default None

        Returns
        -------
        LODA
            Fitted estimator
        """
        raw_data = self.__check_array(X)

        self._shape = raw_data.shape

        # if n_estimators < 1, determine number of histograms automatically (see section 3.4 in paper)
        if self.n_estimators < 1:
            tau = self.n_estimators
            self.n_estimators = 1000
        else:
            tau = 0

        self._init_projections()

        w_X = raw_data @ self.projections_.T

        self.hists_ = []
        X_prob = []

        initial_variance = 1
        previous_est = np.zeros(self._shape[0])
        i = 0
        for w_x in w_X.T:
            new_hist = Histogram(bins=self.bins).fit(w_x)
            self.hists_.append(new_hist)
            prob = new_hist.predict_proba(w_x)
            X_prob.append(prob)

            # if auto choosing the number of histograms (see section 4.1.1 in the paper)
            if tau > 0:
                i_hat = np.mean(np.log(X_prob), axis=0)
                variance = np.nanmean(np.abs(previous_est - i_hat))
                if i == 1:
                    initial_variance = variance
                previous_est = i_hat

                # if the variance has decreased below threshold, stop adding new histograms
                if variance / initial_variance < tau:
                    if self.verbose:
                        print('Optimal number of histograms', i)
                    self.n_estimators = len(self.hists_)
                    break

                i += 1

        # drop projections for which no histogram was built (when auto choosing the number of histograms)
        self.projections_ = self.projections_[:self.n_estimators,:]

        X_scores = np.mean(np.log(X_prob), axis=0)

        self.anomaly_threshold_ = np.quantile(X_scores, self.q)

        return self

    def online_predict(self, raw_data, win_size=100):
        """
        Perform online prediction on the given data one sample at a time.
        Uses incremental histograms (histograms are updated on each new sample).

        See section 3.4 in the paper.

        Parameters
        ----------
        X : ArrayLike
            Input data, shape (n_samples, n_features)

        Returns
        -------
        tuple : (anomaly_labels, anomaly_scores, feature_scores)
        """

        n_samples = raw_data.shape[0]

        # optimize the parameters by calling LODA on the first batch of data
        self.fit(raw_data[:win_size])

        # project the input data
        w_X = raw_data @ self.projections_.T

        window_idx = 0
        window = np.zeros((2 * win_size, len(self.hists_)), dtype=int) # buffer to store history values
        X_scores = np.zeros(n_samples) # anomaly scores
        f_scores = np.zeros((n_samples, self._shape[1])) # feature scores
        zero_projections = self.projections_ == 0
        for i in range(n_samples):
            x = w_X[i]
            x_log_prob = np.zeros(len(self.hists_))
            for j in range(len(self.hists_)):
                hist = self.hists_[j]

                # get index of bin for this point in histogram
                idx = int(np.floor((x[j] - hist.b) / hist.bin_width))

                # get probability for this point in histogram j
                x_log_prob[j] = np.log(hist.predict_proba(x[j]))

                # update the current histogram by inserting the point
                hist.insert_value(x[j])
                
                # remove the contribution of the oldest point in the buffer
                idx_from_win = window[window_idx][j]
                count_from_win = window[window_idx + 1][j]
                if count_from_win > 0:
                    hist.remove_from_bin(idx_from_win, count_from_win)
                
                # finally, update the buffer
                window[window_idx][j] = idx
                window[window_idx + 1][j] = 1

            # get final anomaly score and feature scores for this sample
            X_scores[i] = np.mean(x_log_prob)
            f_scores[i,:] = self.__score_features(-x_log_prob, zero_projections)

            # update the buffer index
            window_idx += 2
            if window_idx >= win_size * 2:
                window_idx = 0

        self.anomaly_threshold_ = np.quantile(X_scores, self.q)

        return np.where(X_scores < self.anomaly_threshold_, -1, 1), X_scores, f_scores

    def online_predict_two_windows(self, raw_data, win_size=100):
        """
        Perform online prediction on the given data one sample at a time.

        Uses two windows of histograms
            * first window is used to predict
            * the other window is being constructed on newly arrived samples
        When the other window is full, it replaces the first window.
        
        See section 3.4 in the paper.

        Parameters
        ----------
        X : ArrayLike
            Input data, shape (n_samples, n_features)

        Returns
        -------
        tuple : (anomaly_labels, anomaly_scores, feature_scores)
        """

        n_samples = raw_data.shape[0]

        # optimize the parameters by calling LODA on the first batch of data
        self.fit(raw_data[:win_size])

        # project the input data
        w_X = raw_data @ self.projections_.T

        updating_window_hists = []
        for i in range(len(self.hists_)):
            new_hist = Histogram()
            new_hist.bin_width = self.hists_[i].bin_width
            new_hist.count = 0
            new_hist.hist = {}
            new_hist.b = self.hists_[i].b
            updating_window_hists.append(new_hist)

        X_scores = np.zeros(n_samples) # anomaly scores
        f_scores = np.zeros((n_samples, self._shape[1])) # feature scores
        zero_projections = self.projections_ == 0
        for i in range(n_samples):
            x = w_X[i]
            x_log_prob = np.zeros(len(self.hists_))
            for j in range(len(self.hists_)):
                hist = self.hists_[j]

                # get index of bin for this point in histogram j
                idx = int(np.floor((x[j] - hist.b) / hist.bin_width))

                # get probability for this point in histogram j
                x_log_prob[j] = np.log(hist.predict_proba(x[j]))

                # update the corresponding histogram in the updating window by the point
                updating_window_hists[j].insert_value(x[j])

            # get final anomaly score and feature scores for this sample
            X_scores[i] = np.mean(x_log_prob)
            f_scores[i,:] = self.__score_features(-x_log_prob, zero_projections)

            # check if we need to switch windows
            if (i + 1) % win_size == 0:
                # switch windows
                self.hists_ = updating_window_hists
                # reset updating window
                updating_window_hists = []
                for i in range(len(self.hists_)):
                    new_hist = Histogram()
                    new_hist.bin_width = self.hists_[i].bin_width
                    new_hist.count = 0
                    new_hist.hist = {}
                    new_hist.b = self.hists_[i].b
                    updating_window_hists.append(new_hist)

        self.anomaly_threshold_ = np.quantile(X_scores, self.q)

        return np.where(X_scores < self.anomaly_threshold_, -1, 1), X_scores, f_scores

    def __check_array(self, X: ArrayLike) -> np.ndarray:
        return check_array(
            X, accept_sparse=True, dtype="numeric", force_all_finite=True
        )

    def __log_prob(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, attributes=["projections_", "hists_"])

        raw_data = self.__check_array(X)

        w_X = raw_data @ self.projections_.T

        X_prob = np.array(
            [hist.predict_proba(w_x) for hist, w_x in zip(self.hists_, w_X.T)]
        )

        return np.log(X_prob)

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Anomaly scores for samples

        Average of the logarithm probabilities estimated of individual projections.
        Output is proportional to the negative log-likelihood of the sample, that
        means the less likely a sample is, the higher the anomaly value it receives [1]_.
        This score is reversed for scikit-learn compatibility.

        Parameters
        ----------
        X : ArrayLike
            Input data, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            The anomaly score of the input samples. The lower, the more abnormal.
            Shape (n_samples,)
        """
        X_log_prob = self.__log_prob(X)

        X_scores = np.mean(X_log_prob, axis=0)

        return X_scores

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict if samples are outliers or not

        Samples with a score lower than :attr:`anomaly_threshold_` are considered
        to be  outliers.

        Parameters
        ----------
        X : ArrayLike
            Input data, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            1 for inlineres, -1 for outliers, shape (n_samples,)
        """
        check_is_fitted(self, attributes=["anomaly_threshold_"])

        scores = self.score_samples(X)

        return np.where(scores < self.anomaly_threshold_, -1, 1)

    def score_features(self, X: ArrayLike) -> np.ndarray:
        r"""Feature importance

        Feature importance is computed as a one-tailed two-sample t-test between
        :math:`-log(\hat{p}_i)` from histograms on projections with and without a
        specific feature. The higher the value is, the more important feature is.

        See full description in **3.3  Explaining the cause of an anomaly** [1]_ for
        more details.

        Parameters
        ----------
        X : ArrayLike
            input data, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            Feature importance in anomaly detection.


        Notes
        -----

        .. math::

            t_j = \frac{\mu_j - \bar{\mu}_j}{
                \sqrt{\frac{s^2_j}{|I_j|} + \frac{\bar{s}^2_j}{|\bar{I_j}|}}}


        """
        X_neg_log_prob = -self.__log_prob(X)

        zero_projections = self.projections_ == 0

        return self.__score_features(X_neg_log_prob, zero_projections)

    def __score_features(self, X_neg_log_prob: np.ndarray, zero_projections: np.ndarray) -> np.ndarray:
        results = []
        # t-test for every feature
        for j_feature in range(self._shape[1]):
            i_with_feature = X_neg_log_prob[~zero_projections[:, j_feature]]
            i_wo_feature = X_neg_log_prob[zero_projections[:, j_feature]]

            t_j = (
                np.mean(i_with_feature, axis=0) - np.mean(i_wo_feature, axis=0)
            ) / np.sqrt(
                np.var(i_with_feature, axis=0) / i_with_feature.shape[0]
                + np.var(i_wo_feature, axis=0) / i_wo_feature.shape[0]
            )

            results.append(t_j)

        return np.vstack(results).T