# Copyright 2021 Vili Ketonen. All Rights Reserved.
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
# ==============================================================================

import numpy as np
from scipy import stats
from sklearn.utils.validation import check_is_fitted

class MEWMA:
    """
    MEWMA control chart.

    References:
        Multivariate Statistical Process Control Charts: An Overview
        (Bersimis 2006): https://mpra.ub.uni-muenchen.de/6399/1/MPRA
        Multivariate EWMA Charts:
        https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc343.htm
    """
    def __init__(self,
                 weight,
                 alpha=0.05):
        """Initialize MEWMA control chart.

        Args:
            weight (float): Weighting factor.
            alpha (float, optional): Confidence level to determine anomaly threshold. Defaults to 0.05.
        """
        self.weight = weight
        self.alpha = alpha

    def fit(self, X):
        """
        Fits the MEWMA control chart, i.e., computes the mean and covariance of historical
        training data of normal operation, which are used in the MEWMA formula during prediction.

        Args:
            X (np.ndarray): Historical training data of normal operation, of shape (n_samples, n_dimensions).
        """
        self.x_train_mean_ = X.mean(axis=0)
        self.x_train_covariance_ = np.cov(X, rowvar = False, ddof = 1)

    def predict(self, X):
        """
        Predict anomaly score for each KPI observation in the given sequence.
        
        Args:
            X: (np.ndarray): KPI observations, input of shape (n_samples, n_features).

        Returns:
            (np.ndarray): 1d array of anomaly scores of shape (n_samples,).
        """
        check_is_fitted(self)

        return self._generate_t2_ewma(X)
        
    def determine_control_limits(self, data, phase=1):
        """Determine upper control limit (UCL).
        Note that this is the UCL computation for Hotelling's T-Squared.
        For MEWMA, the common way to determine the UCL is to find the threshold
        so that one can achieve the specified in-control average run length (ARL),
        where run length is defined as the number of samples before the chart produces a signal.

        Args:
            data (np.ndarray): Data to use to determine the limit, of shape (n_samples, n_dimensions).
            phase (int, optional): Which phase
                1 = analysis of historical data
                2 = real-time process monitoring
                Defaults to 1.

        Returns:
            float: Upper control limit UCL.
        """
        n_obs, n_dims = data.shape
        df1 = n_dims / 2
        df2 = (n_obs - n_dims - 1) / 2
        if phase == 1:
            limit = ((n_obs - 1) ** 2) / n_obs * stats.beta.isf(self.alpha, df1, df2)
        else:
            limit = (n_dims * (n_obs + 1) * (n_obs - 1) /
                          (n_obs * (n_obs - n_dims))) * \
                         stats.f.isf(self.alpha,
                                     n_dims,
                                     n_obs - n_dims)
        return limit

    def _generate_t2_ewma(self, y):
        """Calculate T^2 MEWMA scores for given observations.

        Args:
            y (np.ndarray): Input data, of shape (n_samples, n_dimensions)

        Returns:
            np.ndarray: T^2 scores for each observation.
        """
        ewma_0 = self.x_train_mean_ # use historical mean as ewma_0
        ewma = [ewma_0]
        t2s = []
        covariance = self.x_train_covariance_ # use historical covariance
        cons = (self.weight / (2. - self.weight))
        for i in range(1, y.shape[0] + 1):
            exp_wt = cons * (1 - (1 - self.weight) ** (2 * i))
            cov_ith_ewma = np.linalg.pinv(exp_wt * covariance)
            ewma_i = self.weight * y[i - 1] + (1 - self.weight) * ewma[-1]
            ewma.append(ewma_i)
            t2 = np.matmul(np.matmul(ewma_i - ewma_0, cov_ith_ewma),
                           (ewma_i - ewma_0).transpose())
            t2s.append(t2)
        return np.array(t2s)