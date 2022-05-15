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

import scipy.stats as stats
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from .stationary_utils import make_stationary

def hotelling_t2(X_bar, X_mean, S_inv):
    X_bar_diff = X_bar - X_mean
    T2 = np.diag((X_bar_diff @ S_inv) @ X_bar_diff.T)
    return T2

class VAR_AnomalyDetector:
    """
    Anomaly detection using Vector Autoregression (VAR).
    Following paper "Using vector autoregressive residuals to monitor multivariate
    processes in the presence of serial correlation" by Pan et al.
    See https://www.researchgate.net/publication/223379823_Using_vector_autoregressive_residuals_to_monitor_multivariate_processes_in_the_presence_of_serial_correlation
    """
    def __init__(self):
        self.model_fitted = None
        self.UCL = None
        self.num_differencing = None
        self.best_order = None
        self.residuals_mean = None
        self.residuals_std = None
        self.inv_cov_residuals = None

    def compute_UCL(self, alpha=0.027):
        """
        Compute upper control limit (UCL) for Phase II (real-time process monitoring).
        Common value for alpha is 0.027 (three sigma).
        """
        m = self.model_fitted.nobs  # number of samples
        p = self.model_fitted.resid.shape[-1]  # number of variables
        
        F = stats.f.ppf(1 - alpha, dfn=p, dfd=m - p)
        self.UCL = (p * (m + 1) * (m - 1) / (m * m - m * p)) * F

        return self.UCL

    def fit(self, X, max_lags=15, verbose=False, max_num_differencing=3, missing='none', order_selection_criteria='aic'):
        """Fit model using the given training samples.

        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            max_lags (int, optional): To find the best fitting model, lags from 1 to this number are tested. Defaults to 15.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            max_num_differencing (int, optional): Maximum number of differencing to perform to make the time series stationary.
            missing (str, optional): How to handle missing values. See :class:`statsmodels.base.model.Model` for more information.
            order_selection_criteria (str, optional): Information criterion to use for VAR order selection. 
                See :class:`statsmodels.tsa.vector_ar.var_model.VAR` for more information.

        Returns:
            T2: Hotelling T^2 metric for the fitted training data.
        """
        X = pd.DataFrame(X) # To pandas dataframe (required by `make_stationary` function)

        # Check that each time series in multivariate time series is stationary, perform differencing as needed
        X, self.num_differencing = make_stationary(X, verbose = verbose, max_num_differencing = max_num_differencing)
        if self.num_differencing > 0:
            print(f'Multivariate signal was differenced {self.num_differencing} times to make each signal stationary.')

        X = X.to_numpy() # Back to numpy array
        
        # Find best multivariate model
        model = VAR(endog=X, missing=missing)
        self.model_fitted = model.fit(maxlags=max_lags, ic=order_selection_criteria, verbose=verbose)
        self.best_order = self.model_fitted.k_ar

        # Compute upper control limit (UCL)
        self.compute_UCL()

        # Compute error terms
        residuals = self.model_fitted.resid # error terms
        self.residuals_mean = np.mean(residuals, axis=0) # mean of error terms
        self.residuals_inv_cov = np.linalg.pinv(np.cov(residuals, rowvar=False)) # covariance matrix
        
        # Compute Hotelling T^2 metric on the train data
        T2 = hotelling_t2(residuals, self.residuals_mean, self.residuals_inv_cov)

        return T2

    def predict(self, X, verbose=True):
        """
        Predict anomaly score for each KPI observation in the given sequence.
        
        Args:
            X: (array-like): KPI observations, input of shape (n_samples, n_features).
            
        Returns:
            y_pred (array-like): Array of anomaly scores of shape (n_samples - self.best_order - self.num_differencing,)
                where 1 indicates anomaly and 0 non-anomaly.
            T2 (array-like): Hotelling T^2 metric for each observation 
                of shape (n_samples - self.best_order - self.num_differencing,).
        """
        X = pd.DataFrame(X)

        for _ in range(self.num_differencing):
            X = X.diff().dropna()

        # perform iterative predictions on test data
        X_pred = []
        for i in range(self.best_order, len(X)):
            X_pred.append(self.model_fitted.forecast(X.iloc[i - self.best_order:i].values, steps=1))

        X_pred = np.vstack(X_pred) # shape is (n_obs_test - best_order, n_features)
        
        # Compute Hotelling T^2 metric on the test data
        residuals = X.iloc[self.best_order:].values - X_pred
        T2 = hotelling_t2(residuals, self.residuals_mean, self.residuals_inv_cov)
        
        return X_pred, T2

    def predict_labels(self, X):
        """
        Predict anomaly label for each KPI observation in the given sequence.
        
        Args:
            X: (array-like): KPI observations, input of shape (n_samples, n_features).
            
        Returns:
            y_pred (array): Array of anomaly labels of shape (n_samples - self.best_order,)
                where 1 indicates anomaly and 0 non-anomaly.
        """
        X_pred, T_squared_test = self.predict(X)
        y_pred = (T_squared_test > self.UCL).astype(int)
        return y_pred