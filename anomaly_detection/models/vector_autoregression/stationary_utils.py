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

import pandas as pd
# Augmented Dickey-Fuller Test (ADF Test) / unit root test
from statsmodels.tsa.stattools import adfuller

def adf_test(ts, signif=0.05, verbose=True):
    """Perform Augmented Dickey-Fuller Test to test whether the given series is stationary.

    Args:
        ts (array-like): Time series to test.
        signif (float, optional): Significance level to determine whether the series is stationary. Defaults to 0.05.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        bool: Whether the series is stationary.
    """
    dftest = adfuller(ts, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# Lags', '# Observations'])
    for key, value in dftest[4].items():
       adf['Critical Value (%s)' % key] = value
    if verbose:
        print(adf)
    
    p = adf['p-value']
    if p <= signif:
        if verbose:
            print('Series is Stationary')
            print()
        return True
    else:
        if verbose:
            print('Series is Non-Stationary')
            print()
        return False

def make_stationary(X, verbose=False, max_num_differencing=3):
    """Make the given multivariate time series stationary if needed.
    Performs multiple iterations where each time each individual time series is tested
    for stationarity. If any series is non-stationary, a differencing operation is performed
    to remove the trend from the data. The same process is repeated until all the series are stationary
    until given max iterations.

    Args:
        X (array-like): Multivariate time series, of shape (n_samples, n_features).
        verbose (bool, optional): Whether to print verbose output. Defaults to False.¨
        max_num_differencing (int): Maximum number of times the series is differenced to make it stationary.

    Returns:
        X (array-like): Input with possible differencing performed i times to make the series stationary.
        i: Number of differencing performed to make the series stationary.
    """
    i = 0
    while i < max_num_differencing:
        if verbose:
            print('Testing stationarity of multivariate time series, iteration', i)
        all_channels_stationary = True
        for channel in range(X.shape[1]):
            is_stationary = adf_test(X.to_numpy()[:, channel], verbose=verbose)
            if not is_stationary:
                all_channels_stationary = False
        if all_channels_stationary:
            break
        # apply differencing to remove trend
        X = X.diff().dropna()
        i += 1
    return X, i

# inverting transformation
def invert_transformation(X, X_diff, n_differencing=1):
    """Revert back the differencing to get the differenced time series back to original scale."""
    
    X_diff = X_diff.copy()
    X_diff.loc[-1] = X.iloc[0] # insert original value at index 0 (we will be performing cumsum afterwards on this)
    X_diff.index = X_diff.index + 1  # shifting index
    X_diff.sort_index(inplace=True)
    
    return X_diff.cumsum()