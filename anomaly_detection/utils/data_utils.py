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
import matplotlib.pyplot as plt
import random
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler

def get_timeseries_from_array(data, timesteps):
    # read data from file
    dataX = []
    for i in range(len(data) - timesteps + 1):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)

def plot_predictions(x, x_pred, time_window_idx=0, num_data_points=500):
    """Plot predicted time series against the true (ground truth) time series.

    Args:
        x (np.ndarray): True time series, of shape (n_samples, n_timesteps, n_dimensions).
        x_pred (np.ndarray): Predicted time series, of shape (n_samples, n_timesteps, n_dimensions).
        time_window_idx (int, optional): Which time window index to plot. Defaults to 0.
        num_data_points (int, optional): Number of data points to plot. Defaults to 500.
    """
    for i in range(0, x.shape[-1]):  # plot each feature in a separate plot
        plt.figure(figsize=(20, 4))
        plt.plot(x[:, time_window_idx, i][:num_data_points], label='data')
        plt.plot(x_pred[:, time_window_idx, i][:num_data_points], label='predict')
        plt.legend()
        plt.show()

def preprocess(data, scaler=None):
    # scale the data values between 0 and 1
    if scaler is None:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data, scaler
    else:
        data = scaler.transform(data)
        return data

def generate_univariate_time_series_data(n_samples, periodicity=None, random_state=None, noise_level=0.3):
    rnd = check_random_state(random_state)
    t = np.linspace(0, n_samples - 1, num = n_samples)
    e_t = rnd.normal(0, 1, t.shape) # random noise
    t0 = rnd.uniform(10, 100) # shift in phase
    if periodicity is None:
        periodicity = rnd.uniform(10, 40) # seasonality
    return np.sin((t - t0) / periodicity) + noise_level * e_t

def generate_multivariate_time_series_data(n_samples=4000, n_dim=3, random_state=None, noise_level=0.3):
    rnd = check_random_state(random_state)
    t = np.zeros((n_samples, n_dim))
    for i in range(n_dim):
        t[:, i] = generate_univariate_time_series_data(n_samples, random_state=rnd, noise_level=noise_level)
        
    return t

def generate_multivariate_time_series_data_with_combined_seasonality(n_samples=4000, n_dim=3, combined_seasonality=5, random_state=None, noise_level=0.3):
    rnd = check_random_state(random_state)
    t = np.zeros((n_samples, n_dim))
    for i in range(n_dim):
        seasonal_comp = generate_univariate_time_series_data(n_samples, periodicity=combined_seasonality, random_state=rnd, noise_level=noise_level)
        t[:, i] = generate_univariate_time_series_data(n_samples, random_state=rnd, noise_level=noise_level) * 0.5 + seasonal_comp * 0.5
        
    return t

def generate_anomaly_in_multivariate_time_series_data(t, anomaly_type=None, min_duration=1, max_duration=20, duration=None, intensity=None, n_dim=None, n_dims_max=None, random_state=None, a_start=None, verbose=False):
    rnd = check_random_state(random_state)
    
    if a_start is None:
        a_start = rnd.randint(0, len(t))
    else:
        a_start = rnd.randint(a_start, len(t))
        
    if anomaly_type is None:
        anomaly_type=rnd.choice((-1, 1))
        
    if duration is None:
        if anomaly_type == -1:
            min_duration = 80
            max_duration = min_duration * 2
        duration = rnd.randint(min_duration, max_duration)
    a_end = a_start + duration
    
    if n_dim is None:
        max_dims = t.shape[1]
        if n_dims_max is not None:
            max_dims = n_dims_max
        n_dim = rnd.randint(1, max_dims + 1)
    
    dims = rnd.choice(t.shape[1], n_dim, replace=False)
        
    if anomaly_type == -1:
        if verbose:
            print(f'Generating collective anomaly at index: {a_start}-{a_end}\tdims: {dims}')
        for dim in dims:
            t[a_start:a_end, dim] = t[a_start, dim]
    else:
        if intensity is None:
            intensity = rnd.uniform(1.5, 2.5) # shock

        sign = rnd.choice((-1, 1))
        if verbose:
            print(f'Generating contextual anomaly at index: {a_start}-{a_end}\tdims: {dims}\tsign: {sign}\tintensity: {intensity}')
        for dim in dims:
            t[a_start:a_end, dim] = t[a_start:a_end, dim] + intensity * sign
            
    return {
        'anomaly_type': anomaly_type,
        'a_start': a_start,
        'a_end': a_end,
        'dims': dims
    }

def generate_multivariate_time_series_data_with_anomalies(n_dim=5, n_max_anomaly_dims=None, random_state=None, n_samples=8000, n_anomalies=10, anomalies_start_idx=None, verbose=False, combined_seasonality=None, noise_level=0.3):
    rnd = check_random_state(random_state)
    if combined_seasonality is None:
        t = generate_multivariate_time_series_data(n_dim=n_dim, random_state=rnd, n_samples=n_samples, noise_level=noise_level)
    else:
        t = generate_multivariate_time_series_data_with_combined_seasonality(n_dim=n_dim, random_state=rnd, n_samples=n_samples, combined_seasonality=combined_seasonality, noise_level=noise_level)
    anomalies = []
    for i in range(n_anomalies):
        anomaly = generate_anomaly_in_multivariate_time_series_data(t, random_state=rnd, a_start=anomalies_start_idx, n_dims_max=n_max_anomaly_dims, verbose=verbose)
        anomalies.append(anomaly)

    return t, anomalies