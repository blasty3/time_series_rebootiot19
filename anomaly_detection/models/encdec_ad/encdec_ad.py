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

import tensorflow as tf

from .model import build_encdec_ad
import numpy as np
from scipy.spatial.distance import mahalanobis

class EncDecAD():
    """
    Implementation of following the paper:
    EncDec-AD: LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection
    https://arxiv.org/pdf/1607.00148.pdf
    """
    def __init__(self,
                input_dim,
                timesteps,
                *args,
                optimizer=tf.keras.optimizers.Adam(),
                **kwargs):
        self.model = build_encdec_ad(
            input_dim,
            timesteps,
            *args,
            **kwargs)

        self.model.compile(optimizer=optimizer)

        self.anomaly_threshold = None
        self.timesteps = timesteps
        
    def __get_anomaly_score(self, x):
        """Get anomaly score.

        Args:
            x (np.ndarray): Input data of shape (batch_size, n_timesteps, x_dim)

        Returns:
            np.array: anomaly scores, shape (batch_size)
        """
        x_rec = self.model.predict(x)

        anomaly_scores = []
        for i in range(len(x_rec)):
            e = np.abs(x[i, -1] - x_rec[i, -1]) # error term, always take the final window (-1)
            anomaly_score = mahalanobis(e, self.errors_mean, self.errors_inv_cov) ** 2
            anomaly_scores.append(anomaly_score)

        return np.stack(anomaly_scores)
        
    def get_anomaly_score(self, x, batch_size):
        if not isinstance(x, tf.data.Dataset) and len(x.shape) == 2:
            x = tf.cast(x, tf.keras.backend.floatx())
            # transform the input data into Dataset containing batches of windows of shape (batch_size, window_length, n_dimensions)
            dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=x,
                targets=None,
                sequence_length=self.timesteps,
                sequence_stride=1,
                batch_size=batch_size
            )
            score_array = []
            for batch in dataset: # each batch has shape (batch_size, window_size, input_dim)
                score = self.__get_anomaly_score(batch)
                score_array.append(score)
            return np.concatenate(score_array, axis=0)
        else:
            return self.__get_anomaly_score(x)
            
    def predict_labels(self, x, batch_size, threshold=None, single_label_for_each_observation=True, only_last_point_of_window=True):
        assert threshold is not None or self.anomaly_threshold is not None, 'Call determine_anomaly_threshold-method first, or provide an anomaly threshold'
        if threshold is None:
            threshold = self.anomaly_threshold
            
        anomaly_scores = self.get_anomaly_score(x, batch_size, sum_datapoint_dimensions=single_label_for_each_observation, only_last_point_of_window=only_last_point_of_window)
        y_pred = (anomaly_scores > threshold).astype(int)
        return y_pred
        
    def determine_anomaly_threshold(self, x_train, batch_size):
        # not implemented
        pass

    def init_anomaly_score_calculation(self, valid_ds):
        validation_data_reconstructed = self.model.predict(valid_ds)  # numpy array of shape(batch_size, n_timesteps, x_dim)
        
        # convert valid_ds to numpy
        valid_data = []
        for batch in valid_ds:
            valid_data.append(batch)
        valid_data = tf.concat(valid_data, axis=0).numpy()
        
        # use last window from initial data and reconstruction [:, -1] = (batch_size, x_dim)
        val_errors = np.abs(valid_data[:, -1] - validation_data_reconstructed[:, -1])
        self.errors_mean = np.mean(val_errors, axis=0)
        self.errors_inv_cov = np.linalg.inv(np.cov(val_errors.T))
        
    def fit(self, x, batch_size=None, validation_data=None, *args, **kwargs):
        history = self.model.fit(x, batch_size=batch_size, validation_data=validation_data, *args, **kwargs)
        self.init_anomaly_score_calculation(validation_data)
        return history

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x, *args, **kwargs)