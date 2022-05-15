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
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
tfd = tfp.distributions

from .model import build_dsvae_ad
from anomaly_detection.models.spot.spot import SPOT
import numpy as np
from collections.abc import Mapping

class DSVAE_AD():
    """
    DSVAE-AD: Disentangled Sequential Variational Autoencoder Based Anomaly Detector.
    http://urn.fi/URN:NBN:fi:aalto-202101311730
    
    Based on OmniAnomaly:
    Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network.
    https://dl.acm.org/doi/10.1145/3292500.3330672
    """
    def __init__(self,
                input_dim,
                timesteps,
                *args,
                n_samples_train=1,
                n_samples_predict=1,
                use_anomaly_labels_for_training=False,
                optimizer=tf.keras.optimizers.Adam(),
                **kwargs):
        self.model, _, _ = build_dsvae_ad(
            input_dim,
            timesteps,
            *args,
            use_anomaly_labels_for_training=use_anomaly_labels_for_training,
            n_samples=n_samples_train,
            **kwargs)

        self.model.compile(optimizer=optimizer)
        
        # make identical model for prediction, only the n_samples is different
        self._model_pred, self._encoder_pred, self._decoder_pred = build_dsvae_ad(
            input_dim,
            timesteps,
            *args,
            use_anomaly_labels_for_training=use_anomaly_labels_for_training,
            n_samples=n_samples_predict,
            **kwargs)
        self._model_pred.trainable = False
        self._encoder_pred.trainable = False
        self._decoder_pred.trainable = False

        self.anomaly_threshold = None
        self.timesteps = timesteps
        self.use_anomaly_labels_for_training = use_anomaly_labels_for_training
        
    def __find_anomaly_threshold_with_pot(self, anomaly_scores, q=1e-3, level=0.98):
        """
        Runs the POT algorithm on the given anomaly score array to find a suitable anomaly threshold.
        Args:
            anomaly_scores (np.ndarray): The data used to calculate the threshold. Of shape (n_samples,).
                Here, this array corresponds to the anomaly scores of the training set.
            q (float): Detection level, extreme quantile (risk), regulates the number of false positives.
            level (float): Probability associated with the initial threshold t. Should be adjusted for each different dataset.
        Returns:
            float: Determined anomaly threshold.
        """
        spot = SPOT(q)
        spot.fit(anomaly_scores, []) # import data to SPOT
        spot.initialize(level=level) # run the calibration (initialization) step, i.e. the POT algorithm
        return spot.extreme_quantile
        
    def __get_anomaly_score(self, x, sum_datapoint_dimensions=True, only_last_point_of_window=True):
        """
        Calculates the anomaly score for the given observations.
        The score is defined as the negative log-likelihood (reconstruction probability) of an observation with respect to the reconstructed
        distribution of the observation, i.e. - p(x|z). A high score indicates an anomaly (low reconstruction probability).
        
        Args:
            x (array): Input of shape (n_samples, window_size, n_dimensions).
            
            sum_datapoint_dimensions (bool): Whether to reduce the anomaly scores of a single data
            point to a single number by summing the anomaly score of each individual dimension.
            
            only_last_point_of_window (bool): Whether to return anomaly score only for the last data point in each window.
            
        Returns:
            anomaly_score (tf.Tensor): Output of shape 
            (n_samples,) when sum_datapoint_dimensions = True and only_last_point_of_window = True
                * Useful when one wants to know for each window whether the last point of the window
                  is anomalous based on the other (preceeding) points in the window.
            or (n_samples, n_dimensions) when sum_datapoint_dimensions = False and only_last_point_of_window = True
                * Useful when one wants to know the anomaly scores of each dimension of the last point in the window for each window.
            or (n_samples, window_size), when sum_datapoint_dimensions = True and only_last_point_of_window = False
                * Useful when one wants to know the anomaly score of each data point of each window.
            or (n_samples, window_size, n_dimesions) when sum_datapoint_dimensions = False and only_last_point_of_window = False.
                * Useful whe one wants to know the anomaly score of each dimension of each data point of each window.
        """
        p_x_given_z, _ = self._model_pred(x)  # both have shape (n_samples, batch_size, n_timesteps, input_dim)
        if isinstance(x, Mapping) and self.use_anomaly_labels_for_training:
            x = x['encoder_input']
        x_mean = p_x_given_z.mean()
        x_std = p_x_given_z.stddev()
        # note: we can factorize the multivariate diagonal normal distribution into product of
        # individidual normal distributions for each dimension (assuming independence)
        p_x_given_z = tfd.Normal(loc = x_mean, scale = x_std)
        log_prob = p_x_given_z.log_prob(x)
        if tf.rank(log_prob) == 4:
            log_prob = tf.reduce_mean(log_prob, axis=0) # avg over samples
        if sum_datapoint_dimensions:
            # this would be equal to - MultivariateNormalDiag(x_mean, scale_diag=x_std).log_prob(x)
            reconstruction_prob = - tf.reduce_sum(log_prob, axis=2)
        else:
            reconstruction_prob = -log_prob
        if only_last_point_of_window:
            return reconstruction_prob[:, -1]
        else:
            return reconstruction_prob
        
    def get_anomaly_score(self, x, batch_size, sum_datapoint_dimensions=True, only_last_point_of_window=True):
        """
        Get anomaly score for each KPI observation in the given sequence.
        
        Args:
            x: (array): KPI observations, input of shape (n_samples, n_dimensions).
            
        Returns:
            anomaly_scores (array): Array of anomaly scores of shape
            (n_samples - window_size + 1,) when sum_datapoint_dimensions = True and only_last_point_of_window = True
                * Useful when one wants to know for each window whether the last point of the window
                  is anomalous based on the other (preceeding) points in the window.
            or (n_samples - window_size + 1, n_dimensions) when sum_datapoint_dimensions = False and only_last_point_of_window = True
                * Useful when one wants to know the anomaly scores of each dimension of the last point in the window for each window.
            or (n_samples - window_size + 1, window_size), when sum_datapoint_dimensions = True and only_last_point_of_window = False
                * Useful when one wants to know the anomaly score of each data point of each window.
            or (n_samples - window_size + 1, window_size, n_dimesions) when sum_datapoint_dimensions = False and only_last_point_of_window = False.
                * Useful whe one wants to know the anomaly score of each dimension of each data point of each window.
            The first dimension is `n_samples - window_size + 1` because the input samples are put into a sliding window 
            and there are `n_samples - window_size + 1` of such windows.
        """
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
            if self.use_anomaly_labels_for_training:
                dataset = dataset.map(lambda inputs:  {'encoder_input': inputs, 'x_input_labels': tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1]))})
            score_array = []
            for batch in dataset: # each batch has shape (batch_size, window_size, input_dim)
                score = self.__get_anomaly_score(batch, sum_datapoint_dimensions=sum_datapoint_dimensions, only_last_point_of_window=only_last_point_of_window)
                score_array.append(score)
            return np.concatenate(score_array, axis=0)
        else:
            return self.__get_anomaly_score(x, sum_datapoint_dimensions=sum_datapoint_dimensions, only_last_point_of_window=only_last_point_of_window)
            
    def predict_labels(self, x, batch_size, threshold=None, single_label_for_each_observation=True, only_last_point_of_window=True):
        """
        Predict anomaly label for each KPI observation in the given sequence.
        
        Args:
            x: (array): KPI observations, input of shape (n_samples, n_dimensions).
            
        Returns:
            anomaly_labels (array): Array of anomaly labels of shape
            (n_samples - window_size + 1,) when sum_datapoint_dimensions = True and only_last_point_of_window = True
                * Useful when one wants to know for each window whether the last point of the window
                  is anomalous based on the other (preceeding) points in the window.
            or (n_samples, n_dimensions) when sum_datapoint_dimensions = False and only_last_point_of_window = True
                * Useful when one wants to know the anomaly scores of each dimension of the last point in the window for each window.
            or (n_samples, window_size), when sum_datapoint_dimensions = True and only_last_point_of_window = False
                * Useful when one wants to know the anomaly score of each data point of each window.
            or (n_samples, window_size, n_dimesions) when sum_datapoint_dimensions = False and only_last_point_of_window = False.
                * Useful whe one wants to know the anomaly score of each dimension of each data point of each window.
            The first dimension is `n_samples - window_size + 1` because the input samples are put into a sliding window 
            and there are `n_samples - window_size + 1` of such windows.
        """
        assert threshold is not None or self.anomaly_threshold is not None, 'Call determine_anomaly_threshold-method first, or provide an anomaly threshold'
        if threshold is None:
            threshold = self.anomaly_threshold
            
        anomaly_scores = self.get_anomaly_score(x, batch_size, sum_datapoint_dimensions=single_label_for_each_observation, only_last_point_of_window=only_last_point_of_window)
        y_pred = (anomaly_scores > threshold).astype(int)
        return y_pred
        
    def determine_anomaly_threshold(self, x_train, batch_size, q=1e-3, level=0.98):
        """
        Automatically determine anomaly threshold using POT method calibrated with anomaly scores calculated on training set (calibration) and test set.
        
        Args:
            x_train: (array): KPI observations used to train the model, input of shape (n_samples, n_dimensions).
                Ideally, the training data contains only "normal" data without any anomalies, 
                and it covers the whole range of "normal" operation.
            q (float): Detection level, extreme quantile (risk), regulates the number of false positives.
            level (float): Probability associated with the initial threshold t. Should be adjusted for each different dataset.
                
        Returns:
            float: Determined anomaly threshold.
        """
        anomaly_scores = self.get_anomaly_score(x_train, batch_size=batch_size)
        self.anomaly_threshold = self.__find_anomaly_threshold_with_pot(anomaly_scores, q=q, level=level)
        return self.anomaly_threshold
        
    def fit(self, x, batch_size=None, validation_data=None, shuffle=False, seed=None, *args, **kwargs):
        """
        Fit the model using the given KPI observations.

        Usage when self.use_anomaly_labels_for_training=False (default):
            Input can be a 2d numpy array (n_samples, n_dimensions) in which case the fit method
            handles transforming it into (batch_size, window_length, input_dim) elements.
            Input can also be a `tf.data.Dataset` in which case the dataset should yield elements
            of the following type: (batch_size, window_length, input_dim)
            I.e., the dataset's element_spec must match:
            TensorSpec(shape=(None, None, input_dim), dtype=tf.float32, name=None)

        Usage when self.use_anomaly_labels_for_training=True:
            Use `self.use_anomaly_labels_for_training=True` when you want to train the model with additional
            anomaly labels, which indicate which data points should be ignored in the loss calculation.
            The labels (x_input_labels) should be: 
                0 = include in loss calculation (normal data), 
                1 = ignore in loss calculation (anomalies, missing data, etc...).

            If `self.use_anomaly_labels_for_training=True, the input data `x` and `validation_data` must be
            `tf.data.Dataset`. Both must yield the elements of the following type:

            {'encoder_input': inputs, 'x_input_labels': targets} where
            inputs shape is (batch_size, window_length, input_dim) and
            targets shape is (batch_size, window_length)

            i.e., the dataset's element_spec must match:
            {'encoder_input': TensorSpec(shape=(None, None, input_dim), dtype=tf.float32, name=None),
            'x_input_labels': TensorSpec(shape=(None, None), dtype=tf.float32, name=None)}

            We require specific keys for these two inputs to avoid the issue mentioned in:
            https://github.com/tensorflow/tensorflow/issues/34912#issuecomment-658238173
        
        Args:
            x (tf.data.Dataset or np.array): KPI observations, input of shape (n_samples, n_dimensions).
                If the input type is `tf.data.Dataset`, it should yield
                    elements with shape (batch_size, timesteps, input_dim) when use_anomaly_labels_for_training=False.
                    elements with shape {'encoder_input': inputs, 'x_input_labels': targets} when
                    use_anomaly_labels_for_training = True where
                        inputs shape is (batch_size, window_length, input_dim) and
                        targets shape is (batch_size, window_length)
            batch_size (int): Mini-batch size.
            validation_data (tf.data.Dataset or np.array, optional): KPI observations for validation.
                The type and shape should match the `x`.
            shuffle (bool, optional): Whether to shuffle the training data. Has only an effect when the input data is numpy array.
                When giving a `tf.data.Dataset` as an input, the caller must handle shuffling the dataset.
                Defaults to False.
            seed (int, optional): Random seed using when shuffling the training dataset. Defaults to None.
            args: Arguments to pass to Keras model.fit
            kwargs: Keyword arguments to pass to Keras model.fit

        Returns:
            A `History` object.
        """

        # validate input data
        if self.use_anomaly_labels_for_training:
            if not isinstance(x, tf.data.Dataset):
                raise TypeError(f"Argument x must be of type tf.data.Dataset when self.use_anomaly_labels_for_training=true, not {type(x)}")
            if validation_data is not None and not isinstance(validation_data, tf.data.Dataset):
                raise TypeError(f"Argument validation_data must be of type tf.data.Dataset when self.use_anomaly_labels_for_training=true, not {type(validation_data)}")
            
        if isinstance(x, tf.data.Dataset) and shuffle:
            print('Note that when given input x is already tf.data.Dataset, the shuffle argument has no effect.'
                  'You must shuffle the given tf.data.Dataset instances yourself.')
            
        if not isinstance(x, tf.data.Dataset) and len(x.shape) == 2:
            x = tf.cast(x, tf.keras.backend.floatx())
            # transform the input data into Dataset containing batches of windows of shape (batch_size, window_length, n_dimensions)
            x = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=x,
                targets=None,
                sequence_length=self.timesteps,
                sequence_stride=1, # sliding window
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed
            )

        if validation_data is not None and not isinstance(validation_data, tf.data.Dataset) and len(validation_data.shape) == 2:
            validation_data = tf.cast(validation_data, tf.keras.backend.floatx())
            # transform the input data into Dataset containing batches of windows of shape (batch_size, window_length, n_dimensions)
            validation_data = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=validation_data,
                targets=None,
                sequence_length=self.timesteps,
                sequence_stride=1, # sliding window
                batch_size=batch_size
            )

        history = self.model.fit(x, batch_size=batch_size, validation_data=validation_data, *args, **kwargs)
        
        # copy weights to prediction model
        self._model_pred.set_weights(self.model.get_weights())

        return history

    def predict(self, x, batch_size=None):
        if not isinstance(x, tf.data.Dataset):
            if len(x.shape) == 3:
                dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
            else:
                dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                    data=x,
                    targets=None,
                    sequence_length=self.timesteps,
                    sequence_stride=1, # sliding window
                    batch_size=batch_size
                )
            if self.use_anomaly_labels_for_training:
                dataset = dataset.map(lambda inputs:  {'encoder_input': inputs, 'x_input_labels': tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1]))})

        outputs = []
        for batch in dataset:
            p_x_given_z, _ = self.model(batch)
            x_pred = p_x_given_z
            if tf.rank(x_pred) == 4:
                x_pred = tf.reduce_mean(x_pred, axis=0) # avg over samples
            outputs.append(x_pred)
        x_pred = tf.concat(outputs, axis=0)
        return x_pred