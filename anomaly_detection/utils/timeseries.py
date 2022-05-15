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

def timeseries_dataset_from_series(data, targets, sequence_length, batch_size, exclude_target_value=None, shuffle=False, shuffle_buffer_size=None):
    """
    Creates a dataset of sliding windows over a timeseries provided as array.
    
    This function takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    length of the sequences/windows, spacing between two sequence/windows, etc.,
    to produce batches of timeseries inputs and targets.

    Similar function to `tf.keras.preprocessing.timeseries_dataset_from_array` when `sequence_stride=1`.
    This function allows additional features, such as filtering out sequences where a certain value
    exists in the targets of the sequence. This function is also used as part of the function
    `timeseries_dataset_from_multiple_series`.
    
    Args:
        data: Numpy array or eager tensor
            containing consecutive data points (timesteps).
            Axis 0 is expected to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have same length as `data`. `targets[i]` should be the target
            corresponding to the window that starts at index `i`.
            Pass None if you don't have target data (in this case the dataset will
            only yield the input data).
        sequence_length: Length of the output sequences (in number of timesteps).
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one which could be incomplete).
        exclude_target_value: All windows that contain this value in any of their targets
            will be filtered out. Useful for example if one wants to ignore windows that
            contain anomalous data and wants to keep only windows with normal data.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        shuffle_buffer_size: Size of the shuffle buffer (see Tensorflow documentation).
        
    Example 1:
        Consider input series `[0, 1, 2, ..., 30]` and targets `[0, -1, -2, ..., -30]`.
        With `sequence_length=5, batch_size=1, exclude_target_value=-5, shuffle=False`,
        the dataset yields the following sequences:
        
        ```
        First sequence:  [[0, 1, 2, 3, 4]], [[ 0, -1, -2, -3, -4]]
        Second sequence: [[ 6,  7,  8,  9, 10]], [[ -6,  -7,  -8,  -9, -10]]
        Third sequence:  [[ 7,  8,  9, 10, 11]], [[ -7,  -8,  -9, -10, -11]]
        ...
        Last sequence:   [[31, 32, 33, 34, 35]], [[-31, -32, -33, -34, -35]]
        ```
        We can see that between the first sequence and second sequence the sequences that
        contained the target value -5 were filtered out.
        
    Returns:
        A tf.data.Dataset instance. The dataset yields `batch_of_sequences`.
        Each `batch_of_sequences` has shape (batch_size, sequence_length, num_input_data_features).
    """
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(sequence_length, shift=1, drop_remainder=True) # shift=1 = sliding window of step 1
    ds = ds.flat_map(lambda w: w.batch(sequence_length))
    if targets is not None:
        target_ds = tf.data.Dataset.from_tensor_slices(targets)
        target_ds = target_ds.window(sequence_length, shift=1, drop_remainder=True)
        target_ds = target_ds.flat_map(lambda w: w.batch(sequence_length))
        ds = tf.data.Dataset.zip((ds, target_ds))
        if exclude_target_value is not None:
            def filter_fn(inputs, targets):
                # choose only those sequences (windows) that do not contain the excluded target value in their targets
                return tf.math.logical_not(tf.keras.backend.any(tf.equal(targets, exclude_target_value)))
            ds = ds.filter(filter_fn)
    if shuffle:
        if shuffle_buffer_size is None:
            shuffle_buffer_size = batch_size * 8 # same as tf.keras.preprocessing.timeseries_dataset_from_array
        ds = ds.shuffle(shuffle_buffer_size)
    if batch_size > 0:
        ds = ds.batch(batch_size)
    return ds

def timeseries_dataset_from_multiple_series(list_of_series, list_of_targets, sequence_length, batch_size, exclude_target_value=None, shuffle=False, batches_with_mixed_series=True):
    """
    Creates a dataset of sliding windows over a set of timeseries provided as array.
    
    This function should be used instead of `tf.keras.preprocessing.timeseries_dataset_from_array` when
    the collected timeseries data is not continuous, but consists of multiple separate series, which are
    continuous on their own, but have no continuity together.
    
    This function allows to produce batches of timeseries, where batches can consist of a mix of series
    from different time series (batches_with_mixed_series=True) or each batch can be chosen to contain
    only samples from a single (randomly chosen) series.
    
    Args:
        list_of_series (list of Numpy arrays): Where each series in the list contains 
            consecutive data points (timesteps). In each series, axis 0 is expected 
            to be the time dimension.
        list_of_targets (list of Numpy arrays): Targets of each series that correspond to the
            timesteps of each series. The length of `list_of_targets` should match
            `list_of_series` and each targets length in the list should match the length
            of the corresponding series in the `list_of_series`.
        sequence_length (int): Number of time steps / window size.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one which could be incomplete).
        exclude_target_value: All windows that contain this value in any of their targets
            will be filtered out. Useful for example if one wants to ignore windows that
            contain anomalous data and wants to keep only windows with normal data.
            See Example 1 in `timeseries_dataset_from_series` function.
        shuffle (bool): Whether to shuffle individual windows within each series or draw the series
            in chronological order within each series.
        batches_with_mixed_series (bool): Whether the batches should contain mixed sequences from 
            all of the series at random. If false, each batch contains sequences from a single series.
            
    
    Example 1:
        Consider input series `[1, 2, ..., 30]` and `[-1, -2, ..., -30]`, 
        i.e. [[1, 2, ..., 30], [-1, -2, ..., -30]]. With `batches_with_mixed_series=False`,
        `sequence_length=5, batch_size=2, shuffle=False`, the dataset could yield
        (because batches are randomly sampled from two input series):
        
        ```
        First batch:  [[-1 -2 -3 -4 -5][-2 -3 -4 -5 -6][-3 -4 -5 -6 -7]]
        Second batch: [[ 1  2  3  4  5][ 2  3  4  5  6][ 3  4  5  6  7]]
        Third batch:  [[ 4  5  6  7  8][ 5  6  7  8  9][ 6  7  8  9 10]]
        Fourth batch: [[ 7  8  9 10 11][ 8  9 10 11 12][ 9 10 11 12 13]]
        ...
        Last batch:   [[-25 -26 -27 -28 -29][-26 -27 -28 -29 -30]] (incomplete batch since 
        we ran out of sequences)
        
        As can be seen, the elements for each batch are randomly chosen from either input series
        [1, 2, ..., 30] or series [-1, -2, ..., -30], and because no shuffling is performed,
        these are in timely order within each series.
        
    Example 2:
        Consider input series `[1, 2, ..., 30]` and `[-1, -2, ..., -30]`, 
        i.e. [[1, 2, ..., 30], [-1, -2, ..., -30]]. With `batches_with_mixed_series=True`,
        `sequence_length=5, batch_size=2, shuffle=False`, the dataset could yield
        (because batches are randomly sampled from two input series):
        
        ```
        First batch:  [[ 1  2  3  4  5][ 2  3  4  5  6][-1 -2 -3 -4 -5]]
        Second batch: [[-2 -3 -4 -5 -6][-3 -4 -5 -6 -7][-4 -5 -6 -7 -8]]
        Third batch:  [[-5 -6 -7 -8 -9][ 3  4  5  6  7][-6 -7 -8 -9 -10]]
        Fourth batch: [[ 4  5  6  7  8][-7 -8 -9 -10 -11][-8 -9 -10 -11 -12]]
        ...
        Last batch:   [[26 27 28 29 30]] (incomplete batch since we ran out of sequences)
        
        As can be seen, the elements for each batch are randomly chosen from all input series
        ([1, 2, ..., 30] and [-1, -2, ..., -30]), and because no shuffling is performed,
        these are in timely order within the batches.
    """
    if batches_with_mixed_series:
        all_ds = []
        for idx, series in enumerate(list_of_series):
            targets = None
            if list_of_targets is not None:
                targets = list_of_targets[idx]
            # do not perform batching yet
            ds = timeseries_dataset_from_series(series, targets, sequence_length, batch_size=0, exclude_target_value=exclude_target_value, shuffle=shuffle, shuffle_buffer_size=len(series))
            all_ds.append(ds)
        ds = tf.data.experimental.sample_from_datasets(all_ds)
        # perform batching now, so that batches can contain a mix of different series
        ds = ds.batch(batch_size)
        return ds
    else:
        all_ds = []
        for idx, series in enumerate(list_of_series):
            targets = None
            if list_of_targets is not None:
                targets = list_of_targets[idx]
            ds = timeseries_dataset_from_series(series, targets, sequence_length, batch_size, exclude_target_value=exclude_target_value, shuffle=shuffle)
            all_ds.append(ds)
        ds = tf.data.experimental.sample_from_datasets(all_ds)
        return ds