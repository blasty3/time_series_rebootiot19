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
import math

eps = 1e-7

def get_perf_metrics(y_true, y_pred, pos_label=1, adjust_predicted_labels=True, single_score_per_labeled_sequence=True):
    """
    Computes several performance metrics based on the ground truth labels and predicted labels.

    Args:
        y_true (1d np.ndarray): Ground truth (correct) labels.
        y_pred (1d np.ndarray): Predicted labels.
        adjust_predicted_labels (bool): Whether to adjust predicted labels
            If detected an anomalous point in a true anomalous continuous segment
            mark all the points of this segment as correctly predicted anomalies.
        single_score_per_labeled_sequence (bool): Whether to count only one TP for a 
            correctly detected labeled sequence, even if multiple TP points would occur
            in the continuous labeled sequence, and one FN for undetected labeled sequence.
            
            Following the idea from
            "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
            https://arxiv.org/pdf/1802.04431.pdf
        
    Returns:
        report_dict (dict): A dict of performance metrics (F1 score, precision, recall, TP, TN, FP, FN).
    """
    if adjust_predicted_labels:
        y_pred, detection_latency = get_adjusted_predicted_labels(y_true, y_pred)
    if not single_score_per_labeled_sequence:
        TP = np.sum(np.logical_and(y_pred == pos_label, y_true == pos_label))
        FN = np.sum(np.logical_and(y_pred != pos_label, y_true == pos_label))
    else:
        assert adjust_predicted_labels == True, 'When single_score_per_labeled_sequence=True' \
                                                'adjust_predicted_labels must be True'
        TP, FN = count_true_positives_as_sequences(y_true, y_pred, pos_label)
    TN = np.sum(np.logical_and(y_pred != pos_label, y_true != pos_label))
    FP = np.sum(np.logical_and(y_pred == pos_label, y_true != pos_label))
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    report_dict = {
        'f1-score': f1,
        'precision': precision,
        'recall': recall,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }
    if adjust_predicted_labels:
        report_dict['detection_latency'] = detection_latency
    return report_dict

def count_true_positives_as_sequences(y_true, y_pred, pos_label=1):
    """
    Counts true positives such that a TP results if any portion of a 
    predicted sequence falls within any true labeled sequence. Only one 
    TP is recorded even if multiple points are predicted within a labeled sequence.
    Counts false negatives such that a FN results if no anomalies are detected
    within any portion of a true labeled anomalous sequence. Only one false negative
    is recorded for each undetected labeled sequence.
    
    For example:
    
    y_true
        [0,0,0,0,0,1,1,1,1,0,1,0,0,1,1,0,0]
    y_pred
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0]
    then this function returns TP = 2 (out of 3 true labeled sequences)
    and FN = 1 (one sequence missed)
    """
    TP = 0
    FN = 0
    segments = np.flatnonzero(np.diff(y_true == 1)) # find indices where y label changes
    segments = np.concatenate([[0], segments + 1, [-1]])  # add 0 index and -1 end index and adjust indices
    for start_idx, end_idx in zip(segments[:-1], segments[1:]):  # iterate over each continuous segment
        if y_true[start_idx] == pos_label: # ensure that this is a beginning of an anomalous segment
            if y_pred[start_idx] == pos_label:
                TP += 1  # this segment was correctly detected
            else:
                FN += 1 # this segment was not detected
    return TP, FN

def get_adjusted_predicted_labels(y_true, y_pred):
    """
    Gets adjusted predicted labels using given ground truth labels and predicted labels.
    If any predicted label correctly appears inside a true anomalous segment, mark all the predictions as
    correct in this continuous segment.
    
    Following the idea from several papers:
    https://dl.acm.org/doi/pdf/10.1145/3292500.3330672 
    https://arxiv.org/pdf/1802.04431.pdf 
    https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56.pdf
    https://arxiv.org/pdf/1910.03818.pdf
    
    Args:
        y_true (1d np.ndarray): Ground truth (correct) labels.
        y_pred (1d np.ndarray): Predicted labels.
            
    Returns:
        tuple (y_pred, anomaly_detection_latency): where
            y_pred (np.ndarray): Adjusted predicted labels.
            anomaly_detection_latency (float): Average anomaly detection latency. On average, 
            how many true anomalous data points were missed in a detected anomalous segment.
    
    For example:
    
    y_true
        [0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0]
    y_pred
        [0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0]
    then the final adjusted y_pred is
        [0,0,0,0,0,1,1,1,1,0,1,0,0,1,1,0,0]
    """
    anomaly_detection_latency = 0
    detected_anomaly_count = 0
    # iterate over each true continuous anomalous segment
    segments = np.flatnonzero(np.diff(y_true == 1)) # find indices where y label changes
    segments = np.concatenate([[0], segments + 1, [-1]])  # add 0 index and -1 end index and adjust indices
    for start_idx, end_idx in zip(segments[:-1], segments[1:]):  # iterate over each continuous segment
        if y_true[start_idx]: # ensure that this is a beginning of an anomalous segment
            # get predicted labels on this segment
            y_pred_in_segment = y_pred[start_idx:end_idx]
            # only proceed if at least single point in this segment was correctly identified as an anomaly
            if (np.any(y_pred_in_segment)):
                detected_anomaly_count += 1
                anomaly_in_segment_detected = False # used to calculate the detection latency
                for i in range(start_idx, end_idx):
                    if y_pred[i]:
                        anomaly_in_segment_detected = True
                    else:
                        y_pred[i] = True
                        if not anomaly_in_segment_detected:
                            anomaly_detection_latency += 1
                            
    return y_pred, anomaly_detection_latency / (detected_anomaly_count + eps)

def get_perf_metrics_for_anomaly_scores(y_true, anomaly_scores, anomaly_threshold, single_score_per_labeled_sequence=True):
    """
    Computes performance metrics using the given ground truth labels, anomaly scores, and anomaly threshold.
    
    Args:
        y_true (1d np.ndarray): Ground truth (correct) labels.
        anomaly_scores (1d np.ndarray): Array of anomaly scores for each data point.
        anomaly_threshold (float): The threshold of anomaly score.
            A point is labeled as anomalous if its score is higher than the threshold.
        single_score_per_labeled_sequence (bool): Whether to count only one TP for a 
            correctly detected labeled sequence, even if multiple TP points would occur
            in the continuous labeled sequence. Following the idea from
            "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
            https://arxiv.org/pdf/1802.04431.pdf
            
    Returns:
        report_dict: A dict of performance metrics.
    """
    y_true = np.asarray(y_true)
    anomaly_scores = np.asarray(anomaly_scores)

    # Handle possible NaN values so that y_pred will contain also NaN values at those indices to keep the
    # reported metrics correct. NaN values will occur in output anomaly scores of certain algorithms
    # (OmniAnomaly, VAR, etc...) which cannot provide anomaly scores for the first window_length-1 values
    # because they use sliding windowing method where they need the first window_length-1 values 
    # to predict the anomaly score for window_length+1, hence being unable to score the the initial values
    nan_idxs = np.isnan(anomaly_scores)
    anomaly_scores[nan_idxs] = 0

    y_pred = anomaly_scores > anomaly_threshold

    y_pred[nan_idxs] = np.nan

    report_dict = get_perf_metrics(y_true, y_pred, single_score_per_labeled_sequence=single_score_per_labeled_sequence)
    return report_dict

def find_best_anomaly_threshold(y_true, anomaly_scores, start, end, metric='f1-score', step_size=1, verbose=False, verbose_freq=1):
    """
    Finds the best anomaly threshold based on the chosen metric by testing different
    anomaly score thresholds in range [start, end] with the given step size.
    
    Args:
        y_true (1d np.ndarray): Ground-truth labels.
        anomaly_scores (1d np.ndarray): Array of anomaly scores for each data point.
        start (integer): Start of the threshold search range.
        end (integer): End of the threshold search range.
        metric (string): The metric to use to choose the optimal anomaly threshold.
        step_size (integer): Step size of the threshold search.
        verbose (bool): Whether to print intermediate results.
        verbose_freq (integer): Print intermediate results at every nth step.

    Returns:
        tuple (best_metrics, best_threshold): where
            best_metrics (dict): A dict of performance metrics based on the optimal threshold
            best_threshold (float): the optimal anomaly threshold based on the chosen metric
    """
    if verbose:
        print(f'Threshold search range: [{start}, {end}], step: {step_size}')
    num_steps = int(abs(end - start) / step_size)
    best_metrics = {}
    best_metrics[metric] = -1
    best_threshold = 0.0
    threshold = start
    for i in range(num_steps):
        perf_metrics = get_perf_metrics_for_anomaly_scores(y_true, anomaly_scores, threshold)
        if perf_metrics[metric] > best_metrics[metric]:
            best_threshold = threshold
            best_metrics = perf_metrics
        if verbose and i % verbose_freq == 0:
            print('-----------------')
            print('Testing threshold: ', threshold, perf_metrics)
            print('Best threshold: ', best_threshold, best_metrics)
        threshold += step_size
    return best_metrics, best_threshold

from scipy import optimize
def best_threshold_function_to_minimize(anomaly_threshold, *params):
    """Function used in `find_best_anomaly_threshold_v2` by 
    `scipy.optimize.brute` to perform brute force minimization.
    When this function is minimized, one can find the best anomaly
    threshold for the given metric.

    Args:
        anomaly_threshold (float): Anomaly threshold to test.

    Returns:
        float: Negative performance metric (used to minimize)
    """
    y_true, anomaly_scores, metric, single_score_per_labeled_sequence = params
    perf_metrics = get_perf_metrics_for_anomaly_scores(y_true, anomaly_scores, anomaly_threshold, single_score_per_labeled_sequence=single_score_per_labeled_sequence)
    return -perf_metrics[metric] # negative since this function is being minimized

def find_best_anomaly_threshold_v2(y_true, anomaly_scores, start, end, metric='f1-score', step_size=1, verbose=False, workers=1, single_score_per_labeled_sequence=True):
    """
    Faster version of `find_best_anomaly_threshold` (uses `scipy.optimize.brute`).
    Finds the best anomaly threshold based on the chosen metric by testing different
    anomaly score thresholds in range [start, end] with the given step size.
    Uses `scipy.optimize.brute` to perform the brute force search. Supports parallelism.
    
    Args:
        y_true (1d np.ndarray): Ground-truth labels.
        anomaly_scores (1d np.ndarray): Array of anomaly scores for each data point.
        start (integer): Start of the threshold search range.
        end (integer): End of the threshold search range.
        metric (string): The metric to use to choose the optimal anomaly threshold.
        step_size (float): Step size of the threshold search.
        verbose (bool): Whether to print intermediate results.
        workers (integer): Number of parallellel processes.
            Supply -1 to use all cores available to the Process. 

    Returns:
        tuple (best_metrics, best_threshold): where
            best_metrics (dict): A dict of performance metrics based on the optimal threshold
            best_threshold (float): the optimal anomaly threshold based on the chosen metric
    """
    ranges = (slice(start, end, step_size),)
    resbrute = optimize.brute(best_threshold_function_to_minimize, ranges, args=(y_true, anomaly_scores, metric, single_score_per_labeled_sequence), finish=optimize.fmin, workers=workers, disp=verbose)
    best_threshold = resbrute[0]
    best_metrics = get_perf_metrics_for_anomaly_scores(y_true, anomaly_scores, best_threshold, single_score_per_labeled_sequence)
    return best_metrics, best_threshold

def find_best_f1_score(y_true, 
                       anomaly_scores, 
                       threshold_search_start=None, 
                       threshold_search_end=None, 
                       threshold_search_step_size=0.1,
                       skip_n_first=0,
                       single_score_per_labeled_sequence=True,
                       verbose=True):
    """Find best f1 score.

    Args:
        y_true (1d np.ndarray): Ground-truth labels.
        anomaly_scores (1d np.ndarray): Array of anomaly scores for each data point.
        threshold_search_start (int, optional): Start of the threshold search range. Defaults to None.
        threshold_search_end (int, optional): nd of the threshold search range. Defaults to None.
        threshold_search_step_size (float, optional): Step size of the threshold search. Defaults to 0.1.
        skip_n_first=0 (int, optional): Ignore first n points in performance calculation.
            Useful to fairly compare algorithms where some cannot make predictions for the initial window.
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        float: Best anomaly threshold.
    """
    if threshold_search_start is None:
        threshold_search_start = np.min(anomaly_scores)
    if threshold_search_end is None:
        threshold_search_end = np.max(anomaly_scores)
    if threshold_search_step_size is None:
        threshold_search_step_size = abs(threshold_search_end / threshold_search_start)
        
    # some algorithms use windowing method and make no predictions for the first `offset` values
    # therefore `scores` array might be shorter than `y_true`
    # this makes the `scores` have same length as `y_true` by padding nans in front of `scores`
    offset = len(y_true) - len(anomaly_scores)
    if offset > 0:
        # add nans to the front of scores array to make it same length as y_true
        anomaly_scores = np.concatenate([np.empty(offset) * np.nan, anomaly_scores])

    if skip_n_first > 0:
        # skip first `skip_n_first` points
        y_true = y_true[skip_n_first:-1]
        anomaly_scores = anomaly_scores[skip_n_first:-1]
        
    # find best threshold
    t, th = find_best_anomaly_threshold_v2(y_true, anomaly_scores,
                      start=threshold_search_start,
                      end=threshold_search_end,
                      metric='f1-score',
                      verbose=False,
                      step_size=threshold_search_step_size,
                      workers=-1,
                      single_score_per_labeled_sequence=single_score_per_labeled_sequence)
    if verbose:
        print('t', t)
        print('th', th)
    return th

def get_ranked_root_causes_of_anomaly(idx, anomaly_scores_each_dim):
    """
    Get ranked root causes of the anomaly at the given index.
    
    Args:
        idx (int): Anomaly score index for which to retrieve the ranked root causes.
        anomaly_scores_each_dim (np.ndarray): Anomaly scores for each feature of each
            sample, shape (n_samples, n_features).
            
    Example 1:
        Consider anomaly_scores_each_dim = [
            [-20, 10, -4, 30, 50],
            [20,   5,  2,  4,  7],
            [4,   30, 99,  5,  8],
            [1,   2,   5,  3,  7],
            [8,   2,   1,  20, 8],
            [0,   5,   -2, -4, 2]
        ], i.e., we have anomaly scores for each of the 5 features for 6 samples.
        
        Calling the function with `idx=0` would return
        [4, 3, 1, 2, 0] so the top 3 features contributing to the anomaly
        are 4, 3, 1 (in that specific order).
        Similarly, calling the function with `idx=1` would return
        [0, 4, 1, 3, 2].
            
    Returns:
        np.ndarray: Ranked feature indices by their contribution to the anomaly score.
    """
    # anomaly scores for each feature of the sample at the given data point index
    anomaly_scores_at_data_index = np.squeeze(anomaly_scores_each_dim[idx])
    # sort in descending order, returning the feature indices in the sorted order
    sorted_anomaly_scores_at_data_index = np.argsort(anomaly_scores_at_data_index)[::-1]
    return sorted_anomaly_scores_at_data_index

def get_anomaly_root_cause_hitrate_perf(y_true, y_pred, feature_scores, y_true_root_causes, p=100):
    """
    Get hitrate performance metric by checking how well the predicted feature scores can
    explain the true root causes of the detected true anomalies.

    Note that this function only calculates the hitrates of the predicted anomalies that
    were correctly detected (true positives (TP)), thus, for example, predicting only one sample
    as TP and having a lot of FPs, but having perfect hitrate score on that TP based on the 
    feature scores, this function would return a perfect hitrate of 1.0.
    
    Args:
        y_true (np.ndarray) 1d array of true anomaly labels (0,1) where 1 = anomaly.
        y_pred (np.ndarray) 1d array of predicted anomaly labels (0,1) where 1 = anomaly.
        feature_scores (np.ndarray): Anomaly scores of each feature of each sample,
            shape (n_samples, n_features)
        y_true_root_causes (dict): where key is the sample index of y_true array and
            the value is a 1d np.ndarray of feature indices that indicate the root causes
            of that anomaly.
        p (int): Hit rate percentage. Defaults to 100.
        
    Following the HitRate@P% method described in OmniAnomaly paper.
            
    Example 1:
        y_true: [0,1,0,1,0,1]
        y_pred: [0,1,0,0,0,1]
        feature_scores: [[0,0,0,0,0], [4,22,99,5,70],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[10,9,0,0,0]]
        y_true_root_causes: {
            1: [2,4,1]
            3: [0,1]
            5: [0,1,2,3]
        }
        
        The function would return (average) hitrate score of 0.875,
        since the first detected anomaly (sample 1) has perfect hitrate 1.0 (based on the ranked
        feature scores) and the second detected anomaly (sample 5) has hitrate of 0.75.
        
    Returns:
        float: Average HitRate@P%.
    """
    
    # some algorithms use windowing method and make no predictions for the first `offset` values
    # therefore `y_pred` array might be shorter than `y_true`
    # this makes the `y_pred` have same length as `y_true` by padding -1's in front of `y_pred`
    offset = len(y_true) - len(y_pred)
    if offset > 0:
        # add -1's to the front of scores array to make it same length as y_true
        y_pred = np.concatenate([np.ones(offset) * -1, y_pred]).astype(int)

    # Some algorithms use windowing method and make no predictions for the first `offset`
    # values. Therefore `feature_scores` array might be shorter than `y_true`. This makes 
    # the `feature_scores` have same length as `y_true` by padding nans in front of `feature_scores`
    offset = len(y_true) - len(feature_scores)
    if offset > 0:
        # add nans to the front of feature_scores array to make it same length as y_true
        feature_scores = np.concatenate([np.empty([offset, feature_scores.shape[1]]) * np.nan, feature_scores])
    
    hit_rates = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_pred[i] == y_true[i]: # true positive (true detected anomaly)
            ground_truth_root_causes = y_true_root_causes[i]
            ranked_dim_root_causes = get_ranked_root_causes_of_anomaly(i, feature_scores)
            hitrate = hit_rate(ranked_dim_root_causes, ground_truth_root_causes, p=p)
            hit_rates.append(hitrate)
    return np.mean(hit_rates)

def get_anomaly_root_cause_hitrate_perf_v2(y_true, y_pred, anomaly_scores, feature_scores, y_true_root_causes, p=100, skip_n_first=0):
    """
    Get hitrate performance metric by checking how well the predicted feature scores can
    explain the true root causes of the detected true anomalies.

    Note that this function only calculates the hitrates of the predicted anomalies that
    were correctly detected (true positives (TP)), thus, for example, predicting only one sample
    as TP and having a lot of FPs, but having perfect hitrate score on that TP based on the 
    feature scores, this function would return a perfect hitrate of 1.0.
    
    Args:
        y_true (np.ndarray) 1d array of true anomaly labels (0,1) where 1 = anomaly.
        y_pred (np.ndarray) 1d array of predicted anomaly labels (0,1) where 1 = anomaly.
        feature_scores (np.ndarray): Anomaly scores of each feature of each sample,
            shape (n_samples, n_features)
        y_true_root_causes (dict): where key is the sample index of y_true array and
            the value is a 1d np.ndarray of feature indices that indicate the root causes
            of that anomaly.
        p (int): Hit rate percentage. Defaults to 100.
        skip_n_first (int): Ignore n first points from the calculation.
        
    Follows the HitRate@P% method described in OmniAnomaly paper.
            
    Example 1:
        y_true: [0,1,0,1,0,1]
        y_pred: [0,1,0,0,0,1]
        anomaly_scores: [0, 200, 0, 0, 0, 19]
        feature_scores: [[0,0,0,0,0], [4,22,99,5,70],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[10,9,0,0,0]]
        y_true_root_causes: {
            1: [2,4,1],
            3: [0,1],
            5: [0,1,2,3]
        }
        
        The function would return (average) hitrate score of 0.875,
        since the first detected anomaly (sample 1) has perfect hitrate 1.0 (based on the ranked
        feature scores) and the second detected anomaly (sample 5) has hitrate of 0.75.
        
    Example 2:
        y_true: [0,1,0,1,1,1]
        y_pred: [0,1,0,1,0,1]
        anomaly_scores: [0, 200, 0, 46, 0, 19]
        feature_scores: [[0,0,0,0,0], [4,22,99,5,70],[0,0,0,0,0],[15,11,20,0,0],[0,0,0,0,0],[10,9,0,0,0]]
        y_true_root_causes: {
            1: [2,4,1],        _
            3: [0,1,2],         |
            4: [0,1,2],         |--> these 3 are duplicated since they describe the same anomalous segment of length 3
            5: [0,1,2]         _|
        }
        
        There are two true anomalous segments, [1] and [3, 4, 5] in y_true.
        The function would return (average) hitrate score of 1.0 (perfect),
        since the first detected anomaly (sample 1) has perfect hitrate 1.0 (based on the ranked
        feature scores) and the second detected anomaly (sample 5) has hitrate of 1.0 as well.
        The hitrate of the second anomaly ([3, 4, 5]) is calculated as follows,
            * The method first checks if there is the anomaly was detected at all (any y_pred=1 within the y_true segment)
            => there is since y_pred = [1, 0, 1] at this y_true segment [1, 1, 1]
            * Select the point within this segment with the highest anomaly score => point at index 0 (of segment) with score 46.
            * Use the feature scores of this point to calculate the hitrate, i.e., the scores [15,11,20,0,0].
        
    Returns:
        float: Average HitRate@P%.
    """
    
    # some algorithms use windowing method and make no predictions for the first `offset` values
    # therefore `y_pred` array might be shorter than `y_true`
    # this makes the `y_pred` have same length as `y_true` by padding -1's in front of `y_pred`
    offset = len(y_true) - len(y_pred)
    if offset > 0:
        # add -1's to the front of y_pred array to make it same length as y_true
        y_pred = np.concatenate([np.ones(offset) * -1, y_pred]).astype(int)
        # add nans to the front of anomaly_scores array to make it same length as y_true
        anomaly_scores = np.concatenate([np.empty(offset) * np.nan, anomaly_scores])
        # add nans to the front of feature_scores array to make it same length as y_true
        feature_scores = np.concatenate([np.empty([offset, feature_scores.shape[1]]) * np.nan, feature_scores])

    if skip_n_first > 0:
        # skip first `skip_n_first` points
        y_true = y_true[skip_n_first:-1]
        anomaly_scores = anomaly_scores[skip_n_first:-1]
        feature_scores = feature_scores[skip_n_first:-1]
        
    # store each hit rate score in array
    hit_rates = []
        
    true_segments = np.flatnonzero(np.diff(y_true == 1)) # find indices where y label changes
    true_segments = np.concatenate([[0], true_segments + 1, [-1]])  # add 0 index and -1 end index and adjust indices
    for start_idx, end_idx in zip(true_segments[:-1], true_segments[1:]):  # iterate over each continuous segment
        if y_true[start_idx]: # ensure that this is a beginning of an anomalous segment
            # get predicted labels on this segment
            y_pred_in_segment = y_pred[start_idx:end_idx]
            # only proceed if at least single point in this segment was correctly identified as an anomaly
            if (np.any(y_pred_in_segment == 1)):
                # find the point within this segment where the anomaly score is the highest
                highest_anomaly_idx = np.argmax(anomaly_scores[start_idx:end_idx])
                # get the true root causes of this true anomalous segment
                ground_truth_root_causes = y_true_root_causes[start_idx]
                # get predicted root causes at this index
                ranked_dim_root_causes = get_ranked_root_causes_of_anomaly(highest_anomaly_idx, feature_scores[start_idx:end_idx])
                # get hit rate score
                hitrate = hit_rate(ranked_dim_root_causes, ground_truth_root_causes, p=p)
                hit_rates.append(hitrate)
                
    return np.mean(hit_rates)

def hit_rate(ranked_features, root_causes_ground_truth, p=100):
    """Returns the HitRate@P% as described in the OmniAnomaly paper to score how well the ranked feature scores
    match to the true root cause features of the anomaly.

    Args:
        ranked_features (np.ndarray): 1d array of features ranked by their contribution to the anomaly.
        root_causes_ground_truth (np.ndarray): 1d array of features that are the real root cause of the anomaly.
        p (int, optional): The chosen hitrate percentage. Defaults to 100, i.e., HitRate@P100%. Use either 100 or 150.

    Returns:
        float: HitRate@P%.
    """
    hit_rate = p / 100
    num_overlapping = np.count_nonzero(np.isin(ranked_features[:math.floor(len(root_causes_ground_truth) * hit_rate)], root_causes_ground_truth))
    return num_overlapping / len(root_causes_ground_truth)