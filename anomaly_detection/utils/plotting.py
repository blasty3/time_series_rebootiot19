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

def plot_anomaly_scores(scores, y_true, threshold=None, ylim=None, yscale='linear', figsize=(16,5)):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # some algorithms use windowing method and make no predictions for the first `offset` values
    # therefore `scores` array might be shorter than `y_true` - make `scores` same length by padding nans
    offset = len(y_true) - len(scores)
    if offset > 0:
        # add nans to the front of scores array to make it same length as y_true
        scores = np.concatenate([np.empty(offset) * np.nan, scores])
    
    # plot scores
    plt.plot(np.arange(len(y_true)), scores)
            
    if ylim is not None:
        plt.ylim(ylim)
        
    plt.yscale(yscale)
    
    def groupby_consecutive_segments(data, stepsize=1):
        """
        For example, given a = np.array([0, 47, 48, 49, 50, 97, 98, 99])
        consecutive(a)

        yields
        [array([0]), array([47, 48, 49, 50]), array([97, 98, 99])]
        """
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

    # plot anomaly segments by vertical spanning areas
    anomalous_indices = np.argwhere(y_true == 1).flatten()
    if len(anomalous_indices) > 0:
        anomalous_segments = groupby_consecutive_segments(anomalous_indices)
        i = 0
        for segment in anomalous_segments:
            ax.axvspan(segment[0], segment[-1], alpha=0.2, color='red', linewidth=2, zorder=4, label = '_' * i + 'True anomaly')
            i += 1
            
    ax_t = ax.twiny()
    ax_t.tick_params(width=2, color='#FFCCCC')
    ax_t.set_xticklabels([])
    ax_t.set_xticks(anomalous_indices)
    ax_t.set_xlim(ax.get_xlim())

    ax.set_xlabel('Cycle index')
    ax.set_ylabel('Anomaly score')
    
    # plot anomaly threshold line if given
    if threshold is not None:
        plt.axhline(threshold, c='red', linestyle='--', label='Threshold - (F1 best)')
        plt.legend(loc='upper right', facecolor='white', framealpha=0.95, bbox_to_anchor=(1.0, 0.97), shadow=False)
    plt.show()