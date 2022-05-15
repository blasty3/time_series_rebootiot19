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

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

class AnnealingCallback(Callback):
    def __init__(self, start, annealtime, weight):
        self.start = start
        self.annealtime = annealtime
        self.weight = weight
    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.start:
            new_weight = min(K.get_value(self.weight) + (1./ self.annealtime), 1.)
            K.set_value(self.weight, new_weight)
        print(" - KL weight: " + str(K.get_value(self.weight)))