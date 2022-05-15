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

class VAEComputeLoss(tf.keras.layers.Layer):
    """
    Placeholder layer to calculate VAE loss (-ELBO) with adjustable weight on the KL divergence loss.
    """
    def __init__(self, kl_weight):
        super(VAEComputeLoss, self).__init__(name='vae_loss_layer')
        self.kl_weight = kl_weight
        
    def call(self, inputs):
        """Compute VAE loss (-ELBO) from the input losses = -(log likelihood - KL divergence).

        Args:
            inputs (tuple): Contains (log likelihood, kl_loss), both of shape (batch_size, n_timesteps).
                Note: Log-likelihood, not negative log-likelihood.

        Returns:
            vae_loss (tf.Tensor): Final loss, shape ().
        """
        likelihood_log_prob, kl_loss = inputs
        # each loss from shape (batch_size, n_timesteps) to (), i.e. single value for whole batch
        # mean over timesteps and batches
        elbo = tf.reduce_mean(likelihood_log_prob - kl_loss * self.kl_weight)
        return -elbo