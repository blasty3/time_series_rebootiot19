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
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import reparameterization

class RecurrentDistribution(distribution.Distribution):
    def __init__(self,
                 z_dim,
                 dense_dim,
                 input_q,
                 n_timesteps=100,
                 weight_regularizer=None,
                 dtype=tf.float32,
                 name='RecurrentDistribution'):
        """Construct RecurrentDistribution. This distribution represents 
        the encoder distribution (variational posterior) in the VAE, i.e. q(z | x).
        The distribution is needed to derive z_t using the concatenation of the latent input at 
        previous time step and current hidden input of the encoder [z_{t-1}, e_t].

        Args:
            z_dim (int): Latent space dimensionality.
            dense_dim (int): Number of hidden units in the dense layer.
            input_q (Tensor): Output of the encoder's RNN cell, shape of (batch_size, n_timesteps, rnn_num_hidden).
            n_timesteps (int, optional): Number of time steps. Defaults to 100.
            weight_regularizer (optional): Weight regularizer to use for the layers' weights.
            dtype (optional): Data type of the `Distribution`. Defaults to tf.float32.
            name (str, optional): Name of the `Distribution`. Defaults to 'RecurrentDistribution'.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(RecurrentDistribution, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=True,
            parameters=parameters,
            name=name)

            # transpose from (batch_size, n_timesteps, rnn_num_hidden) to (n_timesteps, batch_size, rnn_num_hidden)
            # (required by tf.scan since it wants to have the iterated dimension (n_timesteps) first)
            self.input_q = tf.transpose(input_q, [1, 0, 2])

            self.n_timesteps = n_timesteps
            self.z_dim = z_dim

            # dense layer h_ϕ with ReLU activation
            self.hidden_dense = Dense(dense_dim, activation='relu', name='encoder_hidden_dense', kernel_regularizer=weight_regularizer)
            # linear μ_[z_t]
            self.linear_mean = Dense(self.z_dim, name='z_mean', kernel_regularizer=weight_regularizer)
            # linear log(σ²_[z_t])
            self.linear_log_variance = Dense(self.z_dim, name='z_log_var', kernel_initializer='zeros', kernel_regularizer=weight_regularizer)
            # cache to store mapping z->log_prob(z) to avoid expensive log_prob calculation for sampled values
            self._intermediates_cache = {}

    def _batch_shape(self):
        # use [batch_size, n_timesteps] as the batch_shape
        return tf.TensorShape([None, self.n_timesteps])

    def _event_shape(self):
        return tf.TensorShape([self.z_dim])

    def _log_prob(self, z, group_event_dims=True):
        """Log probability of the given input.
        
        Args:
            z (Tensor): given samples.
            group_event_dims (bool): Whether to group the event dimension (latent dimension) of the final log probability output.
        Returns:
            log_prob (Tensor): Output of shape `[n_samples, batch_size, n_timesteps]`.
        """
        with tf.name_scope(name = 'log_prob'):
            if len(z.shape) > 3: # shape is (n_samples, batch_size, n_timesteps, z_dim)
                sample_batch_event_shape = tf.convert_to_tensor([tf.shape(z)[0], tf.shape(self.input_q)[1], self.z_dim]) # (n_samples, batch_size, z_dim)
                z = tf.transpose(z, [2, 0, 1, 3]) # now: (n_timesteps, n_samples, batch_size, n_dim)
            else: # shape is (batch_size, n_timesteps, z_dim)
                sample_batch_event_shape = tf.convert_to_tensor([tf.shape(self.input_q)[1], self.z_dim]) # (batch_size, z_dim)
                z = tf.transpose(z, [1, 0, 2])  # now: (n_timesteps, batch_size, n_dim)
                
            def log_prob_step(prev_state, inputs_at_t):
                """Calculate log probability for z_t

                Args:
                    _ (Tensor): previous log_prob (unused)
                    inputs_at_t ([Tensor, Tensor]): Contains (z_t, input_q), i.e. latent sample at time t and encoder RNN hidden output at time t.

                Returns:
                    log_prob (Tensor): Log probability of z_t.
                """
                _, z_previous = prev_state
                z_t, input_q_t = inputs_at_t
                # z_t shape is either (batch_size, n_dim) or (n_samples, batch_size, n_dim)
                if len(z_t.shape) > 2:
                    # broadcast input_q_t from (batch_size, rnn_num_hidden) to (n_samples, batch_size, rnn_num_hidden)
                    input_q_t = tf.broadcast_to(input_q_t, [tf.shape(z_t)[0], tf.shape(input_q_t)[0], input_q_t.shape[1]])

                # concatenate z_[t-1] and e_t, resulting in shape (n_samples, batch_size, rnn_num_hidden + z_dim) or (batch_size, rnn_num_hidden + z_dim)
                input_q_t = tf.concat([input_q_t, z_previous], axis = -1)

                # get dense layer h_ϕ output
                input_q_t = self.hidden_dense(input_q_t)

                # get μ_[z_t]
                z_mean = self.linear_mean(input_q_t) # (n_samples, batch_size, z_dim)

                # get log(σ²_[z_t])
                z_log_var = self.linear_log_variance(input_q_t) # (n_samples, batch_size, z_dim)

                log_prob_t = tfd.Normal(loc=z_mean, scale=tf.exp(0.5 * z_log_var)).log_prob(z_t)
                return log_prob_t, z_t

            log_prob = tf.scan(fn=log_prob_step,
                                elems=(z, self.input_q),
                                initializer=(tf.zeros(sample_batch_event_shape),
                                            tf.zeros(sample_batch_event_shape))
                            )[0] # shape (time_step, n_samples, batch_size, z_dim) or (time_step, batch_size, z_dim)
            if len(z.shape) > 3:
                log_prob = tf.transpose(log_prob, [1, 2, 0, 3]) # back to (n_samples, batch_size, n_timesteps, z_dim)
            else:
                log_prob = tf.transpose(log_prob, [1, 0, 2])  # back to (batch_size, n_timesteps, z_dim)
                
            if group_event_dims:
                log_prob=tf.reduce_sum(log_prob, axis=-1) # reduce to (n_samples, batch_size, n_timesteps) or (batch_size, n_timesteps) 

            return log_prob

    def _call_log_prob(self, value, name, **kwargs):
        # We override `_call_sample_n` so we can ensure that
        # the result of input is not prematurely modified (and thus caching works).

        # check if value is already computed in cache (stored during _sample_n)
        if (value.ref(), 'z->log_prob(z)') in self._intermediates_cache:
            log_prob = self._intermediates_cache.pop((value.ref(), 'z->log_prob(z)'))
            log_prob=tf.reduce_sum(log_prob, axis=-1) # reduce to (n_samples, batch_size, n_timesteps) or (batch_size, n_timesteps) 
            return log_prob

        # value was not cached, lets compute log_prob
        return super()._call_log_prob(value, name, **kwargs)

    def _call_sample_n(self, sample_shape, seed, name, **kwargs):
        # We override `_call_sample_n` so we can ensure that
        # the result of `self._sample_n` is not modified (and thus caching works).
        samples = super()._call_sample_n(sample_shape, seed, name, **kwargs)
        self._add_intermediate_to_cache(self._tmp_log_probs, samples.ref(), 'z->log_prob(z)')
        self._tmp_log_probs = None
        return samples

    def _sample_n(self, n, seed=None):
        """Samples from the recurrent distribution.
        
        Args:
            n (int): Number of samples desired.
            seed (int), seed for RNG. Setting a random seed enforces reproducability
                of the samples between sessions (not within a single session).
        Returns:
            samples (Tensor): Output of shape `[n, batch_size, n_timesteps, z_dim]`.
        """
        with tf.name_scope(name = 'sample_n'):
            n_samples = n
            sample_batch_event_shape = tf.convert_to_tensor([n_samples, tf.shape(self.input_q)[1], self.z_dim])  # (n_samples, batch_size, z_dim)
            
            def sample_step(prev_state, inputs_at_t):
                """Performs a single sample step based on the inputs at the time t.

                Args:
                    prev_state ([Tensor, Tensor, Tensor]): Contains (z_previous, mu_q_previous, log_var_previous), each of shape (n_samples, batch_size, z_dim)
                    inputs_at_t (Tensor): Contains input_q_t, i.e. encoder's RNN hidden output at time t, has shape (batch_size, rnn_num_hidden)

                Returns:
                    ([Tensor, Tensor, Tensor]): (z_t, z_mean, z_log_var), each of shape (n_samples, batch_size, z_dim)
                """
                z_previous, _, _ = prev_state
                input_q_t = inputs_at_t
                # broadcast from (batch_size, rnn_num_hidden) to (n_samples, batch_size, rnn_num_hidden)
                input_q_t = tf.broadcast_to(input_q_t, [tf.shape(z_previous)[0], tf.shape(input_q_t)[0], input_q_t.shape[1]])
                
                # concatenate z_[t-1] and e_t, resulting in shape (n_samples, batch_size, rnn_num_hidden + z_dim)
                input_q_t = tf.concat([input_q_t, z_previous], axis = -1)

                # get dense layer h_ϕ output
                input_q_t = self.hidden_dense(input_q_t)
                
                # get μ_[z_t]
                z_mean = self.linear_mean(input_q_t) # (n_samples, batch_size, z_dim)

                # get log(σ²_[z_t])
                z_log_var = self.linear_log_variance(input_q_t) # (n_samples, batch_size, z_dim)

                # get z_t using reparameterization trick
                epsilon = K.random_normal(shape=(n_samples, tf.shape(z_mean)[1], self.z_dim), seed=seed)
                z_t = z_mean + K.exp(0.5 * z_log_var) * epsilon

                return z_t, z_mean, z_log_var

            samples = tf.scan(fn=sample_step,
                elems=self.input_q,
                initializer=(tf.zeros(sample_batch_event_shape),
                            tf.zeros(sample_batch_event_shape),
                            tf.ones(sample_batch_event_shape))
                ) # array of three elements (z_t, z_mean, z_log_var), each has shape (time_step, n_samples, batch_size, z_dim)
                
            z_mean = samples[1]
            z_mean = tf.transpose(z_mean, [1, 2, 0, 3]) # switch to final order (n_samples, batch_size, n_timesteps, z_dim)
            z_log_var = samples[2]
            z_log_var = tf.transpose(z_log_var, [1, 2, 0, 3]) # switch to final order (n_samples, batch_size, n_timesteps, z_dim)

            samples = samples[0]
            samples = tf.transpose(samples, [1, 2, 0, 3])  # switch to final order (n_samples, batch_size, n_timesteps, z_dim)
            
            # calculate and store log_prob already here to avoid expensive log_prob loop calculation later
            log_probs = tfd.Normal(loc=z_mean, scale=tf.exp(0.5 * z_log_var)).log_prob(samples)
            self._tmp_log_probs = log_probs # temporarily store for caching
                
            return samples

    def _add_intermediate_to_cache(self, intermediate, y, name):
        """
        Internal function used to cache intermediate log_prob values computed during the sample call
        """
        assert((y, name) not in self._intermediates_cache),\
            "key collision in _add_intermediate_to_cache"
        self._intermediates_cache[(y, name)] = intermediate