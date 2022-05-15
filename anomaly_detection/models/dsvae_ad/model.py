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
from tensorflow import keras

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, GRU, SimpleRNN, RNN, GRUCell, LSTMCell, SimpleRNNCell
from tensorflow.keras.layers import Dense, Lambda, GaussianNoise, Bidirectional
from tensorflow.keras import losses

from anomaly_detection.utils.vae_utils import VAEComputeLoss
from .recurrent_distribution import RecurrentDistribution
from .planar_flows import PlanarFlows
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tf.linalg

def build_dynamic_encoder(
        x,
        x_,
        timesteps,
        rnn_num_hidden,
        dense_dim, 
        latent_dim,
        nf_layers,
        allow_cudnn_kernel,
        use_connected_z_q,
        weight_regularizer,
        n_samples,
        rnn,
        rnn_cell,
        bidirectional):
    if allow_cudnn_kernel:
        e_rnn_layer = rnn(rnn_num_hidden, return_sequences=True)
        if bidirectional:
            e_rnn_layer = Bidirectional(e_rnn_layer)
        e_hidden = e_rnn_layer(x_)  # e_{t-T} to e_t
    else:
        e_rnn_layer = RNN(rnn_cell(rnn_num_hidden), return_sequences=True)
        if bidirectional:
            e_rnn_layer = Bidirectional(e_rnn_layer)
        e_hidden = e_rnn_layer(x_)  # e_{t-T} to e_t

    if use_connected_z_q:
        q_z0_given_x = RecurrentDistribution(latent_dim, dense_dim, e_hidden, timesteps, weight_regularizer=weight_regularizer)
    else:
        z_mean = TimeDistributed(Dense(latent_dim, name='z_mean'))(e_hidden)
        z_log_var = TimeDistributed(Dense(latent_dim, name='z_log_var'))(e_hidden)
        q_z0_given_x = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=tf.exp(0.5 * z_log_var)) # approximated posterior distribution q(z_{t-T:t} | x_{t-T:t})

    # sample z0 from q(z0 | x) ~ Normal(mu(x;phi), sigma(x;phi))
    z0 = q_z0_given_x.sample(n_samples)
    
    if nf_layers > 0:
        # planar normalizing flow (transform simple diagonal Gaussian distribution into more complex posterior approximation)
        # q(z0 | x) ~ Normal(mu(x;phi), sigma(x;phi))
        # zK = bijector(z0), i.e. distribution for q(zK | x)
        q_z_given_x = PlanarFlows(latent_dim, q_z0_given_x, nf_layers) # final approximated posterior distribution q(z | x)
        zK = q_z_given_x(z0)
        z = zK
    else:
        # do not use normalizing flows, use the plain diagonal Gaussian as the posterior q(z | x)
        q_z_given_x = q_z0_given_x
        z = z0

    encoder = Model(x, [z], name='encoder')
    return encoder, q_z_given_x, z

def build_dynamic_prior(timesteps, latent_dim, use_connected_z_p):
    if use_connected_z_p:
        # Use a Linear Gaussian State Space Model (SSM), also known as Kalman filter, as a prior p(z)
        # By using the SSM as a prior information we can model how the latent variables evolve over time (using our state space model)
        # and we can force the latent space variables z_[t-T] to z_t to be temporally dependent according to the linear gaussian SSM.
        p_z = tfd.LinearGaussianStateSpaceModel(
            num_timesteps=timesteps,
            transition_matrix=tfl.LinearOperatorIdentity(latent_dim),
            transition_noise=tfd.MultivariateNormalDiag(
                scale_diag=tf.ones([latent_dim])
            ),
            observation_matrix=tfl.LinearOperatorIdentity(latent_dim),
            observation_noise=tfd.MultivariateNormalDiag(
                scale_diag=tf.ones([latent_dim])
            ),
            initial_state_prior=tfd.MultivariateNormalDiag(
                scale_diag=tf.ones([latent_dim])
            ),
            initial_step=tf.convert_to_tensor(0) # this is here only to get rid of a deprecation warning
        )
    else:
        p_z = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim), name='prior')
    return p_z

def build_static_encoder(x, latent_dim_static, rnn_num_hidden, n_samples, nf_layers, rnn, bidirectional):
    e_static_rnn_layer = rnn(rnn_num_hidden, return_sequences=False)
    if bidirectional:
        e_static_rnn_layer = Bidirectional(e_static_rnn_layer)
    e_hidden_static = e_static_rnn_layer(x) # returns last hidden state which holds global information about the whole sequence
    z_mean_static = Dense(latent_dim_static, name='z_mean_static')(e_hidden_static)
    z_log_var_static = Dense(latent_dim_static, name='z_log_var_static')(e_hidden_static)
    q_z_given_x_static = tfd.MultivariateNormalDiag(loc=z_mean_static, scale_diag=tf.exp(0.5 * z_log_var_static))
    z_static = q_z_given_x_static.sample(n_samples)
    if nf_layers > 0:
        # planar normalizing flow (transform simple diagonal Gaussian distribution into more complex posterior approximation)
        # q(z0 | x) ~ Normal(mu(x;phi), sigma(x;phi))
        # zK = bijector(z0), i.e. distribution for q(zK | x)
        q_z_given_x_static = PlanarFlows(latent_dim_static, q_z_given_x_static, nf_layers) # final approximated posterior distribution q(z | x)
        z_static = q_z_given_x_static(z_static)
    return q_z_given_x_static, z_static

def build_static_prior(latent_dim_static):
    return tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim_static), scale_diag=tf.ones(latent_dim_static), name='prior_static')

def build_decoder(latent_dim, latent_dim_static, input_dim, rnn_num_hidden, timesteps, use_static_encoder, allow_cudnn_kernel, rnn, rnn_cell):
    z_input = Input(shape=(None, timesteps, latent_dim), name='z') # input is (n_samples, batch_size, timesteps, latent_dim)
    n_samples = tf.shape(z_input)[0]
    z_input_ = tf.reshape(z_input, [-1, timesteps, latent_dim])  # collapse sample and batch dimesion
    
    if use_static_encoder:
        z_static_input = Input(shape=(None, latent_dim_static), name='z_static_input')  # input will be (n_samples, batch_size, latent_dim_static)
        z_static_input_ = tf.reshape(z_static_input, [-1, latent_dim_static]) # collapse sample and batch dimesion

        # from (batch_size, latent_static_dim) to (batch_size, timesteps, latent_static_dim)
        static = z_static_input_[..., tf.newaxis,:] + tf.zeros([timesteps, 1])
        # concatenate dynamic sample and static sample
        latents = tf.concat([z_input_, static], axis=-1)  # (sample * N, T, latent_dim + latent_static_dim)
        z_input_ = latents
    
    if allow_cudnn_kernel:
        d_hidden = rnn(rnn_num_hidden, return_sequences=True)(z_input_)
    else:
        d_hidden = RNN(rnn_cell(rnn_num_hidden), return_sequences=True)(z_input_)
        
    d_hidden = tf.reshape(d_hidden, [n_samples, -1, timesteps, rnn_num_hidden]) # restore sample dim
    
    d_dense = d_hidden
    
    x_mean = TimeDistributed(Dense(input_dim, name='x_mean'))(d_dense)
    x_log_var = TimeDistributed(Dense(input_dim, name='x_log_var'))(d_dense)
    
    if not use_static_encoder:
        decoder = Model(z_input, [x_mean, x_log_var], name='decoder')
    else:
        decoder = Model(inputs=[z_input, z_static_input], outputs=[x_mean, x_log_var], name='decoder')

    return decoder

def build_dsvae_ad(input_dim, 
    timesteps,
    rnn_num_hidden,
    dense_dim, 
    latent_dim,
    use_static_encoder=False,
    latent_dim_static=None,
    kl_anneal_weight=K.constant(1.),
    rnn_layer='GRU',
    bidirectional=False,
    nf_layers=3,
    allow_cudnn_kernel=False,
    use_connected_z_p=True,
    use_connected_z_q=True,
    weight_regularizer=None,
    use_anomaly_labels_for_training=False,
    noise_stddev=None,
    n_samples=1):

    """
    Creates DSVAE-AD model: Disentangled Sequential Variational Autoencoder Based Anomaly Detector.
    http://urn.fi/URN:NBN:fi:aalto-202101311730

    Args:
        input_dim (int): Number of features in the data.
        timesteps (int): Number of timesteps (window length).
        rnn_num_hidden (int): Number of hidden units in the RNN layers.
        dense_dim (int): Number of hidden units in the intermediate dense layers. 
        latent_dim (int): Dimension of the latent z space. 
        kl_anneal_weight (tensor): Weight factor of the KL term in loss calculation to avoid KL vanishing.
        rnn_layer (string): Type of the RNN layers. Use 'LSTM', 'GRU', or 'Simple'.
        nf_layers (int): Number of planar normalizing flow layers.
        allow_cudnn_kernel (bool): Whether to use optimized CuDNN kernels for LSTM / GRU layers. TODO: CuDNN kernels currently unstable on Windows (crashes randomly).
        use_connected_z_p (bool): Whether to use Linear Gaussian SSM as a prior or a simple diagonal Gaussian with zero mean and unit variance.
        use_connected_z_q (bool): Whether to use a connected diagonal Gaussian distribution as approximate posterior, or a plain diagonal Gaussian.
        weight_regularizer (tf.keras.regularizers.Regularizer): Weight regularizer to use.
        use_anomaly_labels_for_training (bool): Whether to use anomaly labels in the loss calculation (modified ELBO).
        noise_stddev (float, optional): Denoising autoencoding criterion. Apply noise to the inputs during training.
        n_samples (int): Number of Monte Carlo samples of z to take.
        
    Returns:
        (vae, encoder, generator): VAE, encoder, and generator that share the same weights.
    """
    if rnn_layer == 'LSTM':
        rnn = LSTM
        rnn_cell = LSTMCell
    elif rnn_layer == 'GRU':
        rnn = GRU
        rnn_cell = GRUCell
    else:
        rnn = SimpleRNN
        rnn_cell = SimpleRNNCell
    
    # encoder, from inputs to latent space
    x = Input(shape=(timesteps, input_dim), name='encoder_input')  # x_[t-T] to x_t
    if noise_stddev is not None:
        x_ = GaussianNoise(noise_stddev)(x)
    else:
        x_ = x

    # encoder q(z_[t-T:t] | x_[t-T:t])
    encoder, q_z_given_x, z = build_dynamic_encoder(
        x,
        x_,
        timesteps,
        rnn_num_hidden,
        dense_dim, 
        latent_dim,
        nf_layers,
        allow_cudnn_kernel,
        use_connected_z_q,
        weight_regularizer,
        n_samples,
        rnn,
        rnn_cell,
        bidirectional)

    # prior p(z)
    p_z = build_dynamic_prior(timesteps, latent_dim, use_connected_z_p)

    if use_static_encoder:
        # static encoder q(f | x_[t-T:t])
        q_z_given_x_static, z_static = build_static_encoder(x_, latent_dim_static, rnn_num_hidden, n_samples, nf_layers, rnn, bidirectional)
        # static prior p(f)
        p_z_static = build_static_prior(latent_dim_static)

    # decoder (generator), from latent space to reconstructed inputs
    decoder = build_decoder(latent_dim, latent_dim_static, input_dim, rnn_num_hidden, timesteps, use_static_encoder, allow_cudnn_kernel, rnn, rnn_cell)
    if not use_static_encoder:
        x_mean, x_log_var = decoder(z)
    else:
        x_mean, x_log_var = decoder([z, z_static])
    # decoder reconstruction distribution p(x | z)
    p_x_given_z = tfpl.DistributionLambda(
        make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=tf.exp(0.5 * t[1])),
        name="p_x_given_z"
    )([x_mean, x_log_var])

    if use_anomaly_labels_for_training:
        # anomaly (or missing data, for example) labels (0 or 1) for x_[t-T] to x_t
        # 0 = include in loss calculation
        # 1 = ignore in loss calculation
        x_input_labels = Input(shape=(timesteps), name='x_input_labels')
        alpha = 1 - x_input_labels
        beta = tf.reduce_mean(alpha, axis=-1)

        # VAE loss = -ELBO
        # E[log p(x|z)] - log likelihood
        likelihood_log_prob = tf.reduce_sum(alpha * p_x_given_z.log_prob(x), axis=-1) # sum over time
        # D_KL(q(z|x) || p(z|x)) - KL divergence between approximate posterior and latent prior
        if use_connected_z_p:
            # we want to have output shape (batch_size, n_timesteps), but LinearGaussianStateSpaceModel's log_prob returns (batch_size)
            prior_log_prob, _, _, _, _, _, _ = p_z.forward_filter(z)
        else:
            prior_log_prob = p_z.log_prob(z)
        kl_divergence = tf.reduce_sum(q_z_given_x.log_prob(z) - beta[:, tf.newaxis] * prior_log_prob, axis=-1)  # sum over time
        if use_static_encoder:
            kl_divergence_static = q_z_given_x_static.log_prob(z_static) - beta * p_z_static.log_prob(z_static)
            kl_divergence += kl_divergence_static
        vae_loss = VAEComputeLoss(kl_anneal_weight)([likelihood_log_prob, kl_divergence])

        # end-to-end variational autoencoder
        vae = Model(inputs=[x, x_input_labels], outputs=[p_x_given_z, vae_loss], name='vae') # note: need to return redundant vae_loss due to a bug how TF finds relevant nodes during add_loss
        vae.add_loss(vae_loss)
    else:
        # VAE loss = -ELBO
        # E[log p(x|z)] - log likelihood
        likelihood_log_prob = tf.reduce_sum(p_x_given_z.log_prob(x), axis=-1) # sum over time
        # D_KL(q(z|x) || p(z|x)) - KL divergence between approximate posterior and latent prior
        if use_connected_z_p:
            # use forward_filter method since we want to have output shape (batch_size, n_timesteps), 
            # but LinearGaussianStateSpaceModel's log_prob returns (batch_size)
            prior_log_prob, _, _, _, _, _, _ = p_z.forward_filter(z)
        else:
            prior_log_prob = p_z.log_prob(z)
        kl_divergence = tf.reduce_sum(q_z_given_x.log_prob(z) - prior_log_prob, axis=-1)  # sum over time
        if use_static_encoder:
            kl_divergence_static = q_z_given_x_static.log_prob(z_static) - p_z_static.log_prob(z_static) #tfd.kl_divergence(q_z_given_x_static, p_z_static)
            kl_divergence += kl_divergence_static
        vae_loss = VAEComputeLoss(kl_anneal_weight)([likelihood_log_prob, kl_divergence])

        # end-to-end variational autoencoder
        vae = Model(x, [p_x_given_z, vae_loss], name='vae') # note: need to return redundant vae_loss due to a bug how TF finds relevant nodes during add_loss
        vae.add_loss(vae_loss)

    # Monitor how the NLL and KL divergence differ over time
    vae.add_metric(-likelihood_log_prob, name='NLL', aggregation='mean')
    vae.add_metric(kl_divergence, name='KL', aggregation='mean')
    
    return vae, encoder, decoder