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
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, RNN, GRUCell, LSTMCell, SimpleRNNCell, RepeatVector
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, Concatenate

def build_encdec_ad(
    input_dim, 
    timesteps,
    rnn_num_hidden=128,
    rnn_layer='LSTM',
    bidirectional=False,
    allow_cudnn_kernel=False,
    noise_stddev=None):

    if rnn_layer == 'LSTM':
        rnn = LSTM
        rnn_cell = LSTMCell
    elif rnn_layer == 'GRU':
        rnn = GRU
        rnn_cell = GRUCell
    else:
        rnn = SimpleRNN
        rnn_cell = SimpleRNNCell

    x = Input(shape=(timesteps, input_dim), name='encoder_input')  # x_[t-T] to x_t
    enc_rnn_layer = rnn(rnn_num_hidden, return_sequences=False, return_state=True)
    if bidirectional:
        enc_rnn_layer = rnn(int(rnn_num_hidden / 2), return_sequences=False, return_state=True)
        enc_rnn_layer = Bidirectional(enc_rnn_layer)
    if rnn_layer == 'LSTM':
        if bidirectional:
            _, forward_h, forward_c, backward_h, backward_c = enc_rnn_layer(x)
            state_h = Concatenate()([forward_h, backward_h])
            state_c = Concatenate()([forward_c, backward_c])
            enc_state = [state_h, state_c]
        else:
            _, final_memory_state, final_carry_state = enc_rnn_layer(x)
            enc_state = [final_memory_state, final_carry_state]
    else:
        # GRU and SimpleRNN only has single state
        if bidirectional:
            _, forward_h, backward_h = enc_rnn_layer(x)
            enc_state = Concatenate()([forward_h, backward_h])
        else:
            _, enc_state = enc_rnn_layer(x)

    dec_rnn_unit = rnn_cell(rnn_num_hidden)
    dec_linear_layer = Dense(input_dim)
    dec_state = enc_state

    dec_outputs = []
    dec_input = tf.ones((tf.shape(x)[0], input_dim), dtype=tf.float32)
    for _ in range(timesteps):
        dec_output, dec_state = dec_rnn_unit(dec_input, dec_state)
        dec_output = dec_linear_layer(dec_output)
        dec_outputs.append(dec_output)
        dec_input = dec_output
        
    x_rec = tf.stack(dec_outputs[::-1]) # reverse array
    x_rec = tf.transpose(x_rec, [1, 0, 2]) # from (n_timesteps, batch_size, x_dim) to (batch_size, n_timesteps, x_dim)

    loss = tf.keras.losses.mean_squared_error(x, x_rec)

    model = Model(x, [x_rec], name='enc_dec_ae')
    model.add_loss(loss)

    return model