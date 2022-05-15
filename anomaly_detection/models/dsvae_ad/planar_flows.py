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
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow_probability.python.internal import dtype_util
from tensorflow.python.training.tracking.data_structures import NoDependency

class PlanarFlow(tf.keras.layers.Layer):
    """
    Planar normalizing flow
    Following "Variational Inference with Normalizing Flows" - https://arxiv.org/pdf/1505.05770.pdf
    and equations (10) - (12), (21) - (23).

    Note that this module should be implemented using tfp.Bijector, but there is a mysterious bug in tfp.Bijector / tfd.TransformedDistribution
    (TensorFlow Probability 0.11.1) when we use a chain of PlanarFlow bijectors during the inverse_log_det_jacobian calculation, causing 
    the inverse_log_det_jacobian to be incorrect (too small value), which causes invalid log_prob value and therefore messes up KL divergence calculation.
    The problem seems to occur only during training, but not when testing the component alone (the inverse_log_det_jacobian seem to be correct in that case).
    The issue might be related to the internal cache of tfp.Bijector, which somehow gets messed up during training.

    Due to the above reason, the PlanarFlow is implemented as a tf.keras.layers.Layer. The code below could be easily switched back to tfp.Bijector
    after the bug is fixed. Just need to implement the _forward() and _inverse_log_det_jacobian() methods.
    """
    def __init__(self, z_dim, init_stddev=0.01):
        super(PlanarFlow, self).__init__()
        self.d = z_dim
        self.init_stddev = init_stddev
        w_init = tf.random_normal_initializer(stddev=self.init_stddev)
        self.u = self.add_weight('u', shape=[1, self.d], initializer=w_init)
        self.w = self.add_weight('w', shape=[1, self.d], initializer=w_init)
        self.b = self.add_weight('b', shape=[1], initializer=w_init)

    @property
    def normalized_u(self):
        """
        Following "Variational Inference with Normalizing Flows" - https://arxiv.org/pdf/1505.05770.pdf
        (annex A.1. Planar flows), to ensure the invertibility of the flow.
        """
        def m(x):
            return -1 + tf.math.softplus(x)

        wTu = self.u @ tf.transpose(self.w)
        self._normalized_u = self.u + (m(wTu) - wTu) * (self.w / tf.norm(self.w))
        return self._normalized_u

    def h(self, x):
        """
        x: [N,] or [1]
        h: [N,] or [1]
        """
        return tf.math.tanh(x)

    def h_p(self, x):
        """
        x: [N,] or [1]
        h_p: [N,] or [1]
        """
        return 1 - (tf.math.tanh(x) ** 2)

    def psi(self, z):
        """
        z: [N,d]
        psi: [N,d]
        """
        return self.h_p(z @ tf.transpose(self.w) + self.b) @ self.w

    def call(self, z):
        """
        z: [N,d] or [N,timesteps,d]
        fz: [N,d], log_det_jacobian: [1]
        """
        u = self.normalized_u

        det_jacobian = 1 + self.psi(z) @ tf.transpose(u)
        log_det_jacobian = tf.squeeze(tf.math.log(tf.math.abs(det_jacobian)), -1)

        fz = z + self.h(z @ tf.transpose(self.w) + self.b) @ u

        return fz, log_det_jacobian

class PlanarFlows(tf.keras.Model):
    """Transforms the base distribution using planar normalizing flows.
    After the tfp.Bijector bug mentioned in `PlanarFlow` is fixed, 
    this can be replaced by tfd.TransformedDistribution + tfp.Chain(of PlanarFlows)
    which does the same thing in a standardized way.
    """

    _TF_MODULE_IGNORED_PROPERTIES = tf.Module._TF_MODULE_IGNORED_PROPERTIES.union(
        ('_intermediates_cache',)  # need to ignore this property to make model checkpointing work,
        # similar thing is done in Bijector for _cache (BijectorCache) property
        # otherwise we get this error: https://github.com/tensorflow/tensorflow/issues/35837
    )

    # mimics the BijectorCache, we cache certain properties during forward calls
    _intermediates_cache = {}

    def __init__(self, z_dim, base_distribution, n_flows=5):
        super(PlanarFlows, self).__init__()

        self.d = z_dim
        self.n_flows = n_flows
        self.base_distribution = base_distribution

        for i in range(1, self.n_flows + 1):
            setattr(self, "flow%i" % i, PlanarFlow(self.d))

    def call(self, z0):
        sum_log_det_jacobian = tf.cast(0., dtype=dtype_util.base_dtype(z0.dtype))

        zk = z0
        for i in range(1, self.n_flows + 1):
            zk, log_det_jacobian = getattr(self, "flow%i" % i)((zk))
            sum_log_det_jacobian += log_det_jacobian

        self._add_intermediate_to_cache(z0, zk.ref(), 'zk->z0')
        self._add_intermediate_to_cache(sum_log_det_jacobian, zk.ref(), 'zk->sum_ldj')

        return zk

    def log_prob(self, zk):
        if (zk.ref(), 'zk->z0') in self._intermediates_cache:
            z0 = self._intermediates_cache.pop((zk.ref(), 'zk->z0'))
        else:
            # will end up here if try to get log_prob for zk that is not in the cache
            # will happen if calling log_prob twice for same zk since the first call
            # removes it from the cache
            raise KeyError("PlanarFlows log_prob expected to find "
                           "key zk->z0 for zk in intermediates cache but did not.")

        if (zk.ref(), 'zk->sum_ldj') in self._intermediates_cache:
            sum_log_det_jacobian = self._intermediates_cache.pop((zk.ref(), 'zk->sum_ldj'))
        else:
            # will end up here if try to get log_prob for zk that is not in the cache
            # will happen if calling log_prob twice for same zk since the first call
            # removes it from the cache
            raise KeyError("PlanarFlows log_prob expected to find "
                           "key zk->sum_ldj for zk in intermediates cache but did not.")

        return self.base_distribution.log_prob(z0) - sum_log_det_jacobian

    def _add_intermediate_to_cache(self, intermediate, y, name):
        """
        Internal function used to cache intermediate results computed during the forward call
        """
        assert((y, name) not in self._intermediates_cache),\
            "key collision in _add_intermediate_to_cache"
        self._intermediates_cache[(y, name)] = intermediate