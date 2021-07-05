import tensorflow as tf
from cecore.layers import FTanimoto, ConvNormActBlock, build_normalization


class RelFTAttention2D(tf.keras.Model):
    def __init__(self, nkeys,
                 kernel_size=3,
                 padding='same',
                 nheads=1,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 ftdepth=5,
                 **kwards):
        super(RelFTAttention2D, self).__init__(**kwards)
        self.query = ConvNormActBlock(filters=nkeys, kernel_size=kernel_size, padding=padding, normalization=normalization,
                                      groups=nheads)
        self.key = ConvNormActBlock(filters=nkeys, kernel_size=kernel_size, padding=padding, normalization=normalization,
                                    groups=nheads)
        self.value = ConvNormActBlock(filters=nkeys, kernel_size=kernel_size, padding=padding, normalization=normalization,
                                      groups=nheads)
        self.metric_space = FTanimoto(depth=ftdepth, axis=-1)
        self.metric_channel = FTanimoto(depth=ftdepth, axis=[1, 2])
        self.norm = build_normalization(**normalization)

    def call(self, input1, input2, input3):
        q = tf.nn.sigmoid(self.query(input1))
        k = tf.nn.sigmoid(self.key(input2))
        v = tf.nn.sigmoid(self.value(input3))

        att_spat = self.metric_space(q, k)
        v_spat = tf.math.multiply(att_spat, v)
        att_chan = self.metric_channel(q, k)
        v_chan = tf.math.multiply(att_chan, v)
        v_cspat = 0.5 * tf.math.add(v_chan, v_spat)
        v_cspat = self.norm(v_cspat)
        return v_cspat


class FTAttention2D(tf.keras.Model):
    def __init__(self, nkeys, kernel_size=3, padding='same', nheads=1,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 ftdepth=5,
                 **kwards):
        super(FTAttention2D, self).__init__()

        self.att = RelFTAttention2D(nkeys=nkeys, kernel_size=kernel_size, padding=padding, nheads=nheads, normalization=normalization,
                                    ftdepth=ftdepth, **kwards)

    def call(self, inputs):
        return self.att(inputs, inputs, inputs)
