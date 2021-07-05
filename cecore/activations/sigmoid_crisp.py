import tensorflow as tf
from cecore.layers import Scale


class Scale(tf.keras.layers.Layer):
    def __init__(self, name='name', scale=1.0):
        super(Scale, self).__init__()
        self.scale = self.add_weight(name=name, shape=([1]), initializer=tf.keras.initializers.constant(scale),
                                     trainable=True)

    def call(self, inputs):
        return 1.0 / (inputs + tf.sigmoid(self.scale))


class SigmoidCrisp(tf.keras.layers.Layer):
    def __init__(self, smooth=1.e-2, name='gamma'):
        super(SigmoidCrisp, self).__init__()

        self.smooth = smooth
        self.gamma = Scale(name=name, scale=1.0)

    def call(self, inputs):
        out = self.gamma(self.smooth)
        out = tf.math.multiply(inputs, out)
        out = tf.sigmoid(out)
        return out
