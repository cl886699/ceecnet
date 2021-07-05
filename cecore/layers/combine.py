import tensorflow as tf
from cecore.layers import ConvNormActBlock, UpSample


class combine_layers(tf.keras.Model):
    def __init__(self, nfilters,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 ):
        super(combine_layers, self).__init__()
        self.up = UpSample(nfilters, normalization=normalization)
        self.conv_normed = ConvNormActBlock(filters=nfilters,
                                            kernel_size=(1,1),
                                            padding='valid',
                                            normalization=normalization)

    def call(self, _layer_lo, _layer_hi):
        up = tf.nn.relu(self.up(_layer_lo))
        # self.up.summary()
        x = tf.concat([up, _layer_hi], axis=-1)
        x = self.conv_normed(x)
        return x