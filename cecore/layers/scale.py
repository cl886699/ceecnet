import tensorflow as tf
from cecore.layers import ConvNormActBlock


class DownSample(tf.keras.Model):
    def __init__(self, nfilters,
                 factor=2,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5)):
        super(DownSample, self).__init__()
        # Double the size of filters, since you downscale by 2.
        self.factor = factor
        self.nfilters = nfilters * self.factor

        self.kernel_size = (3,3)
        self.strides = (factor,factor)
        self.pad = 'same'

        self.convdn = ConvNormActBlock(filters=self.nfilters,
                                       kernel_size=self.kernel_size,
                                       strides=self.strides,
                                       padding=self.pad,
                                       normalization=normalization
                                       )

    def call(self, inputs):
        return self.convdn(inputs)


class UpSample(tf.keras.Model):
    def __init__(self, nfilters, factor=2,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5)):
        super(UpSample, self).__init__()
        self.factor = (factor, factor)
        self.nfilters = nfilters // factor
        self.upsample = tf.keras.layers.UpSampling2D(size=self.factor, interpolation='nearest')
        self.convup_normed = ConvNormActBlock(filters=self.nfilters, kernel_size=(1, 1), normalization=normalization)

    def call(self, inputs):
        x = self.upsample(inputs)
        return self.convup_normed(x)