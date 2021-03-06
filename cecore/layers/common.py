import tensorflow as tf
from cecore.layers import GroupNormalization, Mish


class Scale(tf.keras.layers.Layer):
    def __init__(self, name='name', scale=1.0):
        super(Scale, self).__init__()
        self.scale = self.add_weight(name=name, shape=([1]), initializer=tf.keras.initializers.constant(scale),
                                     trainable=True)
    def call(self, inputs):
        return self.scale * inputs


def build_normalization(normalization, **kwargs):
    if normalization == "group_norm":
        return GroupNormalization(**kwargs)
    elif normalization == "batch_norm":
        return tf.keras.layers.BatchNormalization(**kwargs)
    else:
        raise TypeError("Could not interpret normalization function identifier: {}".format(
            repr(normalization)))


def build_activation(**kwargs):
    if kwargs["activation"] == "leaky_relu":
        kwargs.pop("activation")
        return tf.keras.layers.LeakyReLU(**kwargs)
    if kwargs["activation"] == "mish":
        kwargs.pop("activation")
        return Mish(**kwargs)

    return tf.keras.layers.Activation(**kwargs)


class ConvNormActBlock(tf.keras.layers.Layer):
    """Conv2D-Norm-Activation block

    Args:
        filters(int): the filters of middle layer, i.e. conv2.
        kernel_size(int[tuple]): the kernel size.
        strides(int[tuple]): the strides.
        padding(str): one of `"valid"` or `"same"` (case-insensitive).
        groups(int): A positive integer specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved separately
            with `filters / groups` filters. The output is the concatenation of all
            the `groups` results along the channel axis. Input channels and `filters`
            must both be divisible by `groups`.
        data_format(string): one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds
            to inputs with shape `(batch_size, height, width, channels)` while
            `channels_first` corresponds to inputs with shape `(batch_size, channels,
            height, width)`. It defaults to the `image_data_format` value found in
            your Keras config file at `~/.keras/keras.json`. If you never set it, then
            it will be `channels_last`.
        dilation_rate(int[tuple]): an integer or tuple/list of 2 integers, specifying the
            dilation rate to use for dilated convolution. Can be a single integer to
            specify the same value for all spatial dimensions. Currently, specifying
            any `dilation_rate` value != 1 is incompatible with specifying any stride
            value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix (see `keras.initializers`).
        trainable(bool): if `True` the weights of this layer will be marked as
            trainable (and listed in `layer.trainable_weights`).
        normalization(dict): the normalization parameter dict.
        gamma_zeros(bool): bool, default False
            Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
        activation(dict[str]): the activation paramerter dict or the name of activation.
        name(str): the block name.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 groups=1,
                 padding="same",
                 data_format="channels_last",
                 dilation_rate=1,
                 kernel_initializer="glorot_uniform",
                 trainable=True,
                 use_bias=False,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 gamma_zeros=False,
                 activation=None,
                 dropblock=None,
                 name=None):
        super(ConvNormActBlock, self).__init__(name=name)

        self.strides = strides
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(strides, int):
            strides = (strides, strides)

        # if strides != (1, 1):
        #     dilation_rate = (1, 1)
        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate, dilation_rate)

        if strides == (1, 1):
            p = 1
            padding == "same"
        else:
            p = ((kernel_size[0] - 1) // 2 * dilation_rate[0], (kernel_size[1] - 1) * dilation_rate[1] // 2)
            self.pad = tf.keras.layers.ZeroPadding2D(p, data_format=data_format)
            padding = "valid"

        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           data_format=data_format,
                                           dilation_rate=dilation_rate,
                                           groups=groups,
                                           use_bias=normalization is None or use_bias,
                                           trainable=trainable,
                                           kernel_initializer=kernel_initializer,
                                           name="conv2d")
        channel_axis = -1 if data_format == "channels_last" else 1
        if not trainable and normalization is not None:
            normalization["trainable"] = False
        if normalization is not None:
            normalization["axis"] = channel_axis

            norm_name = (
                "batch_norm"
                if "batch_norm" in normalization["normalization"]
                else normalization["normalization"])
            if norm_name == "batch_norm" and not trainable:
                normalization["normalization"] = "frozen_batch_norm"

            self.norm = build_normalization(
                gamma_initializer="zeros" if gamma_zeros else "ones",
                name=norm_name,
                **normalization)
        else:
            self.norm = None
        # self.act = build_activation(**activation, name=activation["activation"]) if activation is not None else None
        # self.dropblock = DropBlock2D(**dropblock, name="dropblock") if dropblock is not None else None
        if activation is not None:
            self.act = (build_activation(**activation)
                        if isinstance(activation, dict)
                        else build_activation(activation=activation))
        else:
            self.act = None

    def build(self, input_shape):
        super(ConvNormActBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        if hasattr(self, "pad"):
            x = self.pad(inputs)
        else:
            x = inputs
        d1 = tf.reduce_sum(x)
        x = self.conv(x)
        d2 = tf.reduce_sum(x)
        if self.norm is not None:
            x = self.norm(x, training=training)
            d3 = tf.reduce_sum(x)
        if self.act is not None:
            x = self.act(x)
            d4 = tf.reduce_sum(x)
        # if self.dropblock is not None:
        #     x = self.dropblock(x, training=training)

        return x

    def fused_call(self, inputs, training=None):
        if hasattr(self, "pad"):
            x = self.pad(inputs)
        else:
            x = inputs
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)

        return x

    def get_config(self):
        layer_config = dict()
        layer_config.update(self.conv.get_config())
        if self.norm is not None:
            layer_config.update(self.norm.get_config())
        if self.act is not None:
            layer_config.update(self.act.get_config())

        layer_config.update(super(ConvNormActBlock, self).get_config())

        return layer_config
