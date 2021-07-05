import tensorflow as tf
from core.layers.common import build_normalization

# normalization = dict(normalization='group_norm', groups=4, epsilon=1e-5)
#
# layer = build_normalization(**normalization)
# inputs = tf.keras.Input(shape=(1, 16, 16, 4))
# outputs = layer(inputs)
# model = tf.keras.Model(inputs, outputs)
# model.summary()
# print(layer)


import tensorflow_addons as tfa

# layer = tfa.layers.GroupNormalization(groups=32)
# inputs = tf.keras.Input(shape=(1, 16, 16, 32))
# outputs = layer(inputs)
# model = tf.keras.Model(inputs, outputs)
# model.summary()
# print(layer)