import tensorflow as tf


class FTanimoto(tf.keras.layers.Layer):
    def __init__(self, depth=5, smooth=1.0e-5, axis=[1, 2], **kwargs):
        super().__init__(**kwargs)
        assert depth >= 0, "Expecting depth >= 0, aborting ..."
        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1. / depth
        self.smooth = smooth
        self.axis = axis

    def inner_prod(self, prod, label):
        prod = tf.math.multiply(prod, label)
        prod = tf.math.reduce_sum(prod, axis=self.axis, keepdims=True)
        return prod

    def tnmt_base(self, preds, labels):
        tpl = self.inner_prod(preds, labels)
        tpp = self.inner_prod(preds, preds)
        tll = self.inner_prod(labels, labels)
        num = tpl + self.smooth
        denum = 0.0
        for d in range(self.depth):
            a = 2. ** d
            b = -(2. * a - 1.)
            denum = denum + 1 / (tf.math.multiply(a * (tpp + tll), b * tpl) + self.smooth)

        return tf.math.multiply(num, denum) * self.scale

    def call(self, preds, labels):
        l12 = self.tnmt_base(preds, labels)
        l12 = l12 + self.tnmt_base(1.0 - preds, 1.0 - labels)
        return 0.5 * l12
