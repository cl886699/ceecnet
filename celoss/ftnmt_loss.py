import tensorflow as tf


class ftnmt_loss:
    def __init__(self, depth=5, axis=[1, 2, 3], smooth=1.0e-5):
        assert depth >= 0, ValueError("depth must be >= 0, aborting...")
        self.smooth = smooth
        self.axis = axis

        if depth == 0:
            self.depth = 1
            self.scale = 1.0
        else:
            self.depth = depth
            self.scale = 1./depth

    def inner_prod(self, prob, label):
        prod = tf.math.multiply(prob, label)
        prod = tf.math.reduce_sum(prod, axis=self.axis)
        return prod

    def tnmt_base(self, preds, labels):
        tpl = self.inner_prod(preds, labels)
        tpp = self.inner_prod(preds, preds)
        tll = self.inner_prod(labels, labels)

        num = tpl + self.smooth
        scale = 1./self.depth
        denum = 0.0
        for d in range(self.depth):
            a = 2.**d
            b = -(2.*a - 1.)
            denum = denum + 1./(tf.math.add(a * (tpp + tll), b * tpl) + self.smooth)
        result = tf.math.multiply(num, denum) * scale
        result = tf.math.reduce_mean(result, axis=0)
        return result

    def call(self, preds, labels):
        l1 = self.tnmt_base(preds, labels)
        l2 = self.tnmt_base(1. - preds, 1.0 - labels)
        result = 1.0 - 0.5 * (l1 + l2)
        return result


if __name__ == '__main__':
    import numpy as np
    a = tf.random.normal(shape=(2, 4, 4, 3))
    b = tf.reduce_mean(a, axis=0)
    print(b.shape)