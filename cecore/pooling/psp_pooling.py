import tensorflow as tf
from cecore.layers import ConvNormActBlock


class PSP_Pooling(tf.keras.Model):
    def __init__(self, nfilters, depth=4,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5)):
        super(PSP_Pooling, self).__init__()

        self.nfilters = nfilters
        self.depth = depth
        self.convs = []
        for _ in range(depth):
            self.convs.append(ConvNormActBlock(self.nfilters, kernel_size=(1, 1), padding='valid', normalization=normalization))
        self.conv_norm_final = ConvNormActBlock(self.nfilters, kernel_size=(1, 1), padding='valid', normalization=normalization)

    def HalfSplit(self, a):
        # _, h, w, _ = tf.shape(a)
        cc = tf.shape_n([a])[0]
        h = cc[-3]
        w = cc[-2]
        size1 = tf.cast(h/2, tf.int32)
        size2 = tf.cast(w/2, tf.int32)
        b = tf.split(a, num_or_size_splits=[size1, size1], axis=1)
        c1 = tf.split(b[0], num_or_size_splits=[size2, size2], axis=2)
        c2 = tf.split(b[1], num_or_size_splits=[size2, size2], axis=2)
        d11 = c1[0]
        d12 = c1[1]
        d21 = c2[0]
        d22 = c2[1]
        return [d11, d12, d21, d22]

    def QuarterStich(self, dss):
        temp1 = tf.concat([dss[0], dss[1]], axis=2)
        temp2 =tf.concat([dss[2], dss[3]], axis=2)
        result = tf.concat([temp1, temp2], axis=1)
        return result

    def HalfPooling(self, a):
        ds = self.HalfSplit(a)
        dss = []
        for x in ds:
            tmpx = tf.reduce_max(tf.reduce_max(x, axis=1, keepdims=True), axis=2, keepdims=True)
            dss += [tf.math.multiply(tf.ones_like(x), tmpx)]
        return self.QuarterStich(dss)

    def SplitPooling(self, a, depth):
        if depth==1:
            return self.HalfPooling(a)
        else:
            D = self.HalfSplit(a)
            return self.QuarterStich([self.SplitPooling(d, depth-1) for d in D])

    def call(self, input):
        p = [input]
        p += [self.convs[0](tf.math.multiply(tf.ones_like(input),
                                             tf.reduce_max(tf.reduce_max(input, axis=1, keepdims=True), axis=2, keepdims=True)
                                             ))]
        p += [self.convs[d](self.SplitPooling(input, d)) for d in range(1, self.depth)]
        out = tf.concat(p, axis=-1)
        out = self.conv_norm_final(out)
        return out


if __name__ == '__main__':
    a = tf.random.normal(shape=(1, 16, 16, 2048))
    psp = PSP_Pooling(1024)
    out = psp(a)
    print(out)