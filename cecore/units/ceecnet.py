from cecore.layers.attention import *
from cecore.layers import ConvNormActBlock, Scale


class ExpandLayer(tf.keras.Model):
    def __init__(self, nfilters,
                 ngroups=1,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5)):
        super(ExpandLayer, self).__init__()
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.conv1 = ConvNormActBlock(filters=nfilters, kernel_size=3, padding='same', groups=ngroups,
                                      normalization=normalization)
        self.conv2 = ConvNormActBlock(filters=nfilters, kernel_size=3, padding='same', groups=ngroups,
                                      normalization=normalization)

    def call(self, inputs):
        out = self.upsample(inputs)
        out = tf.nn.relu(self.conv1(out))
        out = tf.nn.relu(self.conv2(out))
        return out


class ExpandCombine(tf.keras.Model):
    def __init__(self, nfilters,
                 ngroups=1,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5)):
        super(ExpandCombine, self).__init__()
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.conv1 = ConvNormActBlock(filters=nfilters, kernel_size=3, padding='same', groups=ngroups,
                                      normalization=normalization)
        self.conv2 = ConvNormActBlock(filters=nfilters, kernel_size=3, padding='same', groups=ngroups,
                                      normalization=normalization)

    def call(self, input1, input2):
        out = self.upsample(input1)
        out = tf.nn.relu(self.conv1(out))
        out2 = self.conv2(tf.concat([out, input2], axis=-1))
        out2 = tf.nn.relu(out2)
        return out2


class CEEC_unit_v1(tf.keras.Model):
    def __init__(self, nfilters, nheads=1, ngroups=1,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 ftdepth=5,
                 name="ceec",
                 ):
        super(CEEC_unit_v1, self).__init__()

        nfilters_init = nfilters // 2
        self.conv_init_1 = ConvNormActBlock(filters=nfilters_init, kernel_size=3, padding='same', strides=1,
                                            groups=ngroups, normalization=normalization)
        self.compr11 = ConvNormActBlock(filters=nfilters_init * 2, kernel_size=3, padding='same', strides=2,
                                        groups=ngroups, normalization=normalization)
        self.compr12 = ConvNormActBlock(filters=nfilters_init * 2, kernel_size=3, padding='same', strides=1,
                                        groups=ngroups, normalization=normalization)
        self.expand1 = ExpandCombine(nfilters=nfilters_init, ngroups=ngroups, normalization=normalization)

        self.conv_init_2 = ConvNormActBlock(filters=nfilters_init, kernel_size=3, padding='same', strides=1,
                                            groups=ngroups, normalization=normalization)
        self.expand2 = ExpandLayer(nfilters=nfilters_init // 2, ngroups=ngroups, normalization=normalization)
        self.compr21 = ConvNormActBlock(filters=nfilters_init, kernel_size=3, padding='same', strides=2, groups=ngroups,
                                        normalization=normalization)
        self.compr22 = ConvNormActBlock(filters=nfilters_init, kernel_size=3, padding='same', strides=1, groups=ngroups,
                                        normalization=normalization)

        self.collect = ConvNormActBlock(filters=nfilters, kernel_size=3, padding='same', strides=1, groups=1, normalization=normalization)
        self.att = FTAttention2D(nkeys=nfilters, nheads=nheads, normalization=normalization, ftdepth=ftdepth)
        self.ratt122 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads, normalization=normalization, ftdepth=ftdepth)
        self.ratt211 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads, normalization=normalization, ftdepth=ftdepth)
        with tf.name_scope(name=name):
            self.gamma1 = Scale('gamma1', scale=0.0)
            self.gamma2 = Scale('gamma2', scale=0.0)
            self.gamma3 = Scale('gamma3', scale=0.0)

    def call(self, input):
        out10 = self.conv_init_1(input)
        out1 = tf.nn.relu(self.compr11(out10))
        out1 = tf.nn.relu(self.compr12(out1))
        out1 = tf.nn.relu(self.expand1(out1, out10))

        out20 = self.conv_init_2(input)
        out2 = tf.nn.relu(self.expand2(out20))
        out2 = tf.nn.relu(self.compr21(out2))
        out2 = tf.nn.relu(self.compr22(tf.concat([out2, out20], axis=-1)))

        att = self.gamma1(self.att(input))
        ratt122 = self.gamma2(self.ratt122(out1, out2, out2))
        ratt211 = self.gamma3(self.ratt211(out2, out1, out1))

        ones1 = tf.ones_like(out10)
        ones2 = tf.ones_like(input)
        out122 = tf.math.multiply(out1, ones1 + ratt122)
        out211 = tf.math.multiply(out2, ones1 + ratt211)
        out12 = tf.nn.relu(self.collect(tf.concat([out122, out211], axis=-1)))
        out_res = tf.math.multiply(input + out12, ones2 + att)
        return out_res


class Fusion(tf.keras.Model):
    def __init__(self, nfilters, kernel_size=3, padding='same', nheads=1,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 ftdepth=5, name='fuse'):
        super(Fusion, self).__init__()
        self.fuse = ConvNormActBlock(filters=nfilters, kernel_size=kernel_size, padding=padding, groups=nheads,
                                     normalization=normalization)
        self.relatt12 = RelFTAttention2D(nkeys=nfilters, kernel_size=kernel_size, padding=padding, nheads=nheads,
                                         normalization=normalization, ftdepth=ftdepth)
        self.relatt21 = RelFTAttention2D(nkeys=nfilters, kernel_size=kernel_size, padding=padding, nheads=nheads,
                                         normalization=normalization, ftdepth=ftdepth)
        with tf.name_scope(name=name):
            self.gamma1 = Scale(name='gamma1', scale=0.0)
            self.gamma2 = Scale(name='gamma2', scale=0.0)

    def call(self, input1, input2):
        relatt12 = self.gamma1(self.relatt12(input1, input2, input2))
        relatt21 = self.gamma2(self.relatt21(input2, input1, input1))
        ones = tf.ones_like(input1)
        out12 = tf.math.multiply(input1, ones + relatt12)
        out21 = tf.math.multiply(input2, ones + relatt21)

        fuse = self.fuse(tf.concat([out12, out21], axis=-1))
        fuse = tf.nn.relu(fuse)

        return fuse


class CATFusion(tf.keras.Model):
    def __init__(self, nfilters_out, nfilters_in, kernel_size=3, padding='same', nheads=1,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 ftdepth=5, name='catfuse'):
        super(CATFusion, self).__init__()
        self.fuse = ConvNormActBlock(filters=nfilters_out, kernel_size=kernel_size, padding=padding, groups=nheads,
                                     normalization=normalization)
        self.relatt12 = RelFTAttention2D(nkeys=nfilters_in, kernel_size=kernel_size, padding=padding, nheads=nheads,
                                         normalization=normalization, ftdepth=ftdepth)
        self.relatt21 = RelFTAttention2D(nkeys=nfilters_in, kernel_size=kernel_size, padding=padding, nheads=nheads,
                                         normalization=normalization, ftdepth=ftdepth)
        with tf.name_scope(name=name):
            self.gamma1 = Scale(name='gamma1', scale=0.0)
            self.gamma2 = Scale(name='gamma2', scale=0.0)

    def call(self, input1, input2):
        relatt12 = self.gamma1(self.relatt12(input1, input2, input2))
        relatt21 = self.gamma2(self.relatt21(input2, input1, input1))
        ones = tf.ones_like(input1)
        out12 = tf.math.multiply(input1, ones + relatt12)
        out21 = tf.math.multiply(input2, ones + relatt21)

        fuse = self.fuse(tf.concat([out12, out21], axis=-1))
        fuse = tf.nn.relu(fuse)

        return fuse


class combine_layers_wthFusion(tf.keras.Model):
    def __init__(self, nfilters, nheads=1,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 ftdepth=5):
        super(combine_layers_wthFusion, self).__init__()
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.conv1 = ConvNormActBlock(filters=nfilters, kernel_size=3, padding='same', groups=nheads,
                                      normalization=normalization)
        self.conv3 = Fusion(nfilters=nfilters, kernel_size=3, padding='same', nheads=nheads,
                            normalization=normalization, ftdepth=ftdepth)

    def call(self, layer_lo, layer_hi):
        up = self.upsample(layer_lo)
        up = tf.nn.relu(self.conv1(up))
        x = self.conv3(up, layer_hi)
        return x


class ExpandNCombine_V3(tf.keras.Model):
    def __init__(self, nfilters,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 ngroups=1, ftdepth=5
                 ):
        super(ExpandNCombine_V3, self).__init__()
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.conv1 = ConvNormActBlock(filters=nfilters, kernel_size=3, padding='same', groups=ngroups,
                                      normalization=normalization)
        self.conv2 = ConvNormActBlock(filters=nfilters, kernel_size=3, padding='same', groups=ngroups,
                                      normalization=normalization)
        self.conv3 = Fusion(nfilters=nfilters, kernel_size=3, padding='same', nheads=ngroups,
                            normalization=normalization, ftdepth=ftdepth)

    def call(self, input1, input2):
        out = self.upsample(input1)
        out1 = tf.nn.relu(self.conv1(out))
        out2 = tf.nn.relu(self.conv2(input2))
        out = tf.nn.relu(self.conv3(out1, out2))
        return out


class CEEC_unit_v2(tf.keras.Model):
    def __init__(self, nfilters, nheads=1, ngroups=1,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 ftdepth=5, name='ceecv2'):
        super(CEEC_unit_v2, self).__init__()

        with tf.name_scope(name=name):
            nfilters_init = nfilters // 2
            self.conv_init_1 = ConvNormActBlock(filters=nfilters_init, kernel_size=3, padding='same', strides=1,
                                                groups=ngroups, normalization=normalization)
            self.compr11 = ConvNormActBlock(filters=nfilters_init * 2, kernel_size=3, padding='same', strides=2,
                                            groups=ngroups, normalization=normalization)
            self.compr12 = ConvNormActBlock(filters=nfilters_init * 2, kernel_size=3, padding='same', strides=1,
                                            groups=ngroups, normalization=normalization)
            self.expand1 = ExpandNCombine_V3(nfilters=nfilters_init, normalization=normalization, ngroups=ngroups, ftdepth=ftdepth)

            self.conv_init_2 = ConvNormActBlock(filters=nfilters_init, kernel_size=3, padding='same', strides=1,
                                                groups=ngroups, normalization=normalization)
            self.expand2 = ExpandLayer(nfilters=nfilters_init // 2, ngroups=ngroups, normalization=normalization)
            self.compr21 = ConvNormActBlock(filters=nfilters_init, kernel_size=3, padding='same', strides=2, groups=ngroups,
                                            normalization=normalization)
            self.compr22 = Fusion(nfilters=nfilters_init, kernel_size=3, padding='same', nheads=ngroups,
                                  normalization=normalization, ftdepth=ftdepth, name=name)

            self.collect = CATFusion(nfilters_out=nfilters, nfilters_in=nfilters_init, kernel_size=3, padding='same',
                                     nheads=1, normalization=normalization, ftdepth=ftdepth)
            self.att = FTAttention2D(nkeys=nfilters, nheads=nheads, normalization=normalization, ftdepth=ftdepth)
            self.ratt122 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads, normalization=normalization, ftdepth=ftdepth)
            self.ratt211 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads, normalization=normalization, ftdepth=ftdepth)
            self.gamma1 = Scale('gamma1', scale=0.0)
            self.gamma2 = Scale('gamma2', scale=0.0)
            self.gamma3 = Scale('gamma3', scale=0.0)

    def call(self, input):
        out10 = self.conv_init_1(input)
        out1 = tf.nn.relu(self.compr11(out10))
        out1 = tf.nn.relu(self.compr12(out1))
        out1 = tf.nn.relu(self.expand1(out1, out10))
        # self.expand1.summary()

        out20 = self.conv_init_2(input)
        out2 = tf.nn.relu(self.expand2(out20))
        # self.expand2.summary()
        out2 = tf.nn.relu(self.compr21(out2))
        out2 = self.compr22(out2, out20)
        # self.compr22.summary()

        att = self.gamma1(self.att(input))
        ratt122 = self.gamma2(self.ratt122(out1, out2, out2))
        ratt211 = self.gamma3(self.ratt211(out2, out1, out1))

        ones1 = tf.ones_like(out10)
        ones2 = tf.ones_like(input)
        out122 = tf.math.multiply(out1, ones1 + ratt122)
        out211 = tf.math.multiply(out2, ones1 + ratt211)
        out12 = self.collect(out122, out211)
        # self.collect.summary()
        out_res = tf.math.multiply(input + out12, ones2 + att)
        return out_res




if __name__ == '__main__':
    c1unit = CEEC_unit_v1(nfilters=32, nheads=2, ngroups=2)
    inputs = tf.random.normal(shape=(1, 16, 16, 32))
    c1unit(inputs)
    c1unit.summary()