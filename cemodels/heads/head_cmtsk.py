from cecore.pooling.psp_pooling import *
from cecore.layers.common import ConvNormActBlock
from cecore.activations.sigmoid_crisp import SigmoidCrisp


class HeadSingle(tf.keras.layers.Layer):
    def __init__(self, nfilters,  NClasses, depth=2,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5)):
        super(HeadSingle, self).__init__()
        self.logits = tf.keras.Sequential()
        for _ in range(depth):
            self.logits.add(ConvNormActBlock(filters=nfilters, kernel_size=(3, 3), padding='same', normalization=normalization, activation='relu'))
        self.logits.add(ConvNormActBlock(filters=NClasses, kernel_size=(1, 1), padding='valid', normalization=None))

    def call(self, inputs):
        return self.logits(inputs)


class Head_CMTSK_BC(tf.keras.Model):
    def __init__(self, nfilters_init, NClasses,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5)):
        super(Head_CMTSK_BC, self).__init__()
        self.model_name = 'Head_CMTSK_BC'
        with tf.name_scope(name=self.model_name):
            self.nfilters = nfilters_init
            self.NClasses = NClasses

            self.psp_2ndlast = PSP_Pooling(nfilters=self.nfilters, normalization=normalization)
            self.bound_logits = HeadSingle(nfilters=self.nfilters, NClasses=self.NClasses, normalization=normalization)
            self.CrispSigm = SigmoidCrisp(name='head_gamma')
            self.bound_Equalizer = ConvNormActBlock(filters=self.nfilters, kernel_size=(1, 1), normalization=normalization)
            self.distance_logits = HeadSingle(nfilters=self.nfilters, NClasses=NClasses, normalization=normalization)
            self.distance_Equalizer = ConvNormActBlock(filters=self.nfilters, kernel_size=(1, 1), normalization=normalization)

            self.Comb_bound_dist = ConvNormActBlock(filters=self.nfilters, kernel_size=(1, 1), normalization=normalization)
            self.final_segm_logits = HeadSingle(nfilters=self.nfilters, NClasses=NClasses, normalization=normalization)
            if self.NClasses == 1:
                #todo
                self.ChannelAct = tf.nn.sigmoid()
            else:
                self.ChannelAct = tf.keras.layers.Softmax(axis=-1)

    def call(self, UpConv4, conv1):
        convl = tf.concat([conv1, UpConv4], axis=-1)
        conv = self.psp_2ndlast(convl)
        conv = tf.nn.relu(conv)

        dist = self.distance_logits(convl)
        dist = self.ChannelAct(dist)
        distEq = tf.nn.relu(self.distance_Equalizer(dist))

        bound = tf.concat([conv, distEq], axis=-1)
        bound = self.bound_logits(bound)
        bound = self.CrispSigm(bound)
        boundEq = tf.nn.relu(self.bound_Equalizer(bound))

        comb_bd = self.Comb_bound_dist(tf.concat([boundEq, distEq], axis=-1))
        comb_bd = tf.nn.relu(comb_bd)

        all_layers = tf.concat([comb_bd, conv], axis=-1)
        final_segm = self.final_segm_logits(all_layers)
        final_segm = self.ChannelAct(final_segm)

        return final_segm, bound, dist