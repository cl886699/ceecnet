from cemodels.heads.head_cmtsk import *
from cemodels.mantis.mantis_dn_features import *


# Mantis conditioned multitasking.
class mantis_dn_cmtsk(tf.keras.Model):
    def __init__(self, nfilters_init, depth, NClasses, widths=[1], psp_depth=4, verbose=True,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 nheads_start=8, model='CEECNetV1', upFuse=False, ftdepth=5):
        super(mantis_dn_cmtsk, self).__init__()

        self.features = mantis_dn_features(nfilters_init=nfilters_init, depth=depth, widths=widths,
                                           psp_depth=psp_depth, verbose=verbose, normalization=normalization, nheads_start=nheads_start, model=model,
                                           upFuse=upFuse, ftdepth=ftdepth)
        self.head = Head_CMTSK_BC(nfilters_init, NClasses, normalization=normalization)

    def call(self, input_t1, input_t2):
        out1, out2 = self.features(input_t1, input_t2)
        # self.features.summary()
        # outputs = self.head(out1, out2)
        # self.head.summary()
        return self.head(out1, out2)

    # def build(self, input_shape=(1, 256, 256, 3)):
    #     input = tf.random.normal(shape=input_shape)
    #     out1, out2 = self.features(input, input)
    #     return self.head(out1, out2)
