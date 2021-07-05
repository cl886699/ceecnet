from cecore.pooling import PSP_Pooling
from cecore.layers.combine import *
from cecore.units.ceecnet import *
from cecore.layers.scale import *


class mantis_dn_features(tf.keras.Model):
    def __init__(self, nfilters_init, depth, widths=[1], psp_depth=4,
                 verbose=True, nheads_start=8,  model='CEECNetV1',
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5),
                 upFuse=False, ftdepth=5):
        super(mantis_dn_features, self).__init__()

        self.depth = depth
        if len(widths) == 1 and depth != 1:
            widths = widths * depth
        else:
            assert depth == len(widths), ValueError("depth and length of widths must match, aborting ...")

        self.conv_first = ConvNormActBlock(nfilters_init, kernel_size=(1, 1), padding='valid', normalization=normalization)
        self.fuse_first = Fusion(nfilters_init, normalization=normalization)
        self.conv_dn = []
        self.pools = []
        self.fuse = []
        for idx in range(depth):
            nheads = nheads_start * 2**idx
            nfilters = nfilters_init * 2**idx
            if verbose:
                print("depth:= {0}, nfilters: {1}, nheads::{2}, widths::{3}".format(idx, nfilters, nheads, widths[idx]))
            tnet = tf.keras.Sequential()
            for indx2 in range(widths[idx]):
                if model == 'CEECNetV1':
                    tnet.add(CEEC_unit_v1(nfilters=nfilters, nheads=nheads, ngroups=nheads,
                                          normalization=normalization, ftdepth=ftdepth,
                                          name=model + '_down' + str(idx) + str(indx2)))
                elif model == 'CEECNetV2':
                    tnet.add(CEEC_unit_v2(nfilters=nfilters, nheads=nheads, ngroups=nheads,
                                          normalization=normalization, ftdepth=ftdepth,
                                          name=model + '_down' + str(idx) + str(indx2)))
                else:
                    raise ValueError("I don't know requested model, aborting ... - Given model::{}".format(model))
            self.conv_dn.append(tnet)
            if idx < depth - 1:
                self.fuse.append(Fusion(nfilters=nfilters, nheads=nheads, normalization=normalization, name='mantis_fuse_' + str(idx)))
                self.pools.append(DownSample(nfilters=nfilters, normalization=normalization))
        self.middle = PSP_Pooling(nfilters=nfilters, depth=psp_depth, normalization=normalization)
        self.convs_up = []
        self.UpCombs = []
        for idx in range(depth - 1, 0, -1):
            nheads = nheads_start * 2 ** idx
            nfilters = nfilters_init * 2 ** (idx - 1)
            if verbose:
                print(
                    "depth:= {0}, nfilters: {1}, nheads::{2}, widths::{3}".format(2 * depth - idx - 1, nfilters, nheads,
                                                                                  widths[idx]))
            tnet = tf.keras.Sequential()
            for idx2 in range(widths[idx]):
                if model == 'CEECNetV1':
                    tnet.add(CEEC_unit_v1(nfilters=nfilters, nheads=nheads, ngroups=nheads,
                                          normalization=normalization, ftdepth=ftdepth,
                                          name=model + '_up' + str(idx) + str(idx2)))
                elif model == 'CEECNetV2':
                    tnet.add(CEEC_unit_v2(nfilters=nfilters, nheads=nheads, ngroups=nheads,
                                          normalization=normalization, ftdepth=ftdepth,
                                          name=model + '_up' + str(idx) + str(idx2)))
                else:
                    raise ValueError("I don't know requested model, aborting ... - Given model::{}".format(model))
            self.convs_up.append(tnet)
            if upFuse == True:
                self.UpCombs.append(combine_layers_wthFusion(nfilters=nfilters, nheads=nheads, normalization=normalization, ftdepth=ftdepth))
            else:
                self.UpCombs.append(combine_layers(nfilters, normalization=normalization))

    def call(self, input1, input2):
        conv1_t1 = self.conv_first(input1)
        conv1_t2 = self.conv_first(input2)
        fuse1 = self.fuse_first(conv1_t1, conv1_t2)
        fusions = []
        pools1 = conv1_t1
        pools2 = conv1_t2
        for idx in range(self.depth):
            conv1 = self.conv_dn[idx](pools1)
            # self.conv_dn[idx].summary()
            conv2 = self.conv_dn[idx](pools2)
            if idx < self.depth - 1:
                fusions = fusions + [self.fuse[idx](conv1, conv2)]
                # self.fuse[idx].summary()
                pools1 = self.pools[idx](conv1)
                pools2 = self.pools[idx](conv2)
                # self.pools[idx].summary()

        middle = tf.nn.relu(self.middle(tf.concat([conv1, conv2], axis=-1)))
        # self.middle.summary()
        fusions = fusions + [middle]
        convs_up = middle
        for idx in range(self.depth - 1):
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx - 2])
            # self.UpCombs[idx].summary()
            convs_up = self.convs_up[idx](convs_up)
            # self.convs_up[idx].summary()

        return convs_up, fuse1


