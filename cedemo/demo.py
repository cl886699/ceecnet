from cemodels.mantis.mantis_dn import *


# for i in range(10):
#     if tf.random.uniform(shape=()) > 0.5:
#         print('a')
#     else:
#         print("b")
#
# import sys
# sys.exit()


# D6nf32 example
depth=6
# groups = 32,
# axis = -1,
# epsilon = 1e-3,
# normalization = dict(normalization='group_norm', groups=4, epsilon=1e-5)
normalization=dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5)
ftdepth=5
NClasses=2
nfilters_init=32
psp_depth=4
nheads_start=4


net = mantis_dn_cmtsk(nfilters_init=nfilters_init, NClasses=NClasses,
                      depth=depth, ftdepth=ftdepth, model='CEECNetV1',
                      normalization=normalization,
                      psp_depth=psp_depth, nheads_start=nheads_start)

print("*"*10, "example outputs", "*"*10)
BatchSize = 1
img_size=256
NChannels = 3
input_img_1 = tf.random.uniform(shape=[BatchSize, img_size, img_size, NChannels], seed=1000)
input_img_2 = tf.random.uniform(shape=[BatchSize, img_size, img_size, NChannels], seed=1000)
net.build(input_shape=[(BatchSize, img_size, img_size, NChannels), (BatchSize, img_size, img_size, NChannels)])
net.summary()
import sys
sys.exit()
outs = net(input_img_1, input_img_2)

weights = net.trainable_variables
for a in weights:
    print(a.name)

# import sys
# sys.exit()

print(tf.reduce_sum(outs[1]))
net.save_weights('tst.h5')
# net.features.save_weights('features.h5')
# net.head.save_weights('head.h5')
# net.features.load_weights('features.h5')
# net.head.load_weights('head.h5')
#
# outs = net(input_img_1, input_img_2)
# print(tf.reduce_sum(outs[1]))

net.summary()
import sys
sys.exit()


print(outs[0])
print(outs[1])
print(outs[2])
for out in outs:
    print(out.shape)
print("*"*10, "example outputs", "*"*10)

# net.features.summary()
# net.head.summary()
net.summary()
# for indx, para in enumerate(net.variables):
#     # print(para)
#     print(para.name + '_' + str(indx), ' ', tf.shape(para).numpy(), ' ', tf.size(para).numpy(), ' ', para.trainable)