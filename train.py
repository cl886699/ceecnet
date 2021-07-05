# coding=utf-8
import os, sys
import gc
import time
import datetime
import numpy as np
from cedata.parse_tfrecord import get_dataset
from cemodels.mantis.mantis_dn import *
from celoss.mtsk_loss import MTSK_loss

if os.name == 'nt':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"  #
per_batch = 2


class Trainer:
    def __init__(self, dis_strategy, ori_model, batch_size=1, num_devices=1, epoch=[0, 2000],
                 trian_dir='', warmup_step=1000, total_step=20000, base_lr=1e-2, warmup_lr=1e-3):
        self.dist_strategy = dis_strategy
        self.model = ori_model
        self.trian_dir = trian_dir
        self.epochs = epoch
        self.batch_size = batch_size
        self.num_devices = num_devices
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.warmup_steps = warmup_step
        self.total_steps = total_step
        self.optimizer = tf.keras.optimizers.SGD(self.warmup_lr, momentum=0.9, nesterov=True)
        self.loss = MTSK_loss()

    def compute_loss(self, y_true, y_pred):
        loss_segm, loss_bound, loss_dists = self.loss.call(y_true, y_pred)
        loss_dict = {"total_loss": (loss_segm + loss_bound + loss_dists) / 3.0, "loss_segm": loss_segm,
                     "loss_bound": loss_bound, "loss_dists": loss_dists}
        return loss_dict

    def train_step(self, imga, imgb, label, boundary, distance):
        with tf.GradientTape() as tape:
            preds = self.model(imga, imgb)
            loss_dict = self.compute_loss([label, boundary, distance], preds)
        grads = tape.gradient(loss_dict["total_loss"], self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_dict

    @tf.function
    def dist_train_step(self, imga, imgb, label, boundary, distance):
        loss_dict = self.dist_strategy.run(
            self.train_step,
            args=(imga, imgb, label, boundary, distance)
        )
        for keys in loss_dict:
            loss_dict[keys] = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_dict[keys], axis=None)
        return loss_dict

    def train(self, train_dts, val_dts):
        # train model
        train_dts = self.dist_strategy.experimental_distribute_dataset(train_dts)
        # not used
        val_dts = self.dist_strategy.experimental_distribute_dataset(val_dts)
        log_dir = self.trian_dir + '/log_dir/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(log_dir)
        train_step = 0
        loss_history = {'total_loss': 0.0, "loss_segm": 0.0, "loss_bound": 0.0, "loss_dists": 0.0, 'lr': 0.0}
        for epoch in range(self.epochs[0], self.epochs[1]):
            train_step_perepoch = 0
            total_train_loss_perepoch = 0
            for train_indx, inputs in enumerate(train_dts):
                imga, imgb, label, boundary, distance = inputs
                if tf.random.uniform(shape=()) > 0.5:
                    loss_dict = self.dist_train_step(imga, imgb, label, boundary, distance)
                else:
                    loss_dict = self.dist_train_step(imgb, imga, label, boundary, distance)
                loss_history['lr'] = self.optimizer.lr
                loss_str = ''
                with file_writer.as_default():
                    for keys in loss_dict.keys():
                        tf.summary.scalar(keys, loss_dict[keys], step=train_step)
                        loss_history[keys] += loss_dict[keys]
                        loss_str += keys + ': '
                        loss_str += str(loss_dict[keys].numpy()) + ' '
                    tf.summary.scalar('lr', loss_history['lr'], step=train_step)
                    loss_history['lr'] += self.optimizer.lr
                file_writer.flush()
                total_train_loss_perepoch += loss_dict['total_loss'].numpy()
                train_step += 1
                train_step_perepoch += 1
                if train_step < self.warmup_steps:
                    lr = train_step / self.warmup_steps * self.base_lr
                else:
                    lr = self.warmup_lr + 0.5 * (self.base_lr - self.warmup_lr) * (
                        (1 + tf.cos((train_step - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi))
                    )
                self.optimizer.lr.assign(lr)
                print('epoch:', epoch, 'step:', train_step, 'lr:', self.optimizer.lr.numpy(), loss_str)
            weights_dir = self.trian_dir + '/weights/epoch_' + str(epoch) + '_loss_'
            total_train_loss_perepoch /= train_step_perepoch
            self.model.save_weights(weights_dir + str(round(total_train_loss_perepoch, 3)) + "_train.h5")
            print(f'gc: {gc.collect()}')
        print(f'gc: {gc.collect()}')


if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    num_devices = strategy.num_replicas_in_sync
    if os.name == 'nt':
        print('platform: windows')

        TRAIN_DATASET_DIR = [
            r'D:\chrome_download\levir-cd\train7120.tfrec',
        ]
        VAL_DATASET_DIR = [
            r'D:\chrome_download\levir-cd\val1024.tfrec',
        ]
    else:
        print('platform: linux')

        TRAIN_DATASET_DIR = [
            r'/share/datasets/allen/levir-cd/train7120.tfrec',
        ]
        VAL_DATASET_DIR = [
            r'/share/datasets/allen/levir-cd/val1024.tfrec',
        ]

    with strategy.scope():
        TRAIN_SIZE, VAL_SIZE = 7120, 1024

        train_dataset, val_dataset = get_dataset(train_dir=TRAIN_DATASET_DIR, val_dir=VAL_DATASET_DIR,
                                                 total_batch=per_batch * num_devices,
                                                 shuffle=100)
        depth = 6
        normalization = dict(normalization='group_norm', groups=4, epsilon=1e-5)
        # normalization = dict(normalization="batch_norm", axis=-1, trainable=True, momentum=0.1, epsilon=1e-5)
        ftdepth = 5
        NClasses = 2
        nfilters_init = 32
        psp_depth = 4
        nheads_start = 4
        net = mantis_dn_cmtsk(nfilters_init=nfilters_init, NClasses=NClasses,
                              depth=depth, ftdepth=ftdepth, model='CEECNetV1',
                              normalization=normalization,
                              psp_depth=psp_depth, nheads_start=nheads_start)

        net(tf.random.normal(shape=(1, 256, 256, 3)), tf.random.normal(shape=(1, 256, 256, 3)))
        net.summary()

        trainer = Trainer(dis_strategy=strategy,
                          ori_model=net,
                          batch_size=per_batch * num_devices,
                          num_devices=num_devices,
                          epoch=[0, 500],
                          trian_dir='./train_dir',
                          warmup_step=500,
                          total_step=7120 / (per_batch * num_devices),
                          base_lr=1e-4,
                          warmup_lr=1e-5
                          )
        trainer.train(train_dataset, val_dataset)
