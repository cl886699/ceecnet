# coding=utf-8
import os, sys
import gc
import time
import datetime
import cv2
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


if os.name == 'nt':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"  #
import numpy as np
from cedata.parse_tfrecord import get_dataset
from cemodels.mantis.mantis_dn import *
per_batch = 1



if __name__ == '__main__':
    TRAIN_DATASET_DIR = [
        r'D:\chrome_download\levir-cd\train7120.tfrec',
    ]
    VAL_DATASET_DIR = [
        r'D:\chrome_download\levir-cd\val1024.tfrec',
    ]
    TRAIN_SIZE, VAL_SIZE = 7120, 1024

    train_dataset, val_dataset = get_dataset(train_dir=TRAIN_DATASET_DIR, val_dir=VAL_DATASET_DIR,
                                             total_batch=per_batch,
                                             shuffle=1000)
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
    net.load_weights('./train_dir/weights/epoch_1_loss_0.171_train.h5')
    # imga, imgb, label, boundary, distance = next(iter(train_dataset))
    # for idxxx in range(10):
    for train_indx, inputs in enumerate(train_dataset):
        imga, imgb, label, boundary, distance = inputs
        lbl = tf.squeeze(label, axis=0).numpy()
        lbl = lbl[..., 1] * 255
        lbl = lbl.astype("uint8")
        bound = tf.squeeze(boundary, axis=0).numpy()
        distance = tf.squeeze(distance, axis=0).numpy()
        bound = bound[..., 0] * 255
        bound = bound.astype("uint8")
        distance = distance[..., 1] * 255
        distance = distance.astype("uint8")
        label_show = np.hstack((lbl, bound, distance))
        cv2.imshow('lbl', label_show)
        preseg, preboundary, predists = net(imga, imgb)
        preseg = tf.squeeze(preseg, axis=0).numpy()
        preseg = preseg[..., 1] * 255
        preseg[preseg >= 128] = 255
        preseg[preseg < 128] = 0
        preseg = preseg.astype("uint8")
        preboundary = tf.squeeze(preboundary, axis=0).numpy()
        predists = tf.squeeze(predists, axis=0).numpy()
        preboundary = preboundary[..., 0] * 255
        preboundary = preboundary.astype("uint8")
        predists = predists[..., 1] * 255
        predists = predists.astype("uint8")
        pred_show = np.hstack((preseg, preboundary, predists))
        cv2.imshow('pred', pred_show)
        cv2.waitKey(0)

        # print(preseg)
