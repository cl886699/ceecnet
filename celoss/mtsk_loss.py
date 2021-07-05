from celoss.ftnmt_loss import *


class MTSK_loss():
    def __init__(self, depth=0):
        self.ftnmt = ftnmt_loss(depth=depth)

    def call(self, y_true, y_pred):
        pred_segm = y_pred[0]
        pred_bound = y_pred[1]
        pred_dists = y_pred[2]

        label_segm = y_true[0]
        label_bound = y_true[1]
        label_dists = y_true[2]

        loss_segm = self.ftnmt.call(pred_segm, label_segm)
        loss_bound = self.ftnmt.call(pred_bound, label_bound)
        loss_dists = self.ftnmt.call(pred_dists, label_dists)
        return loss_segm, loss_bound, loss_dists
        # loss_dict = {"total_loss": (loss_segm + loss_bound + loss_dists)/3.0, "loss_segm": loss_segm,
        #              "loss_bound": loss_bound, "loss_dists": loss_dists}
        # return loss_dict


if __name__ == '__main__':
    import numpy as np
    import cv2

    # a = np.zeros(shape=(256, 256, 1), dtype="uint8")
    # a[:50, :50,0]=255
    # cv2.imshow('a', a)
    # cv2.waitKey()

    alist = []
    blist = []
    counta = 0
    for i in range(256*256):
        if np.random.randint(1, 1001)>980:
            a = 1
            counta +=1
        else:
            a = 0
        b = 1 - a
        alist.append(a)
        blist.append(b)
    print(counta)
    atensor = tf.reshape(tf.constant(alist, tf.float32), shape=(256, 256))
    btensor = tf.reshape(tf.constant(blist, tf.float32), shape=(256, 256))
    a = tf.expand_dims(tf.stack([atensor, btensor], axis=-1), axis=0)
    b2 = tf.ones_like(a[..., 0])
    b1 = tf.zeros_like(a[..., 0])
    b = tf.stack([b1, b2], axis=-1)
    loss = MTSK_loss(depth=0)
    lossdd = loss.call([a, a, a], [b, b, b])
    print(lossdd)

    a = np.array([[[[0.5,0.5], [0, 1], [0, 0.5], [0, 1]], [[0,1], [1, 0], [0, 1], [1, 0]], [[1,0], [0, 1], [0, 1], [0, 1]], [[1,0], [0, 1], [0, 1], [0, 1]]]])
    b = np.array([[[[0,1], [1, 0], [0, 1], [1, 0]], [[1,0], [0, 1], [0, 1], [0, 1]], [[1,0], [0, 1], [0, 1], [0, 1]], [[0,1], [1, 0], [0, 1], [1, 0]]]])
    a = tf.convert_to_tensor(a, dtype=tf.float32)
    b = tf.convert_to_tensor(b, dtype=tf.float32)
    a = tf.reshape(a, shape=(1, 4, 4, 2))
    b = tf.reshape(b, shape=(1, 4, 4, 2))

    # a = tf.reshape(tf.range(0, 12*12*2), shape=(1, 12, 12, 2))
    a = tf.cast(a, tf.float32)
    # b = tf.reshape(tf.range(3, 12*12*2 + 3), shape=(1, 12, 12, 2))
    b = tf.cast(b, tf.float32)
    loss = MTSK_loss(depth=2)
    lossdd = loss.call([a, a, a], [b, b, b])
    print(lossdd)