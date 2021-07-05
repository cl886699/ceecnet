import tensorflow as tf
import os


def get_dataset(train_dir, val_dir, total_batch, shuffle=1000):
    def parser(raw):
        tensors = raw
        parsed = tf.cast(tf.io.parse_tensor(tensors, out_type=tf.float16), dtype=tf.float32)
        # tf.print(parsed)
        imga = parsed[..., :3]/255.0
        imgb = parsed[..., 3:6]/255.0
        label = parsed[..., 6:8]
        boundary = parsed[..., 8:10]
        distance = parsed[..., 10:12]/100.0
        return imga, imgb, label, boundary, distance

    def random_flip_horizontal(imga, imgb, lbl, boun, dist):
        rn = tf.random.uniform(shape=(), maxval=1)
        return tf.cond(rn < 0.5,
                       lambda: (imga, imgb, lbl, boun, dist),
                       lambda: (tf.image.flip_left_right(imga),
                                tf.image.flip_left_right(imgb),
                                tf.image.flip_left_right(lbl),
                                tf.image.flip_left_right(boun),
                                tf.image.flip_left_right(dist)
                                ))

    def random_flip_vertical(imga, imgb, lbl, boun, dist):
        rn = tf.random.uniform(shape=(), maxval=1)
        return tf.cond(rn < 0.5,
                       lambda: (imga, imgb, lbl, boun, dist),
                       lambda: (tf.image.flip_up_down(imga),
                                tf.image.flip_up_down(imga),
                                tf.image.flip_up_down(lbl),
                                tf.image.flip_up_down(boun),
                                tf.image.flip_up_down(dist),
                                ))

    train_ds = tf.data.TFRecordDataset([train_dir]).map(parser, num_parallel_calls=-1)
    val_ds = tf.data.TFRecordDataset([val_dir]).map(parser, num_parallel_calls=-1)

    train_ds = train_ds.map(lambda imga, imgb, lbl, boun, dist:
                            random_flip_horizontal(imga, imgb, lbl, boun, dist),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(lambda imga, imgb, lbl, boun, dist:
                            random_flip_vertical(imga, imgb, lbl, boun, dist),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.shuffle(buffer_size=shuffle).batch(total_batch).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(total_batch).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds


if __name__ == '__main__':
    train_dir = r'D:\chrome_download\levir-cd\train7120.tfrec'
    val_dir = r'D:\chrome_download\levir-cd\val1024.tfrec'
    total_batch = 1

    image_shape = (256, 256, 3)
    import cv2, sys
    import numpy as np
    # a = np.array([[0, 1], [0, 1]])
    # b = np.eye(2, dtype=np.uint8)[a]
    # print(a)
    # print(b)
    # sys.exit()
    train, val = get_dataset(train_dir, val_dir, total_batch, shuffle=10)
    for imga, imgb, lbl, bound, dist in val:
        imga = tf.squeeze(imga * 255.0, axis=0).numpy().astype("uint8")
        imgb = tf.squeeze(imgb * 255.0, axis=0).numpy().astype("uint8")

        # cv2.imshow('a', cv2.hconcat([imga, imgb]))
        lbl = tf.squeeze(lbl, axis=0).numpy().astype("uint8")
        lable = lbl[..., 1]
        lable[lable>0] = 255

        bound = tf.squeeze(bound, axis=0).numpy().astype("uint8")
        bound = bound[..., 0]
        bound[bound>0] = 255

        dist = tf.squeeze(dist * 100.0, axis=0).numpy().astype("uint8")
        dist = dist[..., 1]

        lbl_show = np.hconcat([lable, bound, dist])
        cv2.imshow('lbl', lable)
        cv2.imshow('bound', bound)
        cv2.imshow('dist', dist)
        cv2.waitKey(0)