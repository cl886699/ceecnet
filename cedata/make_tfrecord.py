import tensorflow as tf
import numpy as np
import cv2
import os


def get_boundary(labels, _kernel_size=(3, 3)):
    label = labels.copy()
    for channel in range(label.shape[-1]):
        temp = cv2.Canny(label[..., channel], 0, 1)
        label[..., channel] = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_CROSS, _kernel_size), iterations=1)
    label = label.astype(np.float32)
    label /= 255.
    label = label.astype(np.uint8)
    return label


def get_distance(labels):
    label = labels.copy()
    dists = np.empty_like(label, dtype=np.float32)
    for channel in range(label.shape[-1]):
        dist = cv2.distanceTransform(label[..., channel], cv2.DIST_L2, 0)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dists[..., channel] = dist
    dists = dists * 100.
    dists = dists.astype(np.uint8)
    return dists


def load_data(filename):
    filename = filename.numpy().decode()
    a_pth = os.path.join(os.path.join(path, 'A'), filename)
    b_pth = os.path.join(os.path.join(path, 'B'), filename)
    label_pth = os.path.join(os.path.join(path, 'label'), filename)
    label = cv2.imread(label_pth, cv2.IMREAD_GRAYSCALE)
    label[label > 0] = 1
    label = np.eye(numclass, dtype=np.uint8)[label]
    distance = get_distance(label).astype(save_dtype)
    boundary = get_boundary(label, (3, 3)).astype(save_dtype)
    imagea = cv2.imread(a_pth).astype(save_dtype)
    imageb = cv2.imread(b_pth).astype(save_dtype)
    label = label.astype(save_dtype)
    return tf.concat([imagea, imageb, label, boundary, distance], axis=-1)


def save_dataset(dataset, save_pth):
    print(f"saving to {save_pth}...")
    serialized_ds = dataset.map(lambda tensor: tf.io.serialize_tensor(tensor))
    tfrec = tf.data.experimental.TFRecordWriter(save_pth)
    tfrec.write(serialized_ds)


def make_tfrecords():
    label_pth = os.path.join(path, 'label')
    files = [file for file in os.listdir(label_pth) if file.endswith('.png')]
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(lambda x: tf.py_function(load_data, [x], [save_dtype]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tfrecord_name = os.path.join(path, str(len(files)) + '.tfrec')
    save_dataset(dataset, tfrecord_name)


if __name__ == '__main__':
    path = r'D:\chrome_download\levir-cd\val\split'
    numclass = 2
    save_dtype = "float16"
    make_tfrecords()

    # path =r'D:\chrome_download\levir-cd\train\label'
    # files = [file for file in os.listdir(path) if file.endswith('.png')]
    # for file in files:
    #     label = cv2.imread(os.path.join(path, file))
    #     distance = get_distance(label)
    #     boundary = get_boundary(label, (3, 3))
    #     print(np.max(label), ' ', np.max(distance), ' ', np.max(boundary))
    # # np.concatenate()
    # image = np.column_stack([label, distance, boundary])
    # print(image.shape)
    # cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
