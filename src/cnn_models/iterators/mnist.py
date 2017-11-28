import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class MNISTDataset(object):

    def __init__(self, data_path="./train", resolution="32x32", train_set=True, one_hot=False):
        mnist = input_data.read_data_sets(data_path, one_hot=True)
        self.res_tuple = [int(x_s) for x_s in resolution.split('x')]
        self.mnist_dataset = None
        self.one_hot = one_hot
        self.max_label = 10
        self.min_label = 0
        if train_set:
            self.mnist_dataset = mnist.train
        else:
            self.mnist_dataset = mnist.test
        self.data_path = data_path

    def next_batch(self, number):
        if self.one_hot:
            lab_size = (number, self.max_label)
        else:
            lab_size = (number,)
        batch_x, batch_y = self.mnist_dataset.next_batch(number)
        batch_matrix = np.zeros((number, self.res_tuple[0], self.res_tuple[1], 1))
        batch_labels = np.zeros(lab_size)
        batch_x = batch_x.reshape([number, ] + [28, 28, 1])
        for idx in range(0, number):
            img = cv2.resize(batch_x[idx], tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
            batch_matrix[idx] = img[:, :, np.newaxis]
            if self.one_hot:
                batch_labels[idx] = batch_y[idx]
            else:
                batch_labels[idx] = np.argmax(batch_y[idx])
        return batch_matrix, batch_labels
