import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class MNISTDataset(object):

    def __init__(self, data_path="./train", resolution="32x32", train_set=True):
        mnist = input_data.read_data_sets(data_path, one_hot=True)
        self.res_tuple = [int(x_s) for x_s in resolution.split('x')]
        self.mnist_dataset = None
        if train_set:
            self.mnist_dataset = mnist.train
        else:
            self.mnist_dataset = mnist.test
        self.data_path = data_path

    def next_batch(self, number):
        batch_x, batch_y = self.mnist_dataset.next_batch(number)
        batch_matrix = np.zeros((number, self.res_tuple[0], self.res_tuple[1], 1))
        batch_x = batch_x.reshape([number, ] + [28, 28, 1])
        for idx in range(0, number):
            img = cv2.resize(batch_x[idx], tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
            batch_matrix[idx] = img[:, :, np.newaxis]
        return batch_matrix, batch_y
