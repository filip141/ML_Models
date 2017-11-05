import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class MNISTDataset(object):

    def __init__(self, data_path="./train", resolution="32x32", train_set=True, label_only=None):
        mnist = input_data.read_data_sets(data_path, one_hot=True)
        self.res_tuple = [int(x_s) for x_s in resolution.split('x')]
        self.mnist_dataset = None
        self.label_only = label_only
        if train_set:
            self.mnist_dataset = mnist.train
        else:
            self.mnist_dataset = mnist.test
        self.data_path = data_path

    def next_batch(self, number):
        idx_counter = 0
        counter_batch = 0
        batch_x, batch_y = self.mnist_dataset.next_batch(number)
        batch_matrix = np.zeros((number, self.res_tuple[0], self.res_tuple[1], 1))
        batch_x = batch_x.reshape([number, ] + [28, 28, 1])
        while counter_batch < number:
            if idx_counter == number:
                batch_x, batch_y = self.mnist_dataset.next_batch(number)
                batch_x = batch_x.reshape([number, ] + [28, 28, 1])
                idx_counter = 0
            img = cv2.resize(batch_x[idx_counter], tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
            img_m = (img - np.mean(img))
            img_m_n = img_m / np.max(img_m)
            if self.label_only is not None:
                act_label = np.argmax(batch_y[idx_counter])
                if self.label_only == act_label:
                    batch_matrix[counter_batch] = img_m_n[:, :, np.newaxis]
                    counter_batch += 1
            idx_counter += 1
        return batch_matrix, batch_y