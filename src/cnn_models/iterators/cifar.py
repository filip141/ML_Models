import os
import cv2
import random
import numpy as np

CIFAR10_LABELS = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5,
                  "frog": 6, "horse": 7, "ship": 8, "truck": 9}


class CIFARDataset(object):

    def __init__(self, data_path="./train", resolution="32x32", force_overfit=False):
        self.data_path = data_path
        self.position = 0
        self.files_list = os.listdir(data_path)
        self.res_tuple = [int(x_s) for x_s in resolution.split('x')]
        if force_overfit:
            self.files_list = self.files_list[:20]
        random.shuffle(self.files_list)

    def next_batch(self, number):
        first_file_name = self.files_list[self.position]
        _, first_label = os.path.split(first_file_name)[1].split(".")[0].split("_")
        first_path = os.path.join(self.data_path, first_file_name)

        #  Read image
        first_img = cv2.cvtColor(cv2.imread(first_path), cv2.COLOR_BGR2RGB)
        first_img = cv2.resize(first_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
        first_img_shape = first_img.shape
        first_one_hot_labels = np.zeros((10, ))
        batch_labels = np.zeros((number, 10))
        batch_matrix = np.zeros((number, first_img_shape[0], first_img_shape[1], first_img_shape[2]))
        first_one_hot_labels[CIFAR10_LABELS[first_label]] = 1.0
        batch_matrix[0] = first_img / 255.0
        batch_labels[0] = first_one_hot_labels
        self.position += 1
        self.position = self.position if self.position < len(self.files_list) else 0

        # iterate over elements
        for img_idx in range(1, number):
            img_name = self.files_list[self.position]
            _, img_label = os.path.split(img_name)[1].split(".")[0].split("_")
            img_path = os.path.join(self.data_path, img_name)
            ds_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            ds_img = cv2.resize(ds_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)

            one_hot_labels = np.zeros((10, ))
            one_hot_labels[CIFAR10_LABELS[img_label]] = 1.0
            batch_matrix[img_idx] = ds_img / 255.0
            batch_labels[img_idx] = one_hot_labels
            self.position += 1
            self.position = self.position if self.position < len(self.files_list) else 0
        return batch_matrix, batch_labels
