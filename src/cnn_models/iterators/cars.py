import os
import cv2
import random
import numpy as np


def sort_images(img_s):
    return img_s[:-4]


class CarsDataset(object):

    def __init__(self, data_path="./train", resolution="32x32", normalize_max=True):
        self.position = 0
        self.data_path = data_path
        self.normalize_max = normalize_max
        self.dataset_imgs = os.listdir(data_path)
        self.dataset_len = len(self.dataset_imgs)
        self.dividing_point = int(0.1 * self.dataset_len)
        self.dataset_imgs.sort(key=sort_images)
        random.shuffle(self.dataset_imgs)
        self.res_tuple = [int(x_s) for x_s in resolution.split('x')]

    def next_batch(self, number):
        first_file_name = self.dataset_imgs[self.position]
        first_path = os.path.join(self.data_path, first_file_name)

        #  Read image
        first_img = cv2.cvtColor(cv2.imread(first_path), cv2.COLOR_BGR2RGB)
        first_img = cv2.resize(first_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
        first_img_shape = first_img.shape
        first_one_hot_labels = np.zeros((10,))
        batch_labels = np.zeros((number, 10))
        batch_matrix = np.zeros((number, first_img_shape[0], first_img_shape[1], first_img_shape[2]))
        if self.normalize_max:
            batch_matrix[0] = first_img.astype('float32') / np.max(first_img)
        else:
            batch_matrix[0] = first_img.astype('float32')
        batch_labels[0] = first_one_hot_labels
        self.position += 1
        self.position = self.position if self.position < len(self.dataset_imgs) else 0

        # iterate over elements
        for img_idx in range(1, number):
            img_name = self.dataset_imgs[self.position]
            img_path = os.path.join(self.data_path, img_name)
            ds_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            ds_img = cv2.resize(ds_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)

            one_hot_labels = np.zeros((10, ))
            if self.normalize_max:
                batch_matrix[img_idx] = ds_img.astype('float32') / np.max(ds_img)
            else:
                batch_matrix[img_idx] = ds_img.astype('float32')
            batch_labels[img_idx] = one_hot_labels
            self.position += 1
            self.position = self.position if self.position < len(self.dataset_imgs) else 0
        return batch_matrix, batch_labels
