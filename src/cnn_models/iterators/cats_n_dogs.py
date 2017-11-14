import os
import cv2
import random
import numpy as np
from xml.dom import minidom


def sort_images(img_s):
    return int(img_s[:-4])


class CatsNDogsDataset(object):

    def __init__(self, data_path="./Images", resize_img="64x64", train_set=True, force_overfit=False,
                 divide_point=0.1):
        self.data_path = data_path
        self.divide_point = divide_point
        self.labels = {"cat": 0, "dog": 1}
        self.data_list = []

        # Cats and dogs train test set
        dogs_imgs_path = os.path.join(data_path, "Dog")
        cats_imgs_path = os.path.join(data_path, "Cat")
        cats_files_d = [x for x in os.listdir(cats_imgs_path) if ".jpg" in x]
        dogs_files_d = [x for x in os.listdir(dogs_imgs_path) if ".jpg" in x]
        cats_files_in_dir = sorted(cats_files_d, key=sort_images)
        dogs_files_in_dir = sorted(dogs_files_d, key=sort_images)

        div_point_1 = int(self.divide_point * len(cats_files_in_dir))
        div_point_2 = int(self.divide_point * len(dogs_files_in_dir))

        if train_set:
            cats_files_in_dir = cats_files_in_dir[div_point_1:]
            dogs_files_in_dir = dogs_files_in_dir[div_point_2:]
        else:
            cats_files_in_dir = cats_files_in_dir[:div_point_1]
            dogs_files_in_dir = dogs_files_in_dir[:div_point_2]

        # read files in directory
        for cat_file in cats_files_in_dir:
            self.data_list.append((os.path.join(cats_imgs_path, cat_file), 0))
        for dog_file in dogs_files_in_dir:
            self.data_list.append((os.path.join(dogs_imgs_path, dog_file), 1))

        self.res_tuple = [int(x_s) for x_s in resize_img.split('x')]
        self.train_len = len(self.data_list)
        random.shuffle(self.data_list)

        if force_overfit:
            self.data_list = self.data_list[:20]
        self.position = 0

    def read_image(self):
        try:
            img_path, img_label = self.data_list[self.position]
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        except Exception:
            self.position = self.position + 1
            img_path, img_label = self.data_list[self.position]
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.position += 1
        self.position = self.position if self.position < len(self.data_list) else 0
        return img, img_label

    def next_batch(self, number):
        first_img, first_label = self.read_image()
        first_img = cv2.resize(first_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
        first_img_shape = first_img.shape
        batch_labels = np.zeros((number, 1))
        batch_matrix = np.zeros((number, first_img_shape[0], first_img_shape[1], first_img_shape[2]))
        batch_matrix[0] = (first_img.astype('float32') - np.mean(first_img)) / np.std(first_img)
        batch_labels[0] = first_label

        # iterate over elements
        img_idx = 0
        while img_idx < number:
            ds_img, img_label = self.read_image()
            ds_img = cv2.resize(ds_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)

            batch_matrix[img_idx] = (ds_img.astype('float32') - np.mean(ds_img)) / np.std(ds_img)
            batch_labels[img_idx] = img_label
            img_idx += 1
        return batch_matrix, batch_labels
