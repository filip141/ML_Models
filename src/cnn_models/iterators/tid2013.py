import os
import cv2
import random
import numpy as np


class TID2013Dataset(object):

    def __init__(self, data_path="./train", new_resolution="320x320", patches=None,
                 patches_method='split', no_patches=32, is_train=True):
        self.data_path = data_path
        self.images_mos = []
        self.position = 0
        self.no_patches = no_patches
        self.pathes_method = patches_method
        self.is_train = is_train
        self.patches = patches
        self.new_resolution = new_resolution
        self.mos_file_path = os.path.join(data_path, "mos_with_names.txt")
        self.images_path = os.path.join(data_path, "distorted_images")

        # Define resolution for image
        if new_resolution is None:
            self.res_tuple = None
        else:
            self.res_tuple = [int(x) for x in new_resolution.split("x")]

        # Define resolution for patches
        if patches is None:
            self.patch_res = None
        else:
            self.patch_res = [int(x) for x in patches.split("x")]

        labels_file = open(self.mos_file_path, 'r')
        mos_file_ll = [f_x.split(" ") for f_x in labels_file.readlines()]
        self.images_mos = [(os.path.join(self.images_path, y[:-1]), float(x)) for x, y in mos_file_ll]

        # Divide into test and train
        test_len = int(0.2 * len(self.images_mos))
        self.train_mos = self.images_mos[test_len:]
        self.test_mos = self.images_mos[:test_len]
        random.shuffle(self.train_mos)
        random.shuffle(self.test_mos)

    def next_batch(self, number):
        if self.is_train:
            img_base = self.train_mos
        else:
            img_base = self.test_mos
        batch_labels = np.zeros((number, 1))
        if self.patches is None:
            batch_matrix = np.zeros((number, self.res_tuple[0], self.res_tuple[1], 3))
        else:
            batch_matrix = np.zeros((number, self.patch_res[0], self.patch_res[1], 3))

        # iterate over elements
        img_idx = 0
        while img_idx < number:
            im_path, im_mos = img_base[self.position]
            live_img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            if self.new_resolution is None:
                live_img_trans = live_img
            else:
                live_img_trans = cv2.resize(live_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
            img_shape = live_img_trans.shape

            # Return image or patches
            if self.patches is None:
                batch_matrix[img_idx] = live_img_trans
                batch_labels[img_idx] = im_mos
                img_idx += 1
            else:
                n_patches_w = int(img_shape[1] / float(self.patch_res[0]))
                n_patches_h = int(img_shape[0] / float(self.patch_res[1]))
                if self.pathes_method == 'split':
                    # Create patches
                    for im_x in range(n_patches_h):
                        for im_y in range(n_patches_w):
                            img = live_img_trans[
                                  im_x * self.patch_res[0]: (im_x + 1) * self.patch_res[0],
                                  im_y * self.patch_res[1]: (im_y + 1) * self.patch_res[1]
                                  ]
                            batch_matrix[img_idx] = img
                            batch_labels[img_idx] = im_mos
                            img_idx += 1
                            if img_idx > number:
                                return batch_matrix, batch_labels
                elif self.pathes_method == 'random':
                    for p_idx in range(0, self.no_patches):
                        w_pos = random.randint(0, img_shape[1] - self.patch_res[0])
                        h_pos = random.randint(0, img_shape[0] - self.patch_res[1])
                        img = live_img_trans[h_pos:h_pos + self.patch_res[1], w_pos:w_pos + self.patch_res[0]]
                        batch_matrix[img_idx] = img
                        batch_labels[img_idx] = im_mos
                        img_idx += 1
                        if img_idx >= number:
                            return batch_matrix, batch_labels
                else:
                    raise AttributeError("Method for splitting patches not defined.")
            self.position += 1
            self.position = self.position if self.position < len(img_base) else 0
        return batch_matrix, batch_labels