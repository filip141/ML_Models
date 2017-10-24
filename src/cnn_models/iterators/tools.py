import os
import cv2
import random
import numpy as np

CIFAR10_LABELS = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5,
                  "frog": 6, "horse": 7, "ship": 8, "truck": 9}


class ImageIterator(object):

    def __init__(self, iterator, rotate=10, translate=10, max_zoom=5):
        self.iterator = iterator
        self.rotate = rotate
        self.max_zoom = max_zoom
        self.translate = translate

    def next_batch(self, number):
        # iterate over elements
        batch_x, batch_y = self.iterator.next_batch(number=number)
        res_tuple = batch_x.shape[1:3]
        for img_idx in range(1, number):
            ds_img = batch_x[img_idx]
            random_flip = random.randint(0, 1)
            random_rotate = random.randint(0, 1)
            random_transform = random.randint(0, 1)
            random_zoom = random.randint(0, 1)
            if random_flip == 1:
                ds_img = cv2.flip(ds_img, 1)
            elif random_rotate == 1:
                angle_rot = self.rotate
                angle = random.randint(0, 2 * angle_rot) - angle_rot
                image_center = tuple(np.array(ds_img.shape[:2]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                ds_img = cv2.warpAffine(ds_img, rot_mat, ds_img.shape[:2], flags=cv2.INTER_LINEAR)
            elif random_transform == 1:
                move_px = self.translate
                m1 = random.randint(0, 2 * move_px) - move_px
                m2 = random.randint(0, 2 * move_px) - move_px
                num_rows, num_cols = ds_img.shape[:2]
                translation_matrix = np.float32([[1, 0, m1], [0, 1, m2]])
                ds_img = cv2.warpAffine(ds_img, translation_matrix, (num_cols, num_rows))
            elif random_zoom == 1:
                m1_1 = random.randint(1, self.max_zoom)
                m1_2 = random.randint(1, self.max_zoom)
                m2_1 = random.randint(1, self.max_zoom)
                m2_2 = random.randint(1, self.max_zoom)
                ds_img = cv2.resize(ds_img[m1_1:-m1_2, m2_1:-m2_2], tuple(res_tuple), interpolation=cv2.INTER_AREA)
            batch_x[img_idx] = ds_img
        return batch_x, batch_y

