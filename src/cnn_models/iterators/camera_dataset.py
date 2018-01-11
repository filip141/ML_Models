import os
import cv2
import random
import numpy as np


class CameraDataset(object):

    def __init__(self, data_path="./train", new_resolution="320x320", is_train=True):
        self.data_path = data_path
        self.is_train = is_train
        self.new_resolution = [int(x) for x in new_resolution.split("x")]
        self.labels = os.listdir(self.data_path)
        self.labels_idx = dict([(idx, x) for x, idx in enumerate(self.labels)])
        self.dataset_imgs = []
        self.position = 0

        # Collect Dataset
        for im_dir in self.labels:
            imgs_path = os.path.join(self.data_path, im_dir)
            for image_path in os.listdir(imgs_path):
                self.dataset_imgs.append((os.path.join(imgs_path, image_path), im_dir))

        # Divide into test and train
        test_len = int(0.2 * len(self.dataset_imgs))
        if is_train:
            self.dataset_imgs = self.dataset_imgs[test_len:]
        else:
            self.dataset_imgs = self.dataset_imgs[:test_len]
        random.shuffle(self.dataset_imgs)

    def next_batch(self, number):
        batch_labels = np.zeros((number, 10))
        batch_matrix = np.zeros((number, self.new_resolution[0], self.new_resolution[1], 3))

        # iterate over elements
        for img_idx in range(0, number):
            im_path, label_name = self.dataset_imgs[self.position]
            cam_img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            img_shape = cam_img.shape

            w_pos = random.randint(0, img_shape[1] - self.new_resolution[0])
            h_pos = random.randint(0, img_shape[0] - self.new_resolution[1])
            img = cam_img[h_pos:h_pos + self.new_resolution[1], w_pos:w_pos + self.new_resolution[0]]
            one_hot = np.zeros(shape=(10,))
            one_hot[self.labels_idx[label_name]] = 1.0

            random_flip = random.randint(0, 1)
            if random_flip:
                img = cv2.flip(img, 1)
            img = img / 255.0
            batch_matrix[img_idx] = img
            batch_labels[img_idx] = one_hot
            self.position += 1
            self.position = self.position if self.position < len(self.dataset_imgs) else 0
        return batch_matrix, batch_labels


if __name__ == '__main__':
    cam = CameraDataset(data_path="/home/filip141/Datasets/Camera", new_resolution="512x512")
    images = cam.next_batch(32)
    import matplotlib.pyplot as plt
    plt.imshow(images[0][0])
    plt.show()