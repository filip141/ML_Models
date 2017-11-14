import cv2
import json
import scipy.signal
import numpy as np


class IcebergDataset(object):

    def __init__(self, json_path="./train", resolution="75x75", normalize_max=True, is_test=False,
                 batch_out='angle_concat', divide_point=0.1):
        self.json_path = json_path
        self.position = 0
        self.batch_out = batch_out
        self.normalize_max = normalize_max
        with open(self.json_path, 'r') as data_file:
            self.json_data = json.load(data_file)
        self.dataset_len = len(self.json_data)
        divide_point_len = int(divide_point * self.dataset_len)
        if is_test:
            self.json_data = self.json_data[:divide_point_len]
        else:
            self.json_data = self.json_data[divide_point_len:]
        self.res_tuple = [int(x_s) for x_s in resolution.split('x')]

    def next_batch(self, number):
        last_dim_dict = {"angle_concat": 3, "mean_dim": 1, "mean_median": 1}
        batch_matrix = np.zeros((number, self.res_tuple[0], self.res_tuple[1], last_dim_dict[self.batch_out]))
        batch_labels = np.zeros((number, 1))
        for idx in range(0, number):
            # Read and prepare image
            img_data = self.json_data[self.position]
            img_band_1 = np.array(img_data["band_1"])
            img_band_2 = np.array(img_data["band_2"])
            img_side = int(np.sqrt(img_band_1.shape[0]))
            img_concat = np.zeros([img_side, img_side, 2])
            img_concat[:, :, 0] = img_band_1.reshape([img_side, img_side])
            img_concat[:, :, 1] = img_band_2.reshape([img_side, img_side])
            img = cv2.resize(img_concat, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
            if self.normalize_max:
                img_n_m = img - np.min(img)
                img_n_m /= np.max(img_n_m)
            if self.batch_out == "angle_concat":
                inc_angle = img_data['inc_angle'] if img_data['inc_angle'] != 'na' else 0.0
                additional_dim = (inc_angle / 360.0) * np.ones([img_side, img_side])
                img_n_m = np.concatenate([img_n_m, additional_dim[:, :, np.newaxis]], axis=2)
            elif self.batch_out == "mean_dim":
                img_n_m = np.mean(img_n_m, axis=2)[:, :, np.newaxis]
            elif self.batch_out == "mean_median":
                img_n_m = np.mean(img_n_m, axis=2)
                img_n_m = scipy.signal.medfilt(img_n_m, kernel_size=3)[:, :, np.newaxis]
            else:
                raise AttributeError("Image processing method not defined")
            batch_matrix[idx] = img_n_m
            # Read labels
            batch_labels[idx] = img_data["is_iceberg"]
            self.position += 1
            self.position = self.position if self.position < len(self.json_data) else 0
        return batch_matrix, batch_labels


if __name__ == '__main__':
    json_data_train = "/home/filip/Datasets/Iceberg_data/train/processed/train.json"
    json_data_test = "/home/filip/Datasets/Iceberg_data/test/processed/test.json"
    iceberg_db = IcebergDataset(json_path=json_data_train, batch_out="mean_dim")
    iceberg_db.next_batch(100)