import os
import cv2
import random
import numpy as np
from xml.dom import minidom


class DogsDataset(object):

    def __init__(self, data_path="./Images", labels_path="./Annotation", class_names="./class_names.txt",
                 resize_img="64x64", train_set=True, force_overfit=False):
        self.data_path = data_path
        self.labels_path = labels_path
        self.class_names = class_names
        self.file_list = []

        # read files in directory
        for folder, subs, files in os.walk(self.data_path):
            for file_name in files:
                self.file_list.append(os.path.join(folder, file_name))

        self.res_tuple = [int(x_s) for x_s in resize_img.split('x')]

        self.class_dict = dict([(x[:-1], im_idx) for im_idx, x in enumerate(open(class_names).readlines())])
        self.train_len = int(0.8 * len(self.file_list))
        random.shuffle(self.file_list)

        if force_overfit:
            self.file_list = self.file_list[:20]

        if train_set:
            self.file_list = self.file_list[:self.train_len]
        else:
            self.file_list = self.file_list[self.train_len:]

        self.position = 0

    def next_batch(self, number):
        first_fn = self.file_list[self.position]
        first_label = minidom.parse(first_fn.replace("Images", "Annotation")
                                    .strip(".jpg")).getElementsByTagName("name")[0].childNodes[0].toxml()\
            .lower().replace(" ", "_")
        first_path = os.path.join(self.data_path, first_fn)

        #  Read image
        first_img = cv2.cvtColor(cv2.imread(first_path), cv2.COLOR_BGR2RGB)
        first_img_res = cv2.resize(first_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
        first_one_hot_labels = np.zeros((len(self.class_dict), ))
        batch_labels = np.zeros((number, len(self.class_dict)))
        batch_matrix = np.zeros((number, self.res_tuple[0], self.res_tuple[1], 3))
        first_one_hot_labels[self.class_dict[first_label]] = 1.0
        batch_matrix[0] = (first_img_res.astype('float32') - np.mean(first_img_res)) / np.std(first_img_res)
        batch_labels[0] = first_one_hot_labels
        self.position += 1
        self.position = self.position if self.position < len(self.file_list) else 0

        # iterate over elements
        for img_idx in range(1, number):
            img_fn = self.file_list[self.position]
            parsed_xml_file = minidom.parse(img_fn.replace("Images", "Annotation").strip(".jpg"))
            img_label = parsed_xml_file.getElementsByTagName("name")[0].childNodes[0].toxml()\
                .lower().replace(" ", "_")
            img_path = os.path.join(self.data_path, img_fn)
            ds_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            random_augment = random.randint(0, 2)
            if random_augment == 1:
                ds_img = cv2.flip(ds_img, 1)
            elif random_augment == 2:
                xmin = int(parsed_xml_file.getElementsByTagName("xmin")[0].childNodes[0].toxml())
                ymin = int(parsed_xml_file.getElementsByTagName("ymin")[0].childNodes[0].toxml())
                xmax = int(parsed_xml_file.getElementsByTagName("xmax")[0].childNodes[0].toxml())
                ymax = int(parsed_xml_file.getElementsByTagName("ymax")[0].childNodes[0].toxml())
                ds_img = ds_img[ymin:ymax, xmin:xmax]

            img_res = cv2.resize(ds_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)

            one_hot_labels = np.zeros((len(self.class_dict),))
            one_hot_labels[self.class_dict[img_label]] = 1.0
            batch_matrix[img_idx] = (img_res.astype('float32') - np.mean(img_res)) / np.std(img_res)
            batch_labels[img_idx] = one_hot_labels
            self.position += 1
            self.position = self.position if self.position < len(self.file_list) else 0
        return batch_matrix, batch_labels
