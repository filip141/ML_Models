import os
import cv2
import copy
import random
import numpy as np
import subprocess as sp


CIFAR10_LABELS = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5,
                  "frog": 6, "horse": 7, "ship": 8, "truck": 9}


class ImageIterator(object):

    def __init__(self, iterator, rotate=0, translate=0, max_zoom=0, additive_noise=0, adjust_brightness=0,
                 adjust_contrast=0, preprocess="none", additional_data={}):
        self.iterator = iterator
        self.preprocess = preprocess
        self.rotate = rotate
        self.max_zoom = max_zoom
        self.additive_noise = additive_noise
        self.translate = translate
        self.adjust_brightness = adjust_brightness
        self.adjust_contrast = adjust_contrast
        self.additional_data = additional_data

    def preprocess_img(self, batch_x):
        eps = 0.0001
        if self.preprocess == "none":
            return batch_x
        elif self.preprocess == "lcn":
            k_param = self.additional_data.get("k_param", 7)
            ds_img_shape = batch_x[0].shape
            new_batch = copy.deepcopy(batch_x)
            for h_idx in range(ds_img_shape[0]):
                for w_idx in range(ds_img_shape[1]):
                    h_start = h_idx - int((k_param - 1) / 2.)
                    h_start = h_start if h_start >= 0 else 0
                    h_end = h_idx + int((k_param - 1) / 2.)
                    h_end = h_end if h_end < ds_img_shape[0] else ds_img_shape[0] - 1

                    w_start = w_idx - int((k_param - 1) / 2.)
                    w_start = w_start if w_start >= 0 else 0
                    w_end = w_idx + int((k_param - 1) / 2.)
                    w_end = w_end if w_end < ds_img_shape[1] else ds_img_shape[1] - 1

                    img_patch = batch_x[:, h_start: h_end, w_start: w_end, :]
                    patch_mean = np.mean(img_patch, axis=(1, 2))
                    patch_std = np.std(img_patch, axis=(1, 2))
                    new_batch[:, h_idx, w_idx, :] = (new_batch[:, h_idx, w_idx, :] - patch_mean) / (patch_std + eps)
            return new_batch

    def next_batch(self, number):
        # iterate over elements
        batch_x, batch_y = self.iterator.next_batch(number=number)
        batch_x = self.preprocess_img(batch_x)
        res_tuple = batch_x.shape[1:3]
        for img_idx in range(1, number):
            ds_img = batch_x[img_idx]
            random_flip = random.randint(0, 1)
            random_rotate = random.randint(0, 1) if self.rotate != 0 else 0
            random_transform = random.randint(0, 1) if self.translate != 0 else 0
            random_zoom = random.randint(0, 1) if self.max_zoom != 0 else 0
            random_noise = random.randint(0, 1) if self.additive_noise != 0 else 0
            random_brightness = random.randint(0, 1) if self.adjust_brightness != 0 else 0
            random_contrast = random.randint(0, 1) if self.adjust_contrast != 0 else 0
            if random_contrast == 1:
                contrast_val = random.randint(0, int(self.adjust_contrast * 100))
                ds_img = (1 + contrast_val / 100.0) * ds_img
            if random_brightness == 1:
                brightness_val = random.randint(0, int(self.adjust_brightness * 100))
                ds_img = ds_img + brightness_val / 100.0
            if random_flip == 1:
                ds_img = cv2.flip(ds_img, 1)
            if random_noise == 1:
                gaussian_noise = np.random.normal(scale=self.additive_noise, size=ds_img.shape)
                ds_img += gaussian_noise
            if random_rotate == 1:
                angle_rot = self.rotate
                angle = random.randint(0, 2 * angle_rot) - angle_rot
                image_center = tuple(np.array(ds_img.shape[:2]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                ds_img = cv2.warpAffine(ds_img, rot_mat, ds_img.shape[:2], flags=cv2.INTER_LINEAR)
            if random_transform == 1:
                move_px = self.translate
                m1 = random.randint(0, 2 * move_px) - move_px
                m2 = random.randint(0, 2 * move_px) - move_px
                num_rows, num_cols = ds_img.shape[:2]
                translation_matrix = np.float32([[1, 0, m1], [0, 1, m2]])
                ds_img = cv2.warpAffine(ds_img, translation_matrix, (num_cols, num_rows))
            if random_zoom == 1:
                m1_1 = random.randint(1, self.max_zoom)
                m1_2 = random.randint(1, self.max_zoom)
                m2_1 = random.randint(1, self.max_zoom)
                m2_2 = random.randint(1, self.max_zoom)
                ds_img = cv2.resize(ds_img[m1_1:-m1_2, m2_1:-m2_2], tuple(res_tuple), interpolation=cv2.INTER_AREA)
            if len(ds_img.shape) == 2:
                ds_img = ds_img[:, :, np.newaxis]
            batch_x[img_idx] = ds_img
        return batch_x, batch_y


class FFMPEGVideoReader(object):

    def __init__(self, ffmpeg_bin, video_path, resolution="1920x1080"):
        self.ffmpeg_bin = ffmpeg_bin
        self.video_path = video_path
        self.resolution = [int(x) for x in resolution.split("x")]
        self.command = [self.ffmpeg_bin,
                        '-i', self.video_path,
                        '-f', 'image2pipe',
                        '-pix_fmt', 'rgb24',
                        '-vcodec', 'rawvideo', '-']
        devnull = open(os.devnull, "w")
        self.ffmpeg_pipe = sp.Popen(self.command, stdout=sp.PIPE, bufsize=10**8, stderr=devnull)

    def length(self, fps):
        command = [self.ffmpeg_bin,
                   '-i', self.video_path]
        result = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
        duration = [x.decode("utf-8").split(",")[0].lstrip() for x in result.stdout.readlines()
                    if "Duration" in str(x)][0].split(": ")[1].split(":")
        duration = int(duration[0]) * 3600 + int(duration[1]) * 60 + float(duration[2])
        return int(duration * fps)

    def seek(self, number):
        self.ffmpeg_pipe.stdout.read(np.prod(self.resolution) * 3 * number)

    def next_frame(self):
        # Read bytes from ffmpeg pipe
        raw_image = self.ffmpeg_pipe.stdout.read(np.prod(self.resolution) * 3)

        # Convert to uint8 and reshape image
        image = np.fromstring(raw_image, dtype='uint8')
        if image.size == 0:
            return None
        image = image.reshape((self.resolution[1], self.resolution[0], 3))
        self.ffmpeg_pipe.stdout.flush()
        return image

    def kill(self):
        self.ffmpeg_pipe.kill()


class FFMPEGVideoWritter(object):

    def __init__(self, ffmpeg_bin, video_path, resolution="1920x1080"):
        self.ffmpeg_bin = ffmpeg_bin
        self.video_path = video_path
        self.resolution = [int(x) for x in resolution.split("x")]
        self.command = [self.ffmpeg_bin, '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-s', resolution,
                        '-pix_fmt', 'rgb24',
                        '-r', '24',  '-i', '-',
                        '-an',
                        '-vcodec', 'mpeg4',
                        video_path]
        self.ffmpeg_pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)

    def save_frame(self, data):
        self.ffmpeg_pipe.stdin.write(data.tobytes())

    def kill(self):
        self.ffmpeg_pipe.stdin.close()
        self.ffmpeg_pipe.stderr.close()
        self.ffmpeg_pipe.kill()


if __name__ == '__main__':
    from cnn_models.iterators.cifar import CIFARDataset
    from cnn_models.iterators.tools import ImageIterator
    cifar_train_path = "/home/phoenix/Datasets/cifar/train"
    cifar_test_path = "/home/phoenix/Datasets/cifar/test"
    cifar_train = ImageIterator(CIFARDataset(data_path=cifar_train_path, resolution="32x32", force_overfit=False),
                                rotate=15, translate=3, max_zoom=3, additive_noise=0.1)
    img = cifar_train.next_batch(1)
    import matplotlib.pyplot as plt