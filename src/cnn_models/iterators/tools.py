import os
import cv2
import signal
import random
import numpy as np
import subprocess as sp


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