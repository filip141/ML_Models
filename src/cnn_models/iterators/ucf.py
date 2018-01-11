import os
import cv2
import random
import numpy as np
from cnn_models.iterators.tools import FFMPEGVideoReader

import matplotlib.pyplot as plt


class UCF101(object):

    def __init__(self, ffmpeg_path, data_path, num_frames=25, resolution="320x240", resize="128x128", min_frames=75):
        self.ffmpeg = ffmpeg_path
        self.data_path = data_path

        self.min_frames = min_frames
        self.num_frames = num_frames
        self.resolution_text = resolution
        self.resize = [int(x) for x in resize.split("x")]
        self.resolution = [int(x) for x in resolution.split("x")]
        self.categories = self.get_categories()

    def get_categories(self):
        dirs_in_path = sorted(os.listdir(self.data_path))
        return dirs_in_path

    def next_batch(self, number):
        batch_labels = np.zeros((number, 101))
        batch_matrix = np.zeros((number, self.resize[1], self.resize[0], self.num_frames, 3))

        for movie_idx in range(number):
            # Get category
            cat_idx = random.randint(0, 100)
            batch_labels[movie_idx, cat_idx] = 1.0

            # Specify category path
            cat_path = os.path.join(self.data_path, self.categories[cat_idx])
            movies_list = os.listdir(cat_path)
            rand_movie = random.choice(movies_list)
            movie_path = os.path.join(cat_path, rand_movie)

            # Create FFMPEG instance
            ffmpeg = FFMPEGVideoReader(self.ffmpeg, movie_path, resolution=self.resolution_text)
            movie_len = ffmpeg.length(25)
            if movie_len < self.min_frames:
                continue
            movie_start = random.randint(0, movie_len - self.num_frames)
            ffmpeg.seek(movie_start)
            # Create instance in batch
            for frame_idx in range(self.num_frames):
                frame = ffmpeg.next_frame()
                if frame is None:
                    continue
                frame = cv2.resize(frame, tuple(self.resize), interpolation=cv2.INTER_AREA)
                max_val = np.max(frame)
                max_val = max_val if max_val != 0 else 1.0
                batch_matrix[movie_idx, :, :, frame_idx, :] = frame / float(max_val)
            ffmpeg.kill()
        return batch_matrix, batch_labels


if __name__ == '__main__':
    ucf = UCF101(ffmpeg_path="/usr/bin/ffmpeg", data_path="/home/filip141/Datasets/UCF-101", resolution="320x240",
                 resize="128x128")
    movies = ucf.next_batch(10)
    print(movies[1])
    plt.imshow(movies[0][5, :, :, 5, :])
    plt.show()