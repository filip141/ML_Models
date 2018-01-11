import os
import logging
import numpy as np
import tensorflow as tf
from simple_network.models import MMGANScheme
from simple_network.layers import Deconvolution3DLayer, FullyConnectedLayer, Convolutional3DLayer, \
    ReshapeLayer, LeakyReluLayer, SingleBatchNormLayer, DropoutLayer, Flatten, SigmoidLayer, \
    LinearLayer, ReluLayer, GlobalAveragePoolingLayer, TanhLayer

from cnn_models.iterators.tools import FFMPEGVideoWritter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_video(data, epoch_id):
    video_path = "/home/filip141/tensor_logs/GAN_UFC/video_{}.h264".format(epoch_id)
    mp4_video_path = "/home/filip141/tensor_logs/GAN_UFC/video_{}.mp4".format(epoch_id)
    ffmpeg = FFMPEGVideoWritter("/usr/bin/ffmpeg", video_path, resolution="64x64")
    for frame_idx in range(0, data.shape[3]):
        ffmpeg.save_frame((data[0, :, :, frame_idx, :] * 255.0).astype(np.uint8))
    ffmpeg.kill()
    os.rename(video_path, mp4_video_path)


class GANNetwork(MMGANScheme):

    def __init__(self, generator_input_size, discriminator_input_size, log_path, batch_size=None, labels=None,
                 labels_size=10):
        super(GANNetwork, self).__init__(generator_input_size, discriminator_input_size, log_path, batch_size,
                                         labels=labels, labels_size=labels_size)
        self.batch_size = batch_size

    def build_generator(self, generator):
        generator.add(FullyConnectedLayer(out_neurons=12288, initializer="xavier",
                                          name='fully_connected_g_1'))
        generator.add(LeakyReluLayer(name="relu_1"))
        generator.add(ReshapeLayer(output_shape=[2, 2, 3, 1024]))
        generator.add(Deconvolution3DLayer([5, 5, 5, 512], output_shape=[4, 4, 6, 512], initializer="xavier",
                                           name='deconv_layer_g_2', stride=2, stride_d=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_2"))
        generator.add(LeakyReluLayer(name="relu_2"))

        generator.add(Deconvolution3DLayer([5, 5, 5, 256], output_shape=[8, 8, 12, 256], initializer="xavier",
                                           name='deconv_layer_g_3', stride=2, stride_d=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_3"))
        generator.add(LeakyReluLayer(name="relu_3"))

        generator.add(Deconvolution3DLayer([5, 5, 5, 128], output_shape=[16, 16, 24, 128], initializer="xavier",
                                           name='deconv_layer_g_4', stride=2, stride_d=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_4"))
        generator.add(LeakyReluLayer(name="relu_4"))

        generator.add(Deconvolution3DLayer([5, 5, 5, 64], output_shape=[32, 32, 48, 64], initializer="xavier",
                                           name='deconv_layer_g_5', stride=2, stride_d=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_5"))
        generator.add(LeakyReluLayer(name="relu_5"))

        generator.add(Deconvolution3DLayer([5, 5, 5, 32], output_shape=[64, 64, 48, 32], initializer="xavier",
                                           name='deconv_layer_g_6', stride=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_6"))
        generator.add(LeakyReluLayer(name="relu_6"))

        generator.add(Deconvolution3DLayer([3, 3, 3, 3], output_shape=[64, 64, 48, 3], initializer="xavier",
                                           name='deconv_layer_g_7', stride=1, batch_size=self.batch_size))
        generator.add(TanhLayer(name="tanh_g_7"))

    def build_discriminator(self, discriminator):
        discriminator.add(Convolutional3DLayer([5, 5, 5, 128], initializer="xavier", name='convo_layer_d_1',
                                               stride=2, stride_d=2))
        discriminator.add(SingleBatchNormLayer(name="batch_normalization_d_1"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_1"))

        discriminator.add(Convolutional3DLayer([5, 5, 5, 256], initializer="xavier", name='convo_layer_d_2',
                                               stride=2, stride_d=2))
        discriminator.add(SingleBatchNormLayer(name="batch_normalization_d_2"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_2"))

        discriminator.add(Convolutional3DLayer([5, 5, 5, 512], initializer="xavier", name='convo_layer_d_3',
                                               stride=2, stride_d=2))
        discriminator.add(SingleBatchNormLayer(name="batch_normalization_d_3"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_3"))

        discriminator.add(Convolutional3DLayer([5, 5, 5, 1024], initializer="xavier", name='convo_layer_d_4',
                                               stride=2, stride_d=2))
        discriminator.add(SingleBatchNormLayer(name="batch_normalization_d_4"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_4"))

        discriminator.add(Flatten())
        discriminator.add(FullyConnectedLayer(out_neurons=1, initializer="xavier", name='fully_connected_5'))
        discriminator.add(LinearLayer(name="linear_d_5"))


if __name__ == '__main__':
    from cnn_models.iterators.ucf import UCF101

    tf.set_random_seed(100)
    ucf_train = UCF101(ffmpeg_path="/usr/bin/ffmpeg", data_path="/home/filip141/Datasets/UCF-101",
                       resolution="320x240", resize="64x64", num_frames=48, min_frames=48)
    gan = GANNetwork(generator_input_size=100, discriminator_input_size=[64, 64, 48, 3],
                     log_path="/home/filip141/tensor_logs/GAN_UFC", batch_size=4, labels='convo-semi-supervised',
                     labels_size=101)
    gan.set_discriminator_optimizer("Adam", beta_1=0.5)
    gan.set_generator_optimizer("Adam", beta_1=0.5)
    gan.set_loss("js-non-saturation", label_smooth=True)
    gan.model_compile(generator_learning_rate=0.001, discriminator_learning_rate=0.0001)
    gan.train(ucf_train, train_step=4, epochs=300, sample_per_epoch=1000, restore_model=True,
              store_method=save_video)
