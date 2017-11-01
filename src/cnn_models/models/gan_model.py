import logging
import numpy as np
import tensorflow as tf
from simple_network.models import GANScheme
from simple_network.layers import DeconvolutionLayer, FullyConnectedLayer, ConvolutionalLayer, ReshapeLayer, \
    GlobalAveragePoolingLayer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GANNetwork(GANScheme):

    def __init__(self, generator_input_size, discriminator_input_size, output_size, log_path):
        super(GANNetwork, self).__init__(generator_input_size, discriminator_input_size, output_size, log_path)

    def build_generator(self, generator):
        generator.add(FullyConnectedLayer([np.prod(self.generator_input_size), 4096], initializer="xavier",
                                          name='fully_connected_1'))
        generator.add(ReshapeLayer(output_shape=[2, 2, 1024]))
        generator.add(DeconvolutionLayer([5, 5, 512], output_shape=[4, 4, 512], initializer="xavier",
                                         name='deconv_layer_2', stride=2, activation="relu"))
        generator.add(DeconvolutionLayer([5, 5, 256], output_shape=[8, 8, 256], initializer="xavier",
                                         name='deconv_layer_3', stride=2, activation="relu"))
        generator.add(DeconvolutionLayer([5, 5, 128], output_shape=[16, 16, 128], initializer="xavier",
                                         name='deconv_layer_4', stride=2, activation="relu"))
        generator.add(DeconvolutionLayer([5, 5, 64], output_shape=[32, 32, 64], initializer="xavier",
                                         name='deconv_layer_5', stride=2, activation="relu"))
        generator.add(DeconvolutionLayer([3, 3, 3], output_shape=[32, 32, 3], initializer="xavier",
                                         name='deconv_layer_6', stride=1, activation="tanh"))

    def build_discriminator(self, discriminator):
        discriminator.add(ConvolutionalLayer([5, 5, 128], initializer="xavier", name='convo_layer_1', stride=2,
                                             activation="relu"))
        discriminator.add(ConvolutionalLayer([5, 5, 256], initializer="xavier", name='convo_layer_2', stride=2,
                                             activation="relu"))
        discriminator.add(ConvolutionalLayer([5, 5, 512], initializer="xavier", name='convo_layer_3', stride=2,
                                             activation="relu"))
        discriminator.add(ConvolutionalLayer([5, 5, 1024], initializer="xavier", name='convo_layer_4', stride=2,
                                             activation="relu"))
        discriminator.add(GlobalAveragePoolingLayer())
        discriminator.add(FullyConnectedLayer([1024, 1], initializer="xavier", name='fully_connected_5'))


if __name__ == '__main__':
    from cnn_models.iterators.cifar import CIFARDataset
    train_path = "/home/filip/Datasets/cifar/train"
    test_path = "/home/filip/Datasets/cifar/test"
    cifar_train = CIFARDataset(data_path=train_path)
    cifar_test = CIFARDataset(data_path=test_path)
    gan = GANNetwork(generator_input_size=[100, ], discriminator_input_size=[32, 32, 3], output_size=1000,
                     log_path="/home/filip/tensor_logs")
    gan.build_model(0.0001, 0.0001)
    gan.train(cifar_train, 100, 10, epochs=300, sample_per_epoch=3125)
