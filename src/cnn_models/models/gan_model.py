import logging
import numpy as np
from simple_network.models import GANScheme
from simple_network.layers import DeconvolutionLayer, FullyConnectedLayer, ConvolutionalLayer, ReshapeLayer, \
    GlobalAveragePoolingLayer, LeakyReluLayer, BatchNormalizationLayer, ReluLayer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GANNetwork(GANScheme):

    def __init__(self, generator_input_size, discriminator_input_size, log_path):
        super(GANNetwork, self).__init__(generator_input_size, discriminator_input_size, log_path)

    def build_generator(self, generator):
        generator.add(FullyConnectedLayer([np.prod(self.generator_input_size), 4096], initializer="xavier",
                                          name='fully_connected_1'))
        generator.add(ReluLayer(name="relu_1"))
        generator.add(ReshapeLayer(output_shape=[2, 2, 1024]))
        generator.add(DeconvolutionLayer([5, 5, 512], output_shape=[4, 4, 512], initializer="xavier",
                                         name='deconv_layer_2', stride=2))
        # generator.add(BatchNormalizationLayer(name="batch_normalization_2"))
        generator.add(ReluLayer(name="relu_2"))
        generator.add(DeconvolutionLayer([5, 5, 256], output_shape=[8, 8, 256], initializer="xavier",
                                         name='deconv_layer_3', stride=2))
        # generator.add(BatchNormalizationLayer(name="batch_normalization_3"))
        generator.add(ReluLayer(name="relu_3"))
        generator.add(DeconvolutionLayer([5, 5, 128], output_shape=[16, 16, 128], initializer="xavier",
                                         name='deconv_layer_4', stride=2))
        # generator.add(BatchNormalizationLayer(name="batch_normalization_4"))
        generator.add(ReluLayer(name="relu_4"))
        generator.add(DeconvolutionLayer([5, 5, 64], output_shape=[32, 32, 64], initializer="xavier",
                                         name='deconv_layer_5', stride=2))
        # generator.add(BatchNormalizationLayer(name="batch_normalization_5"))
        generator.add(ReluLayer(name="relu_5"))
        generator.add(DeconvolutionLayer([3, 3, 1], output_shape=[32, 32, 1], initializer="xavier",
                                         name='deconv_layer_6', stride=1))

    def build_discriminator(self, discriminator):
        discriminator.add(ConvolutionalLayer([5, 5, 128], initializer="xavier", name='convo_layer_1', stride=2))
        # discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_1"))
        discriminator.add(LeakyReluLayer(alpha=0.1))
        discriminator.add(ConvolutionalLayer([5, 5, 256], initializer="xavier", name='convo_layer_2', stride=2))
        # discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_2"))
        discriminator.add(LeakyReluLayer(alpha=0.1))
        discriminator.add(ConvolutionalLayer([5, 5, 512], initializer="xavier", name='convo_layer_3', stride=2))
        # discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_3"))
        discriminator.add(LeakyReluLayer(alpha=0.1))
        discriminator.add(ConvolutionalLayer([5, 5, 1024], initializer="xavier", name='convo_layer_4', stride=2))
        # discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_4"))
        discriminator.add(LeakyReluLayer(alpha=0.1))
        discriminator.add(GlobalAveragePoolingLayer())
        discriminator.add(FullyConnectedLayer([1024, 1], initializer="xavier", name='fully_connected_5'))


if __name__ == '__main__':
    from cnn_models.iterators.mnist import MNISTDataset
    mnist = MNISTDataset("/home/filip/Datasets", resolution="32x32")
    # from cnn_models.iterators.cifar import CIFARDataset
    # train_path = "/home/filip/Datasets/cifar/train"
    # test_path = "/home/filip/Datasets/cifar/test"
    # cifar_train = CIFARDataset(data_path=train_path)
    # cifar_test = CIFARDataset(data_path=test_path)
    gan = GANNetwork(generator_input_size=[100, ], discriminator_input_size=[32, 32, 1],
                     log_path="/home/filip/tensor_logs")
    gan.build_model(0.0008, 0.0004)
    gan.train(mnist, 1, 32, epochs=300, sample_per_epoch=1875)
