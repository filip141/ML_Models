import logging
import numpy as np
import tensorflow as tf
from simple_network.models import MMGANScheme
from simple_network.layers import DeconvolutionLayer, FullyConnectedLayer, ConvolutionalLayer, ReshapeLayer, \
    LeakyReluLayer, SingleBatchNormLayer, DropoutLayer, Flatten, SigmoidLayer, \
    LinearLayer, ReluLayer, GlobalAveragePoolingLayer, TanhLayer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GANNetwork(MMGANScheme):

    def __init__(self, generator_input_size, discriminator_input_size, log_path, batch_size=None, labels=None,
                 labels_size=10):
        super(GANNetwork, self).__init__(generator_input_size, discriminator_input_size, log_path, batch_size,
                                         labels=labels, labels_size=labels_size)
        self.batch_size = batch_size

    def build_generator(self, generator):
        generator.add(FullyConnectedLayer(out_neurons=4096, initializer="xavier",
                                          name='fully_connected_g_1'))
        generator.add(LeakyReluLayer(name="relu_1"))
        generator.add(ReshapeLayer(output_shape=[2, 2, 1024]))
        generator.add(DeconvolutionLayer([5, 5, 512], output_shape=[4, 4, 512], initializer="xavier",
                                         name='deconv_layer_g_2', stride=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_2"))
        generator.add(LeakyReluLayer(name="relu_2"))

        generator.add(DeconvolutionLayer([5, 5, 256], output_shape=[8, 8, 256], initializer="xavier",
                                         name='deconv_layer_g_3', stride=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_3"))
        generator.add(LeakyReluLayer(name="relu_3"))

        generator.add(DeconvolutionLayer([5, 5, 128], output_shape=[16, 16, 128], initializer="xavier",
                                         name='deconv_layer_g_4', stride=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_4"))
        generator.add(LeakyReluLayer(name="relu_4"))

        generator.add(DeconvolutionLayer([5, 5, 64], output_shape=[32, 32, 64], initializer="xavier",
                                         name='deconv_layer_g_5', stride=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_5"))
        generator.add(LeakyReluLayer(name="relu_5"))

        generator.add(DeconvolutionLayer([5, 5, 32], output_shape=[64, 64, 32], initializer="xavier",
                                         name='deconv_layer_g_6', stride=2, batch_size=self.batch_size))
        generator.add(SingleBatchNormLayer(name="batch_normalization_g_6"))
        generator.add(LeakyReluLayer(name="relu_6"))

        generator.add(DeconvolutionLayer([3, 3, 3], output_shape=[64, 64, 3], initializer="xavier",
                                         name='deconv_layer_g_7', stride=1, batch_size=self.batch_size))
        generator.add(TanhLayer(name="tanh_g_7"))

    def build_discriminator(self, discriminator):
        discriminator.add(ConvolutionalLayer([5, 5, 128], initializer="xavier", name='convo_layer_d_1', stride=2))
        discriminator.add(SingleBatchNormLayer(name="batch_normalization_d_1"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_1"))

        discriminator.add(ConvolutionalLayer([5, 5, 256], initializer="xavier", name='convo_layer_d_2', stride=2))
        discriminator.add(SingleBatchNormLayer(name="batch_normalization_d_2"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_2"))

        discriminator.add(ConvolutionalLayer([5, 5, 512], initializer="xavier", name='convo_layer_d_3', stride=2))
        discriminator.add(SingleBatchNormLayer(name="batch_normalization_d_3"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_3"))

        discriminator.add(ConvolutionalLayer([5, 5, 1024], initializer="xavier", name='convo_layer_d_4', stride=2))
        discriminator.add(SingleBatchNormLayer(name="batch_normalization_d_4"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_4"))

        discriminator.add(GlobalAveragePoolingLayer())
        discriminator.add(FullyConnectedLayer(out_neurons=1, initializer="xavier", name='fully_connected_5'))
        discriminator.add(LinearLayer(name="linear_d_5"))


if __name__ == '__main__':
    from cnn_models.iterators.imagenet import DogsDataset
    train_path = '/home/filip/Datasets/StanfordDogs/Images'
    labels_path = '/home/filip/Datasets/StanfordDogs/Annotation'
    class_names_path = '/home/filip/Datasets/StanfordDogs/class_names.txt'
    im_net_train = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
                               train_set=True, resize_img="64x64")
    gan = GANNetwork(generator_input_size=100, discriminator_input_size=[64, 64, 3],
                     log_path="/home/filip/tensor_logs/GAN_DOGS", batch_size=32, labels='convo-semi-supervised',
                     labels_size=120)
    gan.set_discriminator_optimizer("Adam", beta_1=0.5)
    gan.set_generator_optimizer("Adam", beta_1=0.5)
    gan.set_loss("js-non-saturation", label_smooth=True)
    gan.model_compile(generator_learning_rate=0.0004, discriminator_learning_rate=0.0002)
    gan.train(im_net_train, train_step=32, epochs=300, sample_per_epoch=1000, restore_model=False)