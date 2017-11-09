import logging
import numpy as np
from simple_network.models import GANScheme
from simple_network.layers import DeconvolutionLayer, FullyConnectedLayer, ConvolutionalLayer, ReshapeLayer, \
    LeakyReluLayer, BatchNormalizationLayer, DropoutLayer, Flatten, SigmoidLayer, \
    LinearLayer, ReluLayer, GlobalAveragePoolingLayer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GANNetwork(GANScheme):

    def __init__(self, generator_input_size, discriminator_input_size, log_path, batch_size=None):
        super(GANNetwork, self).__init__(generator_input_size, discriminator_input_size, log_path)
        self.batch_size = batch_size

    def build_generator(self, generator):
        generator.add(FullyConnectedLayer([np.prod(self.generator_input_size), 4096], initializer="xavier",
                                          name='fully_connected_g_1'))
        generator.add(ReluLayer(name="relu_1"))
        generator.add(ReshapeLayer(output_shape=[2, 2, 1024]))
        generator.add(DeconvolutionLayer([5, 5, 512], output_shape=[4, 4, 512], initializer="xavier",
                                         name='deconv_layer_g_2', stride=2))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_2"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_2"))
        generator.add(ReluLayer(name="relu_2"))

        generator.add(DeconvolutionLayer([5, 5, 256], output_shape=[8, 8, 256], initializer="xavier",
                                         name='deconv_layer_g_3', stride=2))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_3"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_3"))
        generator.add(ReluLayer(name="relu_3"))

        generator.add(DeconvolutionLayer([5, 5, 128], output_shape=[16, 16, 128], initializer="xavier",
                                         name='deconv_layer_g_4', stride=2))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_4"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_4"))
        generator.add(ReluLayer(name="relu_4"))

        generator.add(DeconvolutionLayer([5, 5, 64], output_shape=[32, 32, 64], initializer="xavier",
                                         name='deconv_layer_g_5', stride=2))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_5"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_5"))
        generator.add(ReluLayer(name="relu_5"))
        generator.add(DeconvolutionLayer([3, 3, 3], output_shape=[32, 32, 3], initializer="xavier",
                                         name='deconv_layer_g_6', stride=1))
        generator.add(SigmoidLayer(name="sigmoid_g_3"))

    def build_discriminator(self, discriminator):
        discriminator.add(ConvolutionalLayer([5, 5, 128], initializer="xavier", name='convo_layer_d_1', stride=2))
        discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_1"))
        discriminator.add(DropoutLayer(percent=0.4, name="dropout_d_1"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_1"))

        discriminator.add(ConvolutionalLayer([5, 5, 256], initializer="xavier", name='convo_layer_d_2', stride=2))
        discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_2"))
        discriminator.add(DropoutLayer(percent=0.4, name="dropout_d_2"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_2"))

        discriminator.add(ConvolutionalLayer([5, 5, 512], initializer="xavier", name='convo_layer_d_3', stride=2))
        discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_3"))
        discriminator.add(DropoutLayer(percent=0.4, name="dropout_d_3"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_3"))

        discriminator.add(ConvolutionalLayer([5, 5, 1024], initializer="xavier", name='convo_layer_d_4', stride=2))
        discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_4"))
        discriminator.add(DropoutLayer(percent=0.4, name="dropout_d_4"))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_4"))

        discriminator.add(GlobalAveragePoolingLayer())
        discriminator.add(FullyConnectedLayer([1024, 1], initializer="xavier", name='fully_connected_5'))
        discriminator.add(LinearLayer(name="linear_d_5"))


if __name__ == '__main__':
    from cnn_models.iterators.imagenet import DogsDataset
    train_path = '/home/phoenix/Datasets/StanfordDogs/Images'
    labels_path = '/home/phoenix/Datasets/StanfordDogs/Annotation'
    class_names_path = '/home/phoenix/Datasets/StanfordDogs/class_names.txt'
    im_net_train = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
                               train_set=True, resize_img="32x32", normalize_to_one=True)
    gan = GANNetwork(generator_input_size=[100, ], discriminator_input_size=[32, 32, 3],
                     log_path="/home/phoenix/tensor_logs/GAN_DOGS", batch_size=128)
    gan.set_discriminator_optimizer("Adam", beta_1=0.5)
    gan.set_generator_optimizer("Adam", beta_1=0.5)
    gan.model_compile(generator_learning_rate=0.0008, discriminator_learning_rate=0.0001)
    gan.restore()
    gan.train(im_net_train, discriminator_steps=10, train_step=128, epochs=300, sample_per_epoch=390)
