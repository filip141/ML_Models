import logging
import tensorflow as tf
from simple_network.models import WasserstainGANScheme
from simple_network.layers import DeconvolutionLayer, FullyConnectedLayer, ConvolutionalLayer, ReshapeLayer, \
    LeakyReluLayer, SingleBatchNormLayer, DropoutLayer, Flatten, SigmoidLayer, \
    LinearLayer, ReluLayer, GlobalAveragePoolingLayer, TanhLayer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WGANNetwork(WasserstainGANScheme):

    def __init__(self, generator_input_size, discriminator_input_size, log_path, batch_size=None):
        super(WGANNetwork, self).__init__(generator_input_size, discriminator_input_size, log_path, batch_size)
        self.batch_size = batch_size

    def build_generator(self, generator):
        generator.add(FullyConnectedLayer(out_neurons=4096, initializer="xavier",
                                          name='fully_connected_g_1', use_bias=True))
        generator.add(LeakyReluLayer(name="relu_1"))
        generator.add(ReshapeLayer(output_shape=[2, 2, 1024]))
        generator.add(DeconvolutionLayer([5, 5, 512], output_shape=[4, 4, 512], initializer="xavier",
                                         name='deconv_layer_g_2', stride=2, batch_size=self.batch_size,
                                         use_bias=True))
        generator.add(SingleBatchNormLayer(scale=False, name="batch_normalization_g_2"))
        generator.add(LeakyReluLayer(name="relu_2"))

        generator.add(DeconvolutionLayer([5, 5, 256], output_shape=[8, 8, 256], initializer="xavier",
                                         name='deconv_layer_g_3', stride=2, batch_size=self.batch_size,
                                         use_bias=True))
        generator.add(SingleBatchNormLayer(scale=False, name="batch_normalization_g_3"))
        generator.add(LeakyReluLayer(name="relu_3"))

        generator.add(DeconvolutionLayer([5, 5, 128], output_shape=[16, 16, 128], initializer="xavier",
                                         name='deconv_layer_g_4', stride=2, batch_size=self.batch_size,
                                         use_bias=True))
        generator.add(SingleBatchNormLayer(scale=False, name="batch_normalization_g_4"))
        generator.add(LeakyReluLayer(name="relu_4"))

        generator.add(DeconvolutionLayer([5, 5, 64], output_shape=[32, 32, 64], initializer="xavier",
                                         name='deconv_layer_g_5', stride=2, batch_size=self.batch_size,
                                         use_bias=True))
        generator.add(SingleBatchNormLayer(scale=False, name="batch_normalization_g_5"))
        generator.add(LeakyReluLayer(name="relu_5"))

        generator.add(DeconvolutionLayer([5, 5, 32], output_shape=[64, 64, 32], initializer="xavier",
                                         name='deconv_layer_g_6', stride=2, batch_size=self.batch_size,
                                         use_bias=True))
        generator.add(SingleBatchNormLayer(scale=False, name="batch_normalization_g_6"))
        generator.add(LeakyReluLayer(name="relu_6"))

        generator.add(DeconvolutionLayer([3, 3, 3], output_shape=[64, 64, 3], initializer="xavier",
                                         name='deconv_layer_g_7', stride=1, batch_size=self.batch_size,
                                         use_bias=True))
        generator.add(TanhLayer(name="tanh_g_7"))

    def build_discriminator(self, discriminator):
        discriminator.add(ConvolutionalLayer([5, 5, 128], initializer="xavier", name='convo_layer_d_1', stride=2,
                                             use_bias=True))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_1"))

        discriminator.add(ConvolutionalLayer([5, 5, 256], initializer="xavier", name='convo_layer_d_2', stride=2,
                                             use_bias=True))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_2"))

        discriminator.add(ConvolutionalLayer([5, 5, 512], initializer="xavier", name='convo_layer_d_3', stride=2,
                                             use_bias=True))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_3"))

        discriminator.add(ConvolutionalLayer([5, 5, 1024], initializer="xavier", name='convo_layer_d_4', stride=2,
                                             use_bias=True))
        discriminator.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_d_4"))

        discriminator.add(GlobalAveragePoolingLayer())
        discriminator.add(FullyConnectedLayer(out_neurons=1, initializer="xavier", name='fully_connected_5',
                                              use_bias=True))
        discriminator.add(LinearLayer(name="linear_d_5"))


if __name__ == '__main__':
    from cnn_models.iterators.celeba import CelebADataset

    tf.set_random_seed(100)
    train_path = "/home/filip141/Datasets/CelebA"
    celeba = CelebADataset(data_path=train_path, resolution="64x64")
    gan = WGANNetwork(generator_input_size=100, discriminator_input_size=[64, 64, 3],
                      log_path="/home/filip141/tensor_logs/WGAN_CelebA", batch_size=64)
    gan.set_discriminator_optimizer("Adam")
    gan.set_generator_optimizer("Adam")
    gan.set_loss("gradient-penalty")
    gan.model_compile(generator_learning_rate=0.0001, discriminator_learning_rate=0.0001)
    gan.train(celeba, train_step=64, epochs=300, sample_per_epoch=1000, restore_model=False, generator_steps=6)
