import logging
import numpy as np
from simple_network.models import GANScheme
from simple_network.layers import DeconvolutionLayer, FullyConnectedLayer, ConvolutionalLayer, ReshapeLayer, \
    LeakyReluLayer, BatchNormalizationLayer, DropoutLayer, Flatten, TanhLayer, LinearLayer, MiniBatchDiscrimination, \
    SpatialDropoutLayer, GlobalAveragePoolingLayer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GANNetwork(GANScheme):

    def __init__(self, generator_input_size, discriminator_input_size, log_path, batch_size, labels="none",
                 labels_size=10, use_dropout=False, global_avg=False):
        super(GANNetwork, self).__init__(generator_input_size, discriminator_input_size, log_path, batch_size,
                                         labels=labels, labels_size=labels_size)
        self.use_dropout = use_dropout
        self.global_avg = global_avg
        self.batch_size = batch_size

    def build_generator(self, generator):
        generator.add(FullyConnectedLayer(out_neurons=12544, initializer="xavier",
                                          name='fully_connected_g_1'))
        generator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_g_1"))
        if self.use_dropout:
            generator.add(DropoutLayer(percent=0.5))
        generator.add(ReshapeLayer(output_shape=[7, 7, 256], name="reshape_g_1"))

        # Deconvolution 1
        generator.add(DeconvolutionLayer([4, 4, 64], output_shape=[14, 14, 64], initializer="xavier",
                                         name='deconv_layer_g_2', stride=2, batch_size=self.batch_size))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_2"))
        generator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_g_2"))
        if self.use_dropout:
            generator.add(SpatialDropoutLayer(percent=0.4, name="dropout_g_2"))

        # Deconvolution 2
        generator.add(DeconvolutionLayer([4, 4, 1], output_shape=self.generator_output_size, initializer="xavier",
                                         name='deconv_layer_g_3', stride=2, batch_size=self.batch_size))
        generator.add(TanhLayer(name="tanh_g_3"))

    def build_discriminator(self, discriminator):
        # Convolutional 1
        discriminator.add(ConvolutionalLayer([4, 4, 64], initializer="xavier", name='convo_layer_d_1', stride=2))
        discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_1"))
        discriminator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_d_1"))
        if self.use_dropout:
            discriminator.add(SpatialDropoutLayer(percent=0.4, name="dropout_d_1"))

        # Convolutional 2
        discriminator.add(ConvolutionalLayer([4, 4, 128], initializer="xavier", name='convo_layer_d_2', stride=2))
        discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_2"))
        discriminator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_d_2"))
        if self.use_dropout:
            discriminator.add(SpatialDropoutLayer(percent=0.4, name="dropout_d_2"))

        # Dense 1
        if self.global_avg:
            discriminator.add(GlobalAveragePoolingLayer(name="global_avg_d"))
        else:
            discriminator.add(Flatten(name="flatten_d"))
        discriminator.add(FullyConnectedLayer(out_neurons=256, initializer="xavier", name='fully_connected_d_3'))
        discriminator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_d_3"))
        if self.use_dropout:
            discriminator.add(DropoutLayer(percent=0.5))

        # discriminator.add(MiniBatchDiscrimination(batch_size=self.batch_size))

        # Dense 2
        discriminator.add(FullyConnectedLayer(out_neurons=1, initializer="xavier", name='fully_connected_d_4'))
        discriminator.add(LinearLayer(name="linear_d_4"))


if __name__ == '__main__':
    from cnn_models.iterators.mnist import MNISTDataset
    for drop_arg in [(True, "Drop"), (False, "NoDrop")]:
        for divergence_arg in [("kl-qp-loss", "KLQP"), ("feature-matching", "FMatch")]:
            for global_avg_arg in [(True, "GAvg"), (False, "NGAvg")]:
                if drop_arg[1] == "Drop" and divergence_arg[1] == "KLQP" and global_avg_arg[1] == "GAvg":
                    continue
                mnist = MNISTDataset("/home/filip/Datasets", resolution="28x28", one_hot=True)
                gan = GANNetwork(generator_input_size=100, discriminator_input_size=[28, 28, 1],
                                 log_path="/home/filip/tensor_logs/GAN_MNIST_EXP/{}_BN_LSmooth_NonCond_{}_{}".format(
                                     divergence_arg[1], drop_arg[1], global_avg_arg[1]),
                                 batch_size=128, use_dropout=drop_arg[0], global_avg=global_avg_arg[0])
                gan.set_discriminator_optimizer("Adam", beta_1=0.5)
                gan.set_generator_optimizer("Adam", beta_1=0.5)
                gan.set_loss(divergence_arg[0], label_smooth=True)
                gan.model_compile(generator_learning_rate=0.0002, discriminator_learning_rate=0.00005)
                gan.train(mnist, generator_steps=1, discriminator_steps=1, train_step=128, epochs=15,
                          sample_per_epoch=391)
