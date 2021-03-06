import logging
import numpy as np
from simple_network.models import GANScheme
from simple_network.layers import DeconvolutionLayer, FullyConnectedLayer, ConvolutionalLayer, ReshapeLayer, \
    LeakyReluLayer, BatchNormalizationLayer, DropoutLayer, Flatten, SigmoidLayer, LinearLayer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GANNetwork(GANScheme):

    def __init__(self, generator_input_size, discriminator_input_size, log_path, batch_size=None):
        super(GANNetwork, self).__init__(generator_input_size, discriminator_input_size, log_path)
        self.batch_size = batch_size

    def build_generator(self, generator):
        generator.add(FullyConnectedLayer([np.prod(self.generator_input_size), 12544], initializer="xavier",
                                          name='fully_connected_g_1'))
        generator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_g_1"))
        generator.add(ReshapeLayer(output_shape=[7, 7, 256], name="reshape_g_1"))

        # Deconvolution 1
        generator.add(DeconvolutionLayer([4, 4, 64], output_shape=[14, 14, 64], initializer="xavier",
                                         name='deconv_layer_g_2', stride=2, batch_size=self.batch_size))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_2"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_2"))
        generator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_g_2"))

        # Deconvolution 2
        generator.add(DeconvolutionLayer([4, 4, 1], output_shape=[28, 28, 1], initializer="xavier",
                                         name='deconv_layer_g_3', stride=2, batch_size=self.batch_size))
        generator.add(SigmoidLayer(name="sigmoid_g_3"))

    def build_discriminator(self, discriminator):
        # Convolutional 1
        discriminator.add(ConvolutionalLayer([4, 4, 64], initializer="xavier", name='convo_layer_d_1', stride=2))
        discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_1"))
        discriminator.add(DropoutLayer(percent=0.4, name="dropout_d_1"))
        discriminator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_d_1"))

        # Convolutional 2
        discriminator.add(ConvolutionalLayer([4, 4, 128], initializer="xavier", name='convo_layer_d_2', stride=2))
        discriminator.add(BatchNormalizationLayer(name="batch_normalization_d_2"))
        discriminator.add(DropoutLayer(percent=0.4, name="dropout_d_2"))
        discriminator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_d_2"))

        # Dense 1
        discriminator.add(Flatten(name="flatten_d"))
        discriminator.add(FullyConnectedLayer([6272, 256], initializer="xavier", name='fully_connected_d_3'))
        discriminator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_d_3"))

        # Dense 2
        discriminator.add(FullyConnectedLayer([256, 1], initializer="xavier", name='fully_connected_d_4'))
        discriminator.add(LinearLayer(name="linear_d_4"))

if __name__ == '__main__':
    from cnn_models.iterators.iceberg import IcebergDataset
    json_data_train = "/home/filip/Datasets/Iceberg_data/train/processed/train.json"
    iceberg_db_train = IcebergDataset(json_path=json_data_train, batch_out="mean_dim", resolution="28x28")
    gan = GANNetwork(generator_input_size=[100, ], discriminator_input_size=[28, 28, 1],
                     log_path="/home/filip/tensor_logs", batch_size=128)
    gan.build_model(generator_learning_rate=0.0005, discriminator_learning_rate=0.00005)
    gan.train(iceberg_db_train, discriminator_steps=10, generator_steps=1, train_step=128,
              epochs=300, sample_per_epoch=390)
