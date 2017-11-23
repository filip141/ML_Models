import logging
import numpy as np
from simple_network.models import GANScheme
from simple_network.layers import DeconvolutionLayer, FullyConnectedLayer, ConvolutionalLayer, ReshapeLayer, \
    LeakyReluLayer, BatchNormalizationLayer, DropoutLayer, Flatten, SigmoidLayer, \
    LinearLayer, ReluLayer, GlobalAveragePoolingLayer

from cnn_models.models.vgg16 import VGG16Model, VGG16MAPPING


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
                                         name='deconv_layer_g_2', stride=2, batch_size=self.batch_size))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_2"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_2"))
        generator.add(ReluLayer(name="relu_2"))

        generator.add(DeconvolutionLayer([5, 5, 256], output_shape=[8, 8, 256], initializer="xavier",
                                         name='deconv_layer_g_3', stride=2, batch_size=self.batch_size))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_3"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_3"))
        generator.add(ReluLayer(name="relu_3"))

        generator.add(DeconvolutionLayer([5, 5, 128], output_shape=[16, 16, 128], initializer="xavier",
                                         name='deconv_layer_g_4', stride=2, batch_size=self.batch_size))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_4"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_4"))
        generator.add(ReluLayer(name="relu_4"))

        generator.add(DeconvolutionLayer([5, 5, 64], output_shape=[32, 32, 64], initializer="xavier",
                                         name='deconv_layer_g_5', stride=2, batch_size=self.batch_size))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_5"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_5"))
        generator.add(ReluLayer(name="relu_5"))

        generator.add(DeconvolutionLayer([5, 5, 32], output_shape=[64, 64, 32], initializer="xavier",
                                         name='deconv_layer_g_6', stride=2, batch_size=self.batch_size))
        generator.add(BatchNormalizationLayer(name="batch_normalization_g_6"))
        generator.add(DropoutLayer(percent=0.4, name="dropout_g_6"))
        generator.add(ReluLayer(name="relu_6"))

        generator.add(DeconvolutionLayer([3, 3, 3], output_shape=[64, 64, 3], initializer="xavier",
                                         name='deconv_layer_g_7', stride=1, batch_size=self.batch_size))
        generator.add(SigmoidLayer(name="sigmoid_g_7"))

    def build_discriminator(self, discriminator):
        vgg_model = VGG16Model(input_size=self.discriminator_input_size, output_size=1,
                               log_path="/home/filip/tensor_logs/VGG16",
                               metrics=["accuracy", "cross_entropy"])
        vgg_model.build_model(model_classifier=False, optimizer=False, loss=False)
        for layer in vgg_model.net_model.layers:
            discriminator.add(layer)
        discriminator.add(GlobalAveragePoolingLayer())
        discriminator.add(FullyConnectedLayer([512, 1], initializer="xavier", name='fully_connected_5'))
        discriminator.add(LinearLayer(name="linear_d_5"))

    def load_initial_weights(self, path, mapping):
        weights_dict = dict(np.load(path))
        for op_name, weights in weights_dict.items():
            op_t_name = op_name[:-2]
            op_indicator = op_name[-1]
            if op_t_name in mapping.keys():
                op_names_new = mapping[op_t_name]
                if op_indicator == "b":
                    self.discriminator.sess.run(
                        self.discriminator.get_layer_by_name(op_names_new).bias.assign(weights))
                else:
                    self.discriminator.sess.run(
                        self.discriminator.get_layer_by_name(op_names_new).weights.assign(weights))


if __name__ == '__main__':
    from cnn_models.iterators.celeba import CelebADataset
    train_path = "/home/filip/Datasets/CelebA"
    celeba = CelebADataset(data_path=train_path, resolution="64x64")
    gan = GANNetwork(generator_input_size=[100, ], discriminator_input_size=[64, 64, 3],
                     log_path="/home/filip/tensor_logs/GAN_CelebA_Pretrain", batch_size=32)
    gan.set_discriminator_optimizer("Adam", beta_1=0.5)
    gan.set_generator_optimizer("Adam", beta_1=0.5)
    gan.model_compile(generator_learning_rate=0.0002, discriminator_learning_rate=0.0002)

    del VGG16MAPPING["fc6"]
    del VGG16MAPPING["fc7"]
    del VGG16MAPPING["fc8"]
    gan.load_initial_weights("/home/filip/Weights/vgg16_weights.npz", VGG16MAPPING)
    gan.train(celeba, train_step=32, epochs=300, sample_per_epoch=3166, loss_supervised=False)
