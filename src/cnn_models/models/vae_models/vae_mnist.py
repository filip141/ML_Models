import logging
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from simple_network.models import VAEScheme
from simple_network.models.additional_nodes import NetworkNode
from simple_network.layers import DeconvolutionLayer, FullyConnectedLayer, ConvolutionalLayer, ReshapeLayer, \
    LeakyReluLayer, BatchNormalizationLayer, DropoutLayer, Flatten, SigmoidLayer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VAENetwork(VAEScheme):

    def __init__(self, encoder_input_size, decoder_input_size, batch_size, log_path, labels='none'):
        super(VAENetwork, self).__init__(encoder_input_size, decoder_input_size, batch_size, log_path, labels=labels)
        self.batch_size = batch_size

    def build_encoder(self, encoder):
        # Convolutional 1
        encoder.add(ConvolutionalLayer([5, 5, 16], initializer="normal", name='convo_layer_e_1', stride=2))
        encoder.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_e_1"))

        # Convolutional 2
        encoder.add(ConvolutionalLayer([5, 5, 32], initializer="normal", name='convo_layer_d_2', stride=2))
        encoder.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_d_2"))

        # Dense two streams
        encoder.add(Flatten(name="flatten_e_3"))
        network_node = NetworkNode(name="dense_node_e_3")
        network_node.add(FullyConnectedLayer([1568, self.decoder_input_size[0]], initializer="normal",
                                             name='fully_connected_mean'))
        network_node.add(FullyConnectedLayer([1568, self.decoder_input_size[0]], initializer="normal",
                                             name='fully_connected_std'))
        encoder.add(network_node)

    def build_decoder(self, generator):
        generator.add(FullyConnectedLayer([np.prod(self.decoder_input_size), 1568], initializer="normal",
                                          name='fully_connected_d_1'))
        generator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_d_1"))
        generator.add(ReshapeLayer(output_shape=[7, 7, 32], name="reshape_d_1"))

        # Deconvolution 1
        generator.add(DeconvolutionLayer([5, 5, 16], output_shape=[14, 14, 16], initializer="normal",
                                         name='deconv_layer_d_2', stride=2, batch_size=self.batch_size))
        generator.add(LeakyReluLayer(alpha=0.2, name="leaky_relu_d_2"))

        # Deconvolution 2
        generator.add(DeconvolutionLayer([5, 5, 1], output_shape=self.decoder_output_size,
                                         initializer="normal",
                                         name='deconv_layer_d_3', stride=2, batch_size=self.batch_size))
        generator.add(SigmoidLayer(name="sigmoid_d_3"))

    def predict(self, input_data, label=None):
        if label is not None:
            one_hot = np.zeros(shape=[1, 10])
            one_hot[0, label] = 1.0
            input_data = np.concatenate([input_data, one_hot], axis=1)
        batch_matrix = np.zeros((1, self.decoder_input_size[0]), dtype=np.float32)
        batch_matrix[0] = input_data
        pred_dict = {self.decoder_network.input_layer_placeholder: batch_matrix,
                     self.decoder_network.is_training_placeholder: False}
        eval_result = self.decoder_network.sess.run(
            self.decoder_network.layer_outputs[-1],
            feed_dict=pred_dict)
        return 255 * eval_result[0, :, :, 0]

    def print_fancy_image(self):
        n = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.predict(z_sample)
                digit = x_decoded.reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

    def print_plane(self, dataset, batch_size=256):
        one_hot_dec = lambda lab: [np.argmax(x) for x in lab]
        imgs, labels = dataset.next_batch(batch_size)
        eval_result = self.encoder_network.sess.run(
            self.encoder_network.layer_outputs[-1],
            feed_dict={self.encoder_network.input_layer_placeholder: imgs,
                       self.encoder_network.is_training_placeholder: False})
        plt.figure(figsize=(6, 6))
        plt.scatter(eval_result[0][:, 0], eval_result[0][:, 1], c=one_hot_dec(labels))
        plt.colorbar()
        plt.show()

    def print_fancy_image_rand_multidim(self, n_dim):
        n = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        for i in range(n):
            for j in range(n):
                z_sample = np.random.uniform(0.0, 1.0, size=[1, n_dim])
                x_decoded = self.predict(z_sample)
                digit = x_decoded.reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

if __name__ == '__main__':
    from cnn_models.iterators.mnist import MNISTDataset
    mnist = MNISTDataset("/home/filip/Datasets", resolution="28x28", one_hot=True)
    vae = VAENetwork(decoder_input_size=32, encoder_input_size=[28, 28, 1],
                     log_path="/home/filip/tensor_logs/CVAE_MNIST", batch_size=1, labels="convo_style")
    vae.set_optimizer("Adam", beta_1=0.5)
    vae.model_compile(learning_rate=0.0001)
    vae.restore()
    # vae.train(mnist, train_step=128, epochs=300, sample_per_epoch=430, restore_model=True)
    vec_ref = np.random.uniform(0.0, 1.0, size=[1, 32])
    img = vae.predict(vec_ref, label=7)
    plt.imshow(img, cmap='gray')
    plt.show()
    # vae.print_fancy_image_rand_multidim(n_dim=32)
    # vae.print_plane(mnist, batch_size=10000)
#
