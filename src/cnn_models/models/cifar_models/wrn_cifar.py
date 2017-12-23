from simple_network.models import NetworkParallel, ResidualNode
from simple_network.layers import ConvolutionalLayer, ReluLayer, FullyConnectedLayer, \
    Flatten, BatchNormalizationLayer, GlobalAveragePoolingLayer, LeakyReluLayer


class WRNCifar(object):

    def __init__(self, input_size, output_size, metrics, log_path, n_blocks=2, widening=2):
        self.input_summary = {"img_number": 5}
        self.input_size = input_size
        self.output_size = output_size
        self.widening = widening
        self.n_blocks = n_blocks

        # Define model
        self.net_model = NetworkParallel(input_size, metric=metrics,
                                         input_summary=self.input_summary, summary_path=log_path)

    def build_model(self):
        self.net_model.add(ConvolutionalLayer([3, 3, 16], initializer="xavier", name='convo_layer_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1"))
        self.net_model.add(LeakyReluLayer(name="leaky_relu_1"))

        # First residual block
        for l_idx in range(0, 3):
            # Add bottleneck node
            bottleneck_node = ResidualNode(name="bottleneck_node_{}".format(l_idx + 1), ntimes=1)
            bottleneck_node.add(ConvolutionalLayer([1, 1, 16 * self.widening * 2**l_idx], initializer="xavier",
                                                   name='convo_layer_bottleneck_{}_1'.format(l_idx + 1),
                                                   summaries=False, stride=2))
            bottleneck_node.add(BatchNormalizationLayer(name="batch_normalization_bottleneck_{}_2".format(l_idx + 1),
                                                        summaries=False))

            # Pool residual
            bottleneck_node.add_residual(ConvolutionalLayer([1, 1, 16 * self.widening * 2**l_idx], initializer="xavier",
                                                            stride=2,
                                                            name='convo_layer_bottle_short_{}_1'.format(l_idx + 1),
                                                            summaries=False))
            bottleneck_node.add_residual(BatchNormalizationLayer(name="batch_normalization_bottle_short_{}_2"
                                                                 .format(l_idx + 1),
                                                                 summaries=False))
            self.net_model.add(bottleneck_node)
            # Add residual node
            residual_node = ResidualNode(name="residual_node_{}".format(l_idx + 1), ntimes=self.n_blocks)

            # First convolutional
            residual_node.add(BatchNormalizationLayer(name="batch_normalization_residual_{}_1".format(l_idx + 1),
                                                      summaries=False))
            residual_node.add(LeakyReluLayer(name="leaky_residual_{}_1".format(l_idx + 1), summaries=False))
            residual_node.add(ConvolutionalLayer([3, 3, 16 * self.widening * 2**l_idx], initializer="xavier",
                                                 name='convo_layer_residual_{}_1'.format(l_idx + 1), summaries=False))

            # Second convolutional
            residual_node.add(BatchNormalizationLayer(name="batch_normalization_residual_{}_2".format(l_idx + 1),
                                                      summaries=False))
            residual_node.add(LeakyReluLayer(name="leaky_layer_residual_{}_2".format(l_idx + 1), summaries=False))
            residual_node.add(ConvolutionalLayer([3, 3, 16 * self.widening * 2**l_idx], initializer="xavier",
                                                 name='convo_layer_residual_{}_2'.format(l_idx + 1), summaries=False))
            residual_node.add_act_layer(ReluLayer("relu_layer_final_{}".format(l_idx)))
            self.net_model.add(residual_node)
        self.net_model.add(ReluLayer("relu_layer_model_end"))
        self.net_model.add(GlobalAveragePoolingLayer(name="global_average_pool_1"))
        self.net_model.add(FullyConnectedLayer(out_neurons=self.output_size, initializer="xavier",
                                               name='fully_connected_final'))
        self.net_model.set_loss("cross_entropy", activation="sigmoid")
        self.net_model.set_optimizer("Adam")

    def model_compile(self, learning_rate, decay=None, decay_steps=100000):
        self.net_model.build_model(learning_rate, decay, decay_steps)

    def set_optimizer(self, opt_name, **kwargs):
        self.net_model.set_optimizer(opt_name, **kwargs)

    def set_loss(self, loss_name, **kwargs):
        self.net_model.set_loss(loss_name, **kwargs)

    def train(self, train_iterator, test_iterator, batch_size, batch_size_test=None, restore_model=False, epochs=300,
              summary_step=10, sample_per_epoch=10000):
        if batch_size_test is None:
            batch_size_test = batch_size
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size_test, sample_per_epoch=sample_per_epoch, epochs=epochs,
                             summary_step=summary_step)


if __name__ == '__main__':
    from cnn_models.iterators.cifar import CIFARDataset
    cifar_train_path = "/home/filip141/Datasets/cifar/train"
    cifar_test_path = "/home/filip141/Datasets/cifar/test"
    cifar_train = CIFARDataset(data_path=cifar_train_path, resolution="32x32", force_overfit=False)
    cifar_test = CIFARDataset(data_path=cifar_test_path, resolution="32x32", force_overfit=False)
    wrnnet = WRNCifar(input_size=[32, 32, 3], output_size=10,
                      metrics=["accuracy", "cross_entropy"],
                      log_path="/home/filip141/tensor_logs/WRN_CIFAR", widening=4, n_blocks=2)
    wrnnet.build_model()
    wrnnet.set_optimizer("Adam")
    wrnnet.set_loss("cross_entropy")
    wrnnet.model_compile(0.003, decay=0.96, decay_steps=1562)
    wrnnet.train(cifar_train, cifar_test, batch_size=32, batch_size_test=100, epochs=15, sample_per_epoch=1562,
                 summary_step=80, restore_model=False)
