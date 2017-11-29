from simple_network.models import NetworkParallel, ResidualNode
from simple_network.layers import ConvolutionalLayer, ReluLayer, FullyConnectedLayer, \
    Flatten, BatchNormalizationLayer, GlobalAveragePoolingLayer, LeakyReluLayer, SpatialDropoutLayer


class WRNIceberg(object):

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
            residual_node.add(SpatialDropoutLayer(percent=0.4, name="spatial_dropout_{}_1".format(l_idx + 1)))
            residual_node.add(ConvolutionalLayer([3, 3, 16 * self.widening * 2**l_idx], initializer="xavier",
                                                 name='convo_layer_residual_{}_1'.format(l_idx + 1), summaries=False))

            # Second convolutional
            residual_node.add(BatchNormalizationLayer(name="batch_normalization_residual_{}_2".format(l_idx + 1),
                                                      summaries=False))
            residual_node.add(LeakyReluLayer(name="leaky_layer_residual_{}_2".format(l_idx + 1), summaries=False))
            residual_node.add(SpatialDropoutLayer(percent=0.4, name="spatial_dropout_{}_2".format(l_idx + 1)))
            residual_node.add(ConvolutionalLayer([3, 3, 16 * self.widening * 2**l_idx], initializer="xavier",
                                                 name='convo_layer_residual_{}_2'.format(l_idx + 1), summaries=False))
            self.net_model.add(residual_node)
            self.net_model.add(ReluLayer("relu_layer_final_{}".format(l_idx)))
        self.net_model.add(GlobalAveragePoolingLayer(name="global_average_pool_1"))
        self.net_model.add(FullyConnectedLayer(out_neurons=self.output_size, initializer="xavier",
                                               name='fully_connected_final'))
        self.net_model.set_loss("cross_entropy", activation="sigmoid")
        self.net_model.set_optimizer("Momentum")

    def model_compile(self, learning_rate):
        self.net_model.build_model(learning_rate)

    def train(self, train_iterator, test_iterator, batch_size, batch_size_test=None, restore_model=False, epochs=300):
        if batch_size_test is None:
            batch_size_test = batch_size
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size_test, sample_per_epoch=391, epochs=epochs)

if __name__ == '__main__':
    from cnn_models.iterators.iceberg import IcebergDataset
    from cnn_models.iterators.tools import ImageIterator
    json_data_train = "/home/filip/Datasets/Iceberg_data/train/processed/train.json"
    iceberg_db_train = ImageIterator(IcebergDataset(json_path=json_data_train, batch_out="color_composite",
                                                    divide_point=0.2), rotate=360)
    iceberg_db_test = ImageIterator(IcebergDataset(json_path=json_data_train, is_test=True,
                                                   batch_out="color_composite_nn", divide_point=0.2), rotate=360)
    wrnnet = WRNIceberg(input_size=[75, 75, 3], output_size=1,
                        metrics=["binary_accuracy", "cross_entropy_sigmoid"],
                        log_path="/home/filip/tensor_logs/WRN_ICEBERG", widening=8, n_blocks=3)
    wrnnet.build_model()
    wrnnet.model_compile(0.0001)
    wrnnet.train(iceberg_db_train, iceberg_db_test, batch_size=32, batch_size_test=310)
