from simple_network.models import NetworkParallel, ResidualNode
from simple_network.layers import ConvolutionalLayer, FullyConnectedLayer, \
    Flatten, BatchNormalizationLayer, GlobalAveragePoolingLayer, LeakyReluLayer, SwishLayer


class ResNetIceberg(object):

    def __init__(self, input_size, output_size, metrics, log_path):
        self.input_summary = {"img_number": 5}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkParallel(input_size, metric=metrics,
                                         input_summary=self.input_summary, summary_path=log_path)

    def build_model(self):
        self.net_model.add(ConvolutionalLayer([3, 3, 32], initializer="xavier", name='convo_layer_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1"))

        for l_idx in range(0, 5):
            residual_node = ResidualNode(name="residual_node_{}".format(l_idx + 1), ntimes=4)
            residual_node.add(BatchNormalizationLayer(name="batch_normalization_residual_{}_1".format(l_idx + 1),
                                                      summaries=False))
            residual_node.add(SwishLayer(name="swish_residual_{}_1".format(l_idx + 1), summaries=False))
            residual_node.add(ConvolutionalLayer([3, 3, 32], initializer="xavier",
                                                 name='convo_layer_residual_{}_1'.format(l_idx + 1), summaries=False))
            residual_node.add(BatchNormalizationLayer(name="batch_normalization_residual_{}_2".format(l_idx + 1),
                                                      summaries=False))
            residual_node.add(SwishLayer(name="swish_layer_residual_{}_2".format(l_idx + 1), summaries=False))
            residual_node.add(ConvolutionalLayer([3, 3, 32], initializer="xavier",
                                                 name='convo_layer_residual_{}_2'.format(l_idx + 1), summaries=False))
            self.net_model.add(residual_node)
            self.net_model.add(ConvolutionalLayer([3, 3, 32], initializer="xavier",
                                                  name='convo_layer_{}_3'.format(l_idx + 1), stride=2))
            self.net_model.add(BatchNormalizationLayer(name="batch_normalization_{}_3".format(l_idx + 1)))

        self.net_model.add(ConvolutionalLayer([3, 3, 10], initializer="xavier", name='convo_layer_final'))

        self.net_model.add(GlobalAveragePoolingLayer(name="global_average_pool_1"))
        self.net_model.add(FullyConnectedLayer([10, self.output_size], initializer="xavier",
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
    json_data_train = "/home/phoenix/Datasets/Iceberg_data/train/processed/train.json"
    iceberg_db_train = ImageIterator(IcebergDataset(json_path=json_data_train, batch_out="color_composite_nn",
                                                    divide_point=0.2), rotate=360)
    iceberg_db_test = ImageIterator(IcebergDataset(json_path=json_data_train, is_test=True,
                                                   batch_out="color_composite_nn", divide_point=0.2), rotate=360)
    resnet = ResNetIceberg(input_size=[75, 75, 3], output_size=1,
                           metrics=["binary_accuracy", "cross_entropy_sigmoid"],
                           log_path="/home/phoenix/tensor_logs/ResNet_ICEBERG_NO_SIN")
    resnet.build_model()
    resnet.model_compile(0.0001)
    resnet.train(iceberg_db_train, iceberg_db_test, batch_size=32, batch_size_test=310)
