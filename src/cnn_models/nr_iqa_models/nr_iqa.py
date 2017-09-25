import os
from simple_network.tools.utils import LIVEDataset
from simple_network.models import NetworkParallel, NetworkNode
from simple_network.layers import ConvolutionalLayer, MaxPoolingLayer, ReluLayer, FullyConnectedLayer, \
     Flatten, DropoutLayer, BatchNormalizationLayer, LeakyReluLayer


class LiveModel(object):

    def __init__(self, dataset_path, log_path):
        self.input_summary = {"img_number": 64}

        self.live_train = LIVEDataset(data_path=dataset_path, new_resolution=None, patches="32x32", no_patches=32,
                                      patches_method='random')
        self.live_test = LIVEDataset(data_path=dataset_path, new_resolution=None, patches="32x32", no_patches=32,
                                     is_train=False, patches_method='random')
        # Define model
        self.net_model = NetworkParallel([32, 32, 3], metric=['mae'],
                                         input_summary=self.input_summary, summary_path=log_path)

    def build_model(self, learning_rate):
        self.net_model.add(ConvolutionalLayer([3, 3, 32], initializer="xavier", name='convo_layer_1_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1_1"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_1_1"))
        self.net_model.add(ConvolutionalLayer([3, 3, 32], initializer="xavier", name='convo_layer_1_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1_2"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_1_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_1_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 64], initializer="xavier", name='convo_layer_2_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_2_1"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_2_1"))
        self.net_model.add(ConvolutionalLayer([3, 3, 64], initializer="xavier", name='convo_layer_2_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_2_2"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_2_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_2_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_3_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3_1"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_3_1"))
        self.net_model.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_3_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3_2"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_3_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_3_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_4_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4_1"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_4_1"))
        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_4_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4_2"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_4_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_4_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_5_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_5_1"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_5_1"))
        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_5_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_5_2"))
        self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_5_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_5_1"))

        self.net_model.add(Flatten(name='flatten_1'))

        nm_node = NetworkNode(name="node_1_fc")
        nm_node.add(FullyConnectedLayer([512, 512], initializer="xavier", name='fully_connected_6_1_1'))
        nm_node.add(FullyConnectedLayer([512, 512], initializer="xavier", name='fully_connected_6_1_2'))
        self.net_model.add(nm_node)

        nm_node = NetworkNode(name="node_2_leaky")
        nm_node.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_6_1_1"))
        nm_node.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_6_1_2"))
        self.net_model.add(nm_node)

        nm_node = NetworkNode(name="node_3_drop")
        nm_node.add(DropoutLayer(percent=0.5))
        nm_node.add(DropoutLayer(percent=0.5))
        self.net_model.add(nm_node)

        nm_node = NetworkNode(name="node_4_fc")
        nm_node.add(FullyConnectedLayer([512, 1], initializer="xavier", name='fully_connected_7_1_1'))
        nm_node.add(FullyConnectedLayer([512, 1], initializer="xavier", name='fully_connected_7_1_2', activation='relu'))
        self.net_model.add(nm_node)

        # self.net_model.add(FullyConnectedLayer([512, 512], initializer="xavier", name='fully_connected_6_1_1'))
        # self.net_model.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_6_1_1"))
        # self.net_model.add(DropoutLayer(percent=0.5))
        # self.net_model.add(FullyConnectedLayer([512, 1], initializer="xavier", name='fully_connected_7_1_1'))

        self.net_model.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.net_model.set_loss("mae_weight", nimages=9, reshape_weights=None)
        self.build(learning_rate)


    def train(self, batch_size):
        self.net_model.train(train_iter=self.live_train, train_step=batch_size, test_iter=self.live_test,
                             test_step=batch_size, sample_per_epoch=262, epochs=3000)

    def build(self, learning_rate):
        self.net_model.build_model(learning_rate=learning_rate)

    def restore_model(self):
        self.net_model.restore()


if __name__ == '__main__':
    dataset_path = "/home/phoenix/Datasets/Live2005"
    live_model = LiveModel(dataset_path=dataset_path, log_path="/home/phoenix/tensor_logs")
    live_model.build_model(0.0001)
#    live_model.restore_model()    
    live_model.train(batch_size=288)


