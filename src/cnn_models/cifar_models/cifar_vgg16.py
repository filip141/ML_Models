import os
from simple_network.models import NetworkModel
from simple_network.tools.utils import CIFARDataset
from simple_network.layers import ConvolutionalLayer, MaxPoolingLayer, ReluLayer, FullyConnectedLayer, \
     Flatten, DropoutLayer, BatchNormalizationLayer, LeakyReluLayer


class CifarVGG16Model(object):

    def __init__(self, train_path, test_path, log_path):
        self.input_summary = {"img_number": 30}

        self.cifar_train = CIFARDataset(data_path=train_path)
        self.cifar_test = CIFARDataset(data_path=test_path)

        # Remove old tensor files
        files_in_dir = os.listdir(log_path)
        for s_file in files_in_dir:
            os.remove(os.path.join(log_path, s_file))

        # Define model
        self.net_model = NetworkModel([32, 32, 3], metric=["accuracy", "cross_entropy"],
                                      input_summary=self.input_summary, summary_path=log_path)

    def build_model(self):
        self.net_model.add(ConvolutionalLayer([3, 3, 64], initializer="xavier", name='convo_layer_1_1'))
        self.net_model.add(ReluLayer(name="relu_1_1"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1_1"))
        self.net_model.add(DropoutLayer(percent=0.3))

        self.net_model.add(ConvolutionalLayer([3, 3, 64], initializer="xavier", name='convo_layer_1_2'))
        self.net_model.add(ReluLayer(name="relu_1_2"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_1_2"))

        self.net_model.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_2_1'))
        self.net_model.add(ReluLayer(name="relu_2_1"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_2_1"))
        self.net_model.add(DropoutLayer(percent=0.4))

        self.net_model.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_2_2'))
        self.net_model.add(ReluLayer(name="relu_2_2"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_2_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_2_2"))

        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_3_1'))
        self.net_model.add(ReluLayer(name="relu_3_1"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3_1"))
        self.net_model.add(DropoutLayer(percent=0.4))

        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_3_2'))
        self.net_model.add(ReluLayer(name="relu_3_2"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3_2"))
        self.net_model.add(DropoutLayer(percent=0.4))

        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_3_3'))
        self.net_model.add(ReluLayer(name="relu_3_3"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3_3"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_3_3"))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_4_1'))
        self.net_model.add(ReluLayer(name="relu_4_1"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4_1"))
        self.net_model.add(DropoutLayer(percent=0.4))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_4_2'))
        self.net_model.add(ReluLayer(name="relu_4_2"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4_2"))
        self.net_model.add(DropoutLayer(percent=0.4))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_4_3'))
        self.net_model.add(ReluLayer(name="relu_4_3"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4_3"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_4_3"))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_5_1'))
        self.net_model.add(ReluLayer(name="relu_5_1"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_5_1"))
        self.net_model.add(DropoutLayer(percent=0.4))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_5_2'))
        self.net_model.add(ReluLayer(name="relu_5_2"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_5_2"))
        self.net_model.add(DropoutLayer(percent=0.4))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_5_3'))
        self.net_model.add(ReluLayer(name="relu_5_3"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_5_3"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_5_3"))

        self.net_model.add(DropoutLayer(percent=0.5))
        self.net_model.add(Flatten(name='flatten_6'))

        self.net_model.add(FullyConnectedLayer([512, 512], initializer="xavier", name='fully_connected_6_1'))
        self.net_model.add(ReluLayer(name="relu_6_1"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_6_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        self.net_model.add(FullyConnectedLayer([512, 10], initializer="xavier", name='fully_connected_7_1'))
        self.net_model.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.net_model.set_loss("cross_entropy")

    def train(self, learning_rate, batch_size):
        self.net_model.build_model(learning_rate)
        self.net_model.restore()
        self.net_model.train(train_iter=self.cifar_train, train_step=batch_size, test_iter=self.cifar_test,
                             test_step=batch_size, sample_per_epoch=391, epochs=300)

if __name__ == '__main__':
    train_path = "/home/filip/Datasets/cifar/train"
    test_path = "/home/filip/Datasets/cifar/test"
    cifar_model = CifarVGG16Model(train_path=train_path, test_path=test_path, log_path="/home/filip/tensor_logs")
    cifar_model.build_model()
    cifar_model.train(0.001, 128)

