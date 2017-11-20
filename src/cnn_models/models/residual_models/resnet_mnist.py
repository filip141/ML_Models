from simple_network.models import NetworkParallel, ResidualNode
from simple_network.layers import ConvolutionalLayer, ReluLayer, FullyConnectedLayer, \
    Flatten, BatchNormalizationLayer


class ResNetMNIST(object):

    def __init__(self, input_size, output_size, metrics, log_path):
        self.input_summary = {"img_number": 30}
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
            residual_node.add(ReluLayer(name="relu_layer_residual_{}_1".format(l_idx + 1), summaries=False))
            residual_node.add(ConvolutionalLayer([3, 3, 32], initializer="xavier",
                                                 name='convo_layer_residual_{}_1'.format(l_idx + 1), summaries=False))
            residual_node.add(BatchNormalizationLayer(name="batch_normalization_residual_{}_2".format(l_idx + 1),
                                                      summaries=False))
            residual_node.add(ReluLayer(name="relu_layer_residual_{}_2".format(l_idx + 1), summaries=False))
            residual_node.add(ConvolutionalLayer([3, 3, 32], initializer="xavier",
                                                 name='convo_layer_residual_{}_2'.format(l_idx + 1), summaries=False))
            self.net_model.add(residual_node)
            self.net_model.add(ConvolutionalLayer([3, 3, 32], initializer="xavier",
                                                  name='convo_layer_{}_3'.format(l_idx + 1), stride=2))
            self.net_model.add(BatchNormalizationLayer(name="batch_normalization_{}_3".format(l_idx + 1)))

        self.net_model.add(ConvolutionalLayer([3, 3, 10], initializer="xavier", name='convo_layer_final'))
        self.net_model.add(Flatten(name="flatten_1"))
        self.net_model.add(FullyConnectedLayer([10, 10], initializer="xavier", name='fully_connected_final'))
        self.net_model.set_loss("cross_entropy")
        self.net_model.set_optimizer("SGD")

    def model_compile(self, learning_rate):
        self.net_model.build_model(learning_rate)

    def train(self, train_iterator, test_iterator, batch_size, restore_model=False, epochs=300):
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size, sample_per_epoch=391, epochs=epochs)

if __name__ == '__main__':
    from cnn_models.iterators.mnist import MNISTDataset
    mnist_train = MNISTDataset("/home/filip/Datasets", resolution="28x28", train_set=True)
    mnist_test = MNISTDataset("/home/filip/Datasets", resolution="28x28", train_set=False)
    resnet = ResNetMNIST(input_size=[28, 28, 1], output_size=10,
                         metrics=["accuracy", "cross_entropy"],
                         log_path="/home/filip/tensor_logs/ResNet_MNIST")
    resnet.build_model()
    resnet.model_compile(0.001)
    resnet.train(mnist_train, mnist_test, batch_size=128)