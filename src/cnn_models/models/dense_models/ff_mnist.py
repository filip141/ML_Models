from simple_network.models import NetworkParallel
from simple_network.layers import BatchNormalizationLayer, FullyConnectedLayer, \
     Flatten, LinearLayer, SigmoidLayer, DropoutLayer, ReluLayer


class DenseMNIST(object):

    def __init__(self, input_size, output_size, log_path, metrics=None):
        self.input_summary = {"img_number": 30}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkParallel(input_size, metric=metrics, input_summary=self.input_summary,
                                         summary_path=log_path)

    def build_model(self):
        # Layer 1
        self.net_model.add(Flatten(name="flatten_1"))
        self.net_model.add(FullyConnectedLayer([784, 200],
                                               initializer="xavier", name='fully_connected_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1"))
        self.net_model.add(ReluLayer(name="relu_1"))
        self.net_model.add(DropoutLayer(percent=0.5, name="dropout_1"))

        self.net_model.add(FullyConnectedLayer([200, 100],
                                               initializer="xavier", name='fully_connected_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_2"))
        self.net_model.add(ReluLayer(name="relu_2"))
        self.net_model.add(DropoutLayer(percent=0.5, name="dropout_2"))

        self.net_model.add(FullyConnectedLayer([100, 60],
                                               initializer="xavier", name='fully_connected_3'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3"))
        self.net_model.add(ReluLayer(name="relu_3"))
        self.net_model.add(DropoutLayer(percent=0.5, name="dropout_3"))

        self.net_model.add(FullyConnectedLayer([60, 30],
                                               initializer="xavier", name='fully_connected_4'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4"))
        self.net_model.add(ReluLayer(name="relu_4"))
        self.net_model.add(DropoutLayer(percent=0.5, name="dropout_4"))

        self.net_model.add(FullyConnectedLayer([30, self.output_size],
                                               initializer="xavier", name='fully_connected_5'))
        self.net_model.add(LinearLayer(name="linear_5"))

    def model_compile(self, learning_rate):
        self.net_model.build_model(learning_rate)

    def set_optimizer(self, opt_name, **kwargs):
        self.net_model.set_optimizer(opt_name, **kwargs)

    def set_loss(self, loss_name, **kwargs):
        self.net_model.set_loss(loss_name, **kwargs)

    def add(self, layer):
        self.net_model.add(layer)

    def train(self, train_iterator, test_iterator, batch_size, batch_size_test=None, restore_model=False, epochs=300,
              embedding_num=None, early_stop=None, sample_per_epoch=391):
        if batch_size_test is None:
            batch_size_test = batch_size
        if early_stop is not None:
            early_stop = {"accuracy": early_stop}
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size_test, sample_per_epoch=sample_per_epoch, epochs=epochs,
                             embedding_num=embedding_num, early_stop=early_stop)

    def restore(self):
        self.net_model.restore()


if __name__ == '__main__':
    from cnn_models.iterators.mnist import MNISTDataset
    mnist_train = MNISTDataset("/home/filip/Datasets", resolution="28x28", train_set=True)
    mnist_test = MNISTDataset("/home/filip/Datasets", resolution="28x28", train_set=False)
    dense_mnist = DenseMNIST(input_size=[28, 28, 1], output_size=10,
                             log_path="/home/filip/tensor_logs/Dense_MNIST",
                             metrics=["accuracy", "cross_entropy"])
    dense_mnist.build_model()
    dense_mnist.set_optimizer("Adam")
    dense_mnist.set_loss("cross_entropy", activation="softmax")
    dense_mnist.model_compile(0.003)
    dense_mnist.train(mnist_train, mnist_test, batch_size_test=10000, batch_size=100, epochs=550)
