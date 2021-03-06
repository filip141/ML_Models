from simple_network.models import NetworkParallel
from simple_network.layers import BatchNormalizationLayer, FullyConnectedLayer, \
     Flatten, LinearLayer, ConvolutionalLayer, DropoutLayer, ReluLayer, MaxPoolingLayer, \
    LeakyReluLayer, SpatialDropoutLayer, SingleBatchNormLayer


class CNNCifar10(object):

    def __init__(self, input_size, output_size, log_path, metrics=None):
        self.input_summary = {"img_number": 30}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkParallel(input_size, metric=metrics, input_summary=self.input_summary,
                                         summary_path=log_path)

    def build_model(self):
        # Layer 1
        self.net_model.add(ConvolutionalLayer([3, 3, 48], initializer="xavier", name='convo_layer_1'))
        self.net_model.add(SingleBatchNormLayer(name="batch_norm_1"))
        self.net_model.add(ReluLayer(name="relu_1"))
        self.net_model.add(ConvolutionalLayer([3, 3, 48], initializer="xavier", name='convo_layer_2'))
        self.net_model.add(SingleBatchNormLayer(name="batch_norm_2"))
        self.net_model.add(ReluLayer(name="relu_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_2"))

        self.net_model.add(ConvolutionalLayer([3, 3, 96], initializer="xavier", name='convo_layer_3'))
        self.net_model.add(SingleBatchNormLayer(name="batch_norm_3"))
        self.net_model.add(ReluLayer(name="relu_3"))
        self.net_model.add(ConvolutionalLayer([3, 3, 96], initializer="xavier", name='convo_layer_4'))
        self.net_model.add(SingleBatchNormLayer(name="batch_norm_4"))
        self.net_model.add(ReluLayer(name="relu_4"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_4"))

        self.net_model.add(ConvolutionalLayer([3, 3, 192], initializer="xavier", name='convo_layer_5'))
        self.net_model.add(SingleBatchNormLayer(name="batch_norm_5"))
        self.net_model.add(ReluLayer(name="relu_5"))
        self.net_model.add(ConvolutionalLayer([3, 3, 192], initializer="xavier", name='convo_layer_6'))
        self.net_model.add(SingleBatchNormLayer(name="batch_norm_6"))
        self.net_model.add(ReluLayer(name="relu_6"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_6"))

        self.net_model.add(Flatten(name="flatten_7"))
        self.net_model.add(FullyConnectedLayer(out_neurons=512, initializer="xavier", name='fully_connected_7'))
        self.net_model.add(SingleBatchNormLayer(name="batch_norm_7"))
        self.net_model.add(ReluLayer(name="relu_7"))
        self.net_model.add(DropoutLayer(percent=0.5, name="dropout_7"))

        self.net_model.add(FullyConnectedLayer(out_neurons=256, initializer="xavier", name='fully_connected_8'))
        self.net_model.add(SingleBatchNormLayer(name="batch_norm_8"))
        self.net_model.add(ReluLayer(name="relu_8"))
        self.net_model.add(DropoutLayer(percent=0.5, name="dropout_8"))

        self.net_model.add(FullyConnectedLayer(out_neurons=self.output_size, initializer="xavier",
                                               name='fully_connected_9'))
        self.net_model.add(LinearLayer(name="linear_9"))

    def model_compile(self, learning_rate, decay=None, decay_steps=100000):
        self.net_model.build_model(learning_rate, decay, decay_steps)

    def set_optimizer(self, opt_name, **kwargs):
        self.net_model.set_optimizer(opt_name, **kwargs)

    def set_loss(self, loss_name, **kwargs):
        self.net_model.set_loss(loss_name, **kwargs)

    def add(self, layer):
        self.net_model.add(layer)

    def train(self, train_iterator, test_iterator, batch_size, batch_size_test=None, restore_model=False, epochs=300,
              embedding_num=None, early_stop=None, sample_per_epoch=391, summary_step=10):
        if batch_size_test is None:
            batch_size_test = batch_size
        if early_stop is not None:
            early_stop = {"accuracy": early_stop}
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size_test, sample_per_epoch=sample_per_epoch, epochs=epochs,
                             embedding_num=embedding_num, early_stop=early_stop, summary_step=summary_step,
                             avg_buffor_size=30)

    def restore(self):
        self.net_model.restore()


if __name__ == '__main__':
    from cnn_models.iterators.cifar import CIFARDataset
    from cnn_models.iterators.tools import ImageIterator
    cifar_train_path = "/home/filip141/Datasets/cifar/train"
    cifar_test_path = "/home/filip141/Datasets/cifar/test"
    cifar_train = ImageIterator(CIFARDataset(data_path=cifar_train_path, resolution="32x32", force_overfit=False),
                                rotate=10, translate=3, max_zoom=2, adjust_contrast=0.2, additive_noise=0.01,
                                adjust_brightness=0.1)
    cifar_test = CIFARDataset(data_path=cifar_test_path, resolution="32x32", force_overfit=False)
    cnn_net = CNNCifar10(input_size=[32, 32, 3], output_size=10,
                         log_path="/home/filip141/tensor_logs/CNN_CIFAR",
                         metrics=["accuracy", "cross_entropy"])
    cnn_net.build_model()
    cnn_net.set_optimizer("Adam")
    cnn_net.set_loss("cross_entropy")
    cnn_net.model_compile(0.001)
    cnn_net.train(cifar_train, cifar_test, batch_size_test=128, batch_size=128, epochs=200, restore_model=False,
                  sample_per_epoch=391, summary_step=150)
