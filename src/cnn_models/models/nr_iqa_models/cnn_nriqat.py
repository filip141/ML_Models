import tensorflow as tf
from simple_network.models import NetworkParallel
from simple_network.layers import BatchNormalizationLayer, FullyConnectedLayer, \
     Flatten, LinearLayer, ConvolutionalLayer, DropoutLayer, ReluLayer, MaxPoolingLayer, SpatialDropoutLayer, \
    SwishLayer


class CNNNRQIQA(object):

    def __init__(self, input_size, output_size, log_path, metrics=None):
        self.input_summary = {"img_number": 30}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkParallel(input_size, metric=metrics, input_summary=self.input_summary,
                                         summary_path=log_path)

    def build_model(self):
        # Layer 1
        self.net_model.add(ConvolutionalLayer([3, 3, 32], initializer="xavier", name='convo_layer_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1"))
        self.net_model.add(SwishLayer(name="swish_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 64], initializer="xavier", name='convo_layer_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_2"))
        self.net_model.add(SwishLayer(name="swish_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_2"))
        self.net_model.add(SpatialDropoutLayer(percent=0.4, name="dropout_2"))
        #
        # self.net_model.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_3'))
        # self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3"))
        # self.net_model.add(SwishLayer(name="swish_3"))
        # self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_3"))
        # self.net_model.add(SpatialDropoutLayer(percent=0.4, name="dropout_3"))

        self.net_model.add(Flatten(name="flatten_6"))
        self.net_model.add(FullyConnectedLayer(out_neurons=256, initializer="xavier", name='fully_connected_6'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_6"))
        self.net_model.add(SwishLayer(name="swish_6"))
        self.net_model.add(DropoutLayer(percent=0.5, name="dropout_6"))

        self.net_model.add(FullyConnectedLayer(out_neurons=self.output_size, initializer="xavier",
                                               name='fully_connected_7'))
        self.net_model.add(LinearLayer(name="linear_7"))

    def model_compile(self, learning_rate):
        self.net_model.build_model(learning_rate)

    def set_optimizer(self, opt_name, **kwargs):
        self.net_model.set_optimizer(opt_name, **kwargs)

    def set_loss(self, loss_name, **kwargs):
        self.net_model.set_loss(loss_name, **kwargs)

    def add(self, layer):
        self.net_model.add(layer)

    def train(self, train_iterator, test_iterator, train_step, test_step, restore_model=False,
              epochs=300, embedding_num=None, early_stop=None, sample_per_epoch=391):
        if early_stop is not None:
            early_stop = {"accuracy": early_stop}
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=train_step, test_iter=test_iterator,
                             test_step=test_step, sample_per_epoch=sample_per_epoch, epochs=epochs,
                             embedding_num=embedding_num, early_stop=early_stop)

    def restore(self):
        self.net_model.restore()


def convolutional_learning_test(learning_rate, batch_size):
    from cnn_models.iterators.live_dataset import LIVEDataset
    dataset_path = "/home/phoenix/Datasets/Live2005"
    cnd_train = LIVEDataset(data_path=dataset_path, new_resolution=None, patches="64x64", patches_method='random',
                            no_patches=1, is_train=True)
    cnd_test = LIVEDataset(data_path=dataset_path, new_resolution=None, patches="64x64", patches_method='random',
                           no_patches=1, is_train=False)
    cnn_nrq_iqa = CNNNRQIQA(input_size=[64, 64, 3], output_size=1,
                            log_path="/home/phoenix/tensor_logs/CNN_NRQIQA_2l_{}_{}".format(learning_rate, batch_size),
                            metrics=["mse", "mae"])
    cnn_nrq_iqa.build_model()
    cnn_nrq_iqa.set_optimizer("SGD")
    cnn_nrq_iqa.set_loss("mae")
    cnn_nrq_iqa.model_compile(learning_rate)
    cnn_nrq_iqa.train(cnd_train, cnd_test, train_step=batch_size, test_step=49, epochs=5)
    tf.reset_default_graph()


if __name__ == '__main__':
    import random
    for n in range(0, 15):
        learning_rate = random.randint(1, 100) / 1000.0
        batch_size = random.randint(1, 32)
        convolutional_learning_test(learning_rate, batch_size)