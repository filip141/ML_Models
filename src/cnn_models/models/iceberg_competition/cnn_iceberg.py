from simple_network.models import NetworkParallel
from simple_network.layers import BatchNormalizationLayer, FullyConnectedLayer, \
     Flatten, LinearLayer, ConvolutionalLayer, DropoutLayer, ReluLayer, MaxPoolingLayer


class CNNIceberg(object):

    def __init__(self, input_size, output_size, log_path, metrics=None):
        self.input_summary = {"img_number": 5}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkParallel(input_size, metric=metrics, input_summary=self.input_summary,
                                         summary_path=log_path)

    def build_model(self):
        # Layer 1
        self.net_model.add(ConvolutionalLayer([6, 6, 64], initializer="xavier", name='convo_layer_1'))
        self.net_model.add(ReluLayer(name="relu_1"))

        self.net_model.add(ConvolutionalLayer([5, 5, 128], initializer="xavier", name='convo_layer_2'))
        self.net_model.add(ReluLayer(name="relu_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[5, 5], stride=2, padding="valid", name="pooling_2"))

        self.net_model.add(Flatten(name="flatten_3"))
        self.net_model.add(FullyConnectedLayer([165888, 1024],
                                               initializer="xavier", name='fully_connected_3'))
        self.net_model.add(ReluLayer(name="relu_3"))
        self.net_model.add(DropoutLayer(percent=0.4, name="dropout_3"))

        self.net_model.add(FullyConnectedLayer([1024, self.output_size],
                                               initializer="xavier", name='fully_connected_4'))
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
    from cnn_models.iterators.iceberg import IcebergDataset
    from cnn_models.iterators.tools import ImageIterator
    json_data_train = "/home/filip/Datasets/Iceberg_data/train/processed/train.json"
    iceberg_db_train = IcebergDataset(json_path=json_data_train, batch_out="color_composite_nn", divide_point=0.2)
    iceberg_db_test = IcebergDataset(json_path=json_data_train, is_test=True, batch_out="color_composite_nn",
                                     divide_point=0.2)
    cnn_iceberg = CNNIceberg(input_size=[75, 75, 3], output_size=1,
                             metrics=["binary_accuracy", "cross_entropy_sigmoid"],
                             log_path="/home/filip/tensor_logs/CNN_ICEBERG_NN")
    cnn_iceberg.build_model()
    cnn_iceberg.set_optimizer("SGD")
    cnn_iceberg.set_loss("cross_entropy", activation="sigmoid")
    cnn_iceberg.model_compile(0.0001)
    cnn_iceberg.train(iceberg_db_train, iceberg_db_test, batch_size=32, batch_size_test=310)
