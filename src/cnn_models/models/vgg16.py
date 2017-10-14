from simple_network.models import NetworkModel
from simple_network.layers import ConvolutionalLayer, MaxPoolingLayer, ReluLayer, FullyConnectedLayer, \
     Flatten, DropoutLayer, BatchNormalizationLayer


class VGG16Model(object):

    def __init__(self, input_size, output_size, log_path):
        self.input_summary = {"img_number": 30}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkModel(input_size, metric=["accuracy", "cross_entropy"],
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

        self.net_model.add(FullyConnectedLayer([2048, 512], initializer="xavier", name='fully_connected_6_1'))
        self.net_model.add(ReluLayer(name="relu_6_1"))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_6_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        self.net_model.add(FullyConnectedLayer([512, self.output_size], initializer="xavier",
                                               name='fully_connected_7_1'))
        self.net_model.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.net_model.set_loss("cross_entropy")

    def train(self, train_iterator, test_iterator, learning_rate, batch_size, restore_model=False, epochs=300):
        self.net_model.build_model(learning_rate)
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size, sample_per_epoch=391, epochs=epochs)

if __name__ == '__main__':
    from cnn_models.iterators.imagenet import DogsDataset
    train_path = '/home/phoenix/Datasets/StanfordDogs/Images'
    labels_path = '/home/phoenix/Datasets/StanfordDogs/Annotation'
    class_names_path = '/home/phoenix/Datasets/StanfordDogs/class_names.txt'
    im_net_train = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
                               train_set=True)
    im_net_test = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
                              train_set=False)
    im_net_model = VGG16Model(input_size=[64, 64, 3], output_size=120, log_path="/home/phoenix/tensor_logs")
    im_net_model.build_model()
    im_net_model.train(im_net_train, im_net_test, 0.001, 64, epochs=300)


