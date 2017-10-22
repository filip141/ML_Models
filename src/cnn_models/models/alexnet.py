from simple_network.models import NetworkModel
from simple_network.layers import ConvolutionalLayer, MaxPoolingLayer, ReluLayer, FullyConnectedLayer, \
     Flatten, DropoutLayer, BatchNormalizationLayer


class AlexNetModel(object):

    def __init__(self, input_size, output_size, log_path):
        self.input_summary = {"img_number": 30}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkModel(input_size, metric=["accuracy", "cross_entropy"],
                                      input_summary=self.input_summary, summary_path=log_path)

    def build_model(self):
        self.net_model.add(ConvolutionalLayer([5, 5, 96], initializer="xavier", name='convo_layer_1_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1_1"))
        self.net_model.add(ReluLayer(name="relu_1_1"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_1_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_2_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_2_1"))
        self.net_model.add(ReluLayer(name="relu_2_1"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_2_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 384], initializer="xavier", name='convo_layer_3_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3_1"))
        self.net_model.add(ReluLayer(name="relu_3_1"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_2_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 384], initializer="xavier", name='convo_layer_4_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4_1"))
        self.net_model.add(ReluLayer(name="relu_4_1"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_4_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_5_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_5_1"))
        self.net_model.add(ReluLayer(name="relu_5_1"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_5_1"))

        self.net_model.add(Flatten(name='flatten_5'))

        self.net_model.add(FullyConnectedLayer([256 * ((self.input_size[0] / 32.0) - 1)**2, 4096],
                                               initializer="xavier", name='fully_connected_6_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_6_1"))
        self.net_model.add(ReluLayer(name="relu_6_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        self.net_model.add(FullyConnectedLayer([4096, 4096], initializer="xavier", name='fully_connected_7_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_7_1"))
        self.net_model.add(ReluLayer(name="relu_7_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        self.net_model.add(FullyConnectedLayer([4096, self.output_size], initializer="xavier",
                                               name='fully_connected_8_1'))
        # self.net_model.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.net_model.set_optimizer("SGD")
        self.net_model.set_loss("cross_entropy")

    def train(self, train_iterator, test_iterator, learning_rate, batch_size, restore_model=False, epochs=300):
        self.net_model.build_model(learning_rate)
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size, sample_per_epoch=391, epochs=epochs)

if __name__ == '__main__':
    # from cnn_models.iterators.imagenet import DogsDataset
    # train_path = '/home/filip/Datasets/StanfordDogs/Images'
    # labels_path = '/home/filip/Datasets/StanfordDogs/Annotation'
    # class_names_path = '/home/filip/Datasets/StanfordDogs/class_names.txt'
    # im_net_train = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
    #                            train_set=True, resize_img="64x64")
    # im_net_test = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
    #                           train_set=False, resize_img="64x64")
    from cnn_models.iterators.cifar import CIFARDataset
    train_path = "/home/filip/Datasets/cifar/train"
    test_path = "/home/filip/Datasets/cifar/test"
    cifar_train = CIFARDataset(data_path=train_path, resolution="64x64")
    cifar_test = CIFARDataset(data_path=test_path, resolution="64x64")

    im_net_model = AlexNetModel(input_size=[64, 64, 3], output_size=10, log_path="/home/filip/tensor_logs")
    im_net_model.build_model()
    im_net_model.train(cifar_train, cifar_test, 0.001, 32, epochs=300, restore_model=True)


