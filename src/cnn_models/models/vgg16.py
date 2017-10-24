import cv2
import numpy as np
from simple_network.models import NetworkParallel
from simple_network.layers import ConvolutionalLayer, MaxPoolingLayer, ReluLayer, FullyConnectedLayer, \
     Flatten, DropoutLayer, BatchNormalizationLayer


class VGG16Model(object):

    def __init__(self, input_size, output_size, log_path):
        self.input_summary = {"img_number": 30}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkParallel(input_size, metric=["accuracy", "cross_entropy"],
                                         input_summary=self.input_summary, summary_path=log_path)

    def build_model(self, model_classifier=True):
        self.net_model.add(ConvolutionalLayer([3, 3, 64], initializer="xavier", name='convo_layer_1_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1_1"))
        self.net_model.add(ReluLayer(name="relu_1_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 64], initializer="xavier", name='convo_layer_1_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_1_2"))
        self.net_model.add(ReluLayer(name="relu_1_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_1_2"))

        self.net_model.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_2_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_2_1"))
        self.net_model.add(ReluLayer(name="relu_2_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_2_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_2_2"))
        self.net_model.add(ReluLayer(name="relu_2_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_2_2"))

        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_3_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3_1"))
        self.net_model.add(ReluLayer(name="relu_3_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_3_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3_2"))
        self.net_model.add(ReluLayer(name="relu_3_2"))

        self.net_model.add(ConvolutionalLayer([3, 3, 256], initializer="xavier", name='convo_layer_3_3'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_3_3"))
        self.net_model.add(ReluLayer(name="relu_3_3"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_3_3"))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_4_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4_1"))
        self.net_model.add(ReluLayer(name="relu_4_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_4_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4_2"))
        self.net_model.add(ReluLayer(name="relu_4_2"))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_4_3'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_4_3"))
        self.net_model.add(ReluLayer(name="relu_4_3"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_4_3"))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_5_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_5_1"))
        self.net_model.add(ReluLayer(name="relu_5_1"))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_5_2'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_5_2"))
        self.net_model.add(ReluLayer(name="relu_5_2"))

        self.net_model.add(ConvolutionalLayer([3, 3, 512], initializer="xavier", name='convo_layer_5_3'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_5_3"))
        self.net_model.add(ReluLayer(name="relu_5_3"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pooling_5_3"))
        if model_classifier:
            self.add_model_classifier()

    def add_model_classifier(self):
        self.net_model.add(Flatten(name='flatten_6'))

        self.net_model.add(FullyConnectedLayer([512 * (self.input_size[0] / 32.0)**2, 512],
                                               initializer="xavier", name='fully_connected_6_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_6_1"))
        self.net_model.add(ReluLayer(name="relu_6_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        self.net_model.add(FullyConnectedLayer([512, 512],
                                               initializer="xavier", name='fully_connected_7_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_7_1"))
        self.net_model.add(ReluLayer(name="relu_7_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        self.net_model.add(FullyConnectedLayer([512, self.output_size], initializer="xavier",
                                               name='fully_connected_8_1'))
        self.set_optimizer()

    def set_optimizer(self):
        # self.net_model.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # self.net_model.set_optimizer("SGD")
        self.net_model.set_optimizer("Momentum")
        self.net_model.set_loss("cross_entropy")

    def add(self, layer):
        self.net_model.add(layer)

    def train(self, train_iterator, test_iterator, learning_rate, batch_size, restore_model=False, epochs=300,
              embedding_num=None, early_stop=None):
        self.net_model.build_model(learning_rate)
        if early_stop is not None:
            early_stop = {"accuracy": early_stop}
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size, sample_per_epoch=391, epochs=epochs, embedding_num=embedding_num,
                             early_stop=early_stop)

    def restore(self):
        self.build_model()
        self.net_model.build_model(0.001)
        self.net_model.restore()

    def predict(self, img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tuple(self.input_size[:-1]), interpolation=cv2.INTER_AREA)
        img_shape = img.shape
        batch_matrix = np.zeros((1, img_shape[0], img_shape[1], img_shape[2]))
        batch_matrix[0] = (img.astype('float32') - np.min(img)) / np.std(img)
        eval_list = self.net_model.sess.run(self.net_model.get_last_layer_prediction(),
                                            feed_dict={self.net_model.input_layer_placeholder: batch_matrix,
                                                       self.net_model.is_training_placeholder: False})
        return np.argmax(eval_list[0])

if __name__ == '__main__':
    from cnn_models.iterators.tools import ImageIterator
    from cnn_models.iterators.imagenet import DogsDataset
    train_path = '/home/phoenix/Datasets/StanfordDogs/Images'
    labels_path = '/home/phoenix/Datasets/StanfordDogs/Annotation'
    class_names_path = '/home/phoenix/Datasets/StanfordDogs/class_names.txt'
    im_net_train = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
                               train_set=True, resize_img="224x224")
    im_net_test = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
                              train_set=False, resize_img="224x224")
    im_net_train = ImageIterator(im_net_train, max_zoom=5, translate=10, rotate=20)
    im_net_test = ImageIterator(im_net_test)

    im_net_model = VGG16Model(input_size=[224, 224, 3], output_size=120, log_path="/home/phoenix/tensor_logs")
    im_net_model.build_model(model_classifier=False)
    im_net_model.add(Flatten(name='flatten_6'))
    im_net_model.add(DropoutLayer(percent=0.4))
    im_net_model.add(FullyConnectedLayer([25088, 2048],
                                         initializer="xavier", name='fully_connected_6_1'))
    im_net_model.add(BatchNormalizationLayer(name="batch_normalization_6_1"))
    im_net_model.add(ReluLayer(name="relu_6_1"))
    im_net_model.add(DropoutLayer(percent=0.4))

    im_net_model.add(FullyConnectedLayer([2048, 2048],
                                         initializer="xavier", name='fully_connected_7_1'))
    im_net_model.add(BatchNormalizationLayer(name="batch_normalization_7_1"))
    im_net_model.add(ReluLayer(name="relu_7_1"))
    im_net_model.add(DropoutLayer(percent=0.2))

    im_net_model.add(FullyConnectedLayer([2048, 120], initializer="xavier",
                                         name='fully_connected_8_1'))
    im_net_model.set_optimizer()
    im_net_model.train(im_net_train, im_net_test, 0.0001, 16, epochs=300, early_stop=0.9)


