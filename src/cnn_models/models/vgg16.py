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

    def build_model(self):
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

        self.net_model.add(Flatten(name='flatten_6'))

        self.net_model.add(FullyConnectedLayer([512 * (self.input_size[0] / 32.0)**2, 512],
                                               initializer="xavier", name='fully_connected_6_1'))
        self.net_model.add(BatchNormalizationLayer(name="batch_normalization_6_1"))
        self.net_model.add(ReluLayer(name="relu_6_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        self.net_model.add(FullyConnectedLayer([512, self.output_size], initializer="xavier",
                                               name='fully_connected_7_1'))
        # self.net_model.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.net_model.set_optimizer("SGD")
        # self.net_model.set_optimizer("RMSprop")
        self.net_model.set_loss("cross_entropy")

    def train(self, train_iterator, test_iterator, learning_rate, batch_size, restore_model=False, epochs=300,
              embedding_num=None):
        self.net_model.build_model(learning_rate)
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size, sample_per_epoch=391, epochs=epochs, embedding_num=embedding_num)

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
    # from cnn_models.iterators.imagenet import DogsDataset
    # train_path = '/home/filip/Datasets/StanfordDogs/Images'
    # labels_path = '/home/filip/Datasets/StanfordDogs/Annotation'
    # class_names_path = '/home/filip/Datasets/StanfordDogs/class_names.txt'
    # im_net_train = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
    #                            train_set=True)
    # im_net_test = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
    #                           train_set=False)
    # im_net_model = VGG16Model(input_size=[64, 64, 3], output_size=120, log_path="/home/filip/tensor_logs")
    # im_net_model.build_model()
    # im_net_model.train(im_net_train, im_net_test, 0.0005, 64, epochs=300)
    from cnn_models.iterators.cifar import CIFARDataset
    train_path = "/home/filip/Datasets/cifar/train"
    test_path = "/home/filip/Datasets/cifar/test"
    cifar_train = CIFARDataset(data_path=train_path, resolution="32x32")
    cifar_test = CIFARDataset(data_path=test_path, resolution="32x32")

    im_net_model = VGG16Model(input_size=[32, 32, 3], output_size=10, log_path="/home/filip/tensor_logs")
    # im_net_model.train(cifar_train, cifar_train, 0.001, 32, epochs=300)
    im_net_model.restore()
    print(im_net_model.predict("/home/filip/car.jpg"))


