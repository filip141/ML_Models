import cv2
import numpy as np
import tensorflow as tf
from simple_network.models import NetworkNode, NetworkParallel
from simple_network.layers import ConvolutionalLayer, MaxPoolingLayer, ReluLayer, FullyConnectedLayer, \
     Flatten, DropoutLayer, BatchNormalizationLayer, LocalResponseNormalization, SplitterLayer


ALEXNET_MAPPING_TWO_STREAMS = {"conv1": "convo_layer_1_1", "conv2": ("convo_layer_2_1", "convo_layer_2_1"),
                               "conv3": "convo_layer_3_1", "conv4": ("convo_layer_4_1", "convo_layer_4_2"),
                               "conv5": ("convo_layer_5_1", "convo_layer_5_2"), "fc6": "fully_connected_6_1",
                               "fc7": "fully_connected_7_1", "fc8": "fully_connected_8_1"}


class AlexNetModelBN(object):

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


class AlexNetModel(object):

    def __init__(self, input_size, output_size, log_path):
        self.input_summary = {"img_number": 30}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkParallel(input_size, metric=["accuracy", "cross_entropy"],
                                         input_summary=self.input_summary, summary_path=log_path)

    def build_model(self, true_build=False):
        # Layer 1
        self.net_model.add(ConvolutionalLayer([11, 11, 96], initializer="xavier", name='convo_layer_1_1', stride=4,
                                              activation="relu", padding="valid"))
        self.net_model.add(LocalResponseNormalization(depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0,
                                                      name="lrn_1_1"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_1_1"))

        # Layer 2
        self.net_model.add(SplitterLayer(num_split=2))
        net_node = NetworkNode(name="convolutional_node_layer_2", reduce_output="concat")
        net_node.add(ConvolutionalLayer([5, 5, 128], initializer="xavier", name='convo_layer_2_1', stride=1,
                                        activation="relu"))
        net_node.add(ConvolutionalLayer([5, 5, 128], initializer="xavier", name='convo_layer_2_2', stride=1,
                                        activation="relu"))
        self.net_model.add(net_node)
        self.net_model.add(LocalResponseNormalization(depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0,
                                                      name="lrn_2_1"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_2_1"))

        # Layer 3
        self.net_model.add(ConvolutionalLayer([3, 3, 384], initializer="xavier", name='convo_layer_3_1', stride=1,
                                              activation="relu"))

        # Layer 4
        self.net_model.add(SplitterLayer(num_split=2))
        net_node = NetworkNode(name="convolutional_node_layer_4", reduce_output="concat")
        net_node.add(ConvolutionalLayer([3, 3, 192], initializer="xavier", name='convo_layer_4_1', stride=1,
                                        activation="relu"))
        net_node.add(ConvolutionalLayer([3, 3, 192], initializer="xavier", name='convo_layer_4_2', stride=1,
                                        activation="relu"))
        self.net_model.add(net_node)

        # Layer 5
        self.net_model.add(SplitterLayer(num_split=2))
        net_node = NetworkNode(name="convolutional_node_layer_5", reduce_output="concat")
        net_node.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_5_1', stride=1,
                                        activation="relu"))
        net_node.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_5_2', stride=1,
                                        activation="relu"))
        self.net_model.add(net_node)
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_2_1"))

        # Layer 6
        self.net_model.add(Flatten(name='flatten_6'))
        self.net_model.add(FullyConnectedLayer([9216, 4096],
                                               initializer="xavier", name='fully_connected_6_1'))
        self.net_model.add(ReluLayer(name="relu_6_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        # Layer 7
        self.net_model.add(FullyConnectedLayer([4096, 4096], initializer="xavier", name='fully_connected_7_1'))
        self.net_model.add(ReluLayer(name="relu_7_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        # Layer 8
        self.net_model.add(FullyConnectedLayer([4096, self.output_size], initializer="xavier",
                                               name='fully_connected_8_1'))
        # self.net_model.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.net_model.set_optimizer("SGD")
        self.net_model.set_loss("cross_entropy")
        if true_build:
            self.net_model.build_model(0.001)

    def train(self, train_iterator, test_iterator, learning_rate, batch_size, restore_model=False, epochs=300,
              model_build=True):
        if model_build:
            self.net_model.build_model(learning_rate)
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size, sample_per_epoch=391, epochs=epochs)

    def load_initial_weights(self, path, mapping):
        weights_dict = np.load(path, encoding='bytes').item()
        for op_name, weights in weights_dict.items():
            if op_name in mapping.keys():
                op_names_new = mapping[op_name]
                if isinstance(op_names_new, str):
                    op_names_new = (op_names_new,)
                t_weights = np.split(weights[0], len(op_names_new), axis=-1)
                t_biases = np.split(weights[1], len(op_names_new), axis=-1)
                for t_name, t_w, t_b in zip(op_names_new, t_weights, t_biases):
                    with tf.variable_scope(t_name, reuse=True):
                        var = tf.get_variable('biases', trainable=False)
                        self.net_model.sess.run(var.assign(t_b))
                        var = tf.get_variable('weights', trainable=False)
                        self.net_model.sess.run(var.assign(t_w))

    def predict(self, img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tuple(self.input_size[:-1]), interpolation=cv2.INTER_AREA)
        img_shape = img.shape
        batch_matrix = np.zeros((1, img_shape[0], img_shape[1], img_shape[2]))
        batch_matrix[0] = img - np.array([104., 117., 124.], dtype=np.float32)
        eval_list = self.net_model.sess.run(self.net_model.get_last_layer_prediction(),
                                            feed_dict={self.net_model.input_layer_placeholder: batch_matrix,
                                                       self.net_model.is_training_placeholder: False})
        return np.argmax(np.exp(eval_list[0]) / np.sum(np.exp(eval_list[0]), axis=0))


if __name__ == '__main__':
    # from cnn_models.iterators.imagenet import DogsDataset
    # from cnn_models.iterators.tools import ImageIterator
    # train_path = '/home/filip/Datasets/StanfordDogs/Images'
    # labels_path = '/home/filip/Datasets/StanfordDogs/Annotation'
    # class_names_path = '/home/filip/Datasets/StanfordDogs/class_names.txt'
    # im_net_train = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
    #                            train_set=True, resize_img="227x227")
    # im_net_test = DogsDataset(data_path=train_path, labels_path=labels_path, class_names=class_names_path,
    #                           train_set=False, resize_img="227x227")
    # im_net_train = ImageIterator(im_net_train)
    # im_net_test = ImageIterator(im_net_test)
    im_net_model = AlexNetModel(input_size=[227, 227, 3], output_size=1000, log_path="/home/filip/tensor_logs")
    im_net_model.build_model(true_build=True)
    im_net_model.load_initial_weights("/home/filip/Weights/bvlc_alexnet.npy", ALEXNET_MAPPING_TWO_STREAMS)
    print(im_net_model.predict("/home/filip/PycharmProjects/finetune_alexnet_with_tensorflow/images/zebra.jpeg"))
    # im_net_model.train(im_net_train, im_net_test, 0.001, 16, epochs=300, model_build=False)

