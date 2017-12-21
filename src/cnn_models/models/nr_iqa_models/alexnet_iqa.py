import cv2
import random
import numpy as np
import tensorflow as tf
from simple_network.models import NetworkNode, NetworkParallel
from simple_network.layers import ConvolutionalLayer, MaxPoolingLayer, ReluLayer, FullyConnectedLayer, \
     Flatten, DropoutLayer, LocalResponseNormalization, SplitterLayer, SpatialDropoutLayer


ALEXNET_MAPPING_TWO_STREAMS = {"conv1": "convo_layer_1_1", "conv2": ("convo_layer_2_1", "convo_layer_2_2"),
                               "conv3": "convo_layer_3_1", "conv4": ("convo_layer_4_1", "convo_layer_4_2"),
                               "conv5": ("convo_layer_5_1", "convo_layer_5_2"), "fc6": "fully_connected_6_1",
                               "fc7": "fully_connected_7_1", "fc8": "fully_connected_8_1"}


class AlexNetModel(object):

    def __init__(self, input_size, output_size, log_path, metrics=None):
        self.input_summary = {"img_number": 30}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkParallel(input_size, metric=metrics, input_summary=self.input_summary,
                                         summary_path=log_path)

    def build_model(self, model_classifier=True, optimizer=True, loss=True):
        # Layer 1
        self.net_model.add(ConvolutionalLayer([11, 11, 96], initializer="xavier", name='convo_layer_1_1', stride=4,
                                              activation="relu", padding="valid"))
        self.net_model.add(LocalResponseNormalization(depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0,
                                                      name="lrn_1_1"))
        self.net_model.add(SpatialDropoutLayer(percent=0.4, name='dropout_1_1'))
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
        self.net_model.add(SpatialDropoutLayer(percent=0.4, name='dropout_2_1'))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_2_1"))

        # Layer 3
        self.net_model.add(ConvolutionalLayer([3, 3, 384], initializer="xavier", name='convo_layer_3_1', stride=1,
                                              activation="relu"))
        self.net_model.add(SpatialDropoutLayer(percent=0.4, name='dropout_3_1'))

        # Layer 4
        self.net_model.add(SplitterLayer(num_split=2))
        net_node = NetworkNode(name="convolutional_node_layer_4", reduce_output="concat")
        net_node.add(ConvolutionalLayer([3, 3, 192], initializer="xavier", name='convo_layer_4_1', stride=1,
                                        activation="relu"))
        net_node.add(ConvolutionalLayer([3, 3, 192], initializer="xavier", name='convo_layer_4_2', stride=1,
                                        activation="relu"))
        self.net_model.add(net_node)
        self.net_model.add(SpatialDropoutLayer(percent=0.4, name='dropout_4_1'))

        # Layer 5
        self.net_model.add(SplitterLayer(num_split=2))
        net_node = NetworkNode(name="convolutional_node_layer_5", reduce_output="concat")
        net_node.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_5_1', stride=1,
                                        activation="relu"))
        net_node.add(ConvolutionalLayer([3, 3, 128], initializer="xavier", name='convo_layer_5_2', stride=1,
                                        activation="relu"))
        self.net_model.add(net_node)
        self.net_model.add(SpatialDropoutLayer(percent=0.4, name='dropout_5_1'))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_2_1"))

        if model_classifier:
            self.add_model_classifier(optimizer, loss)

    def model_compile(self, learning_rate):
        self.net_model.build_model(learning_rate)

    def add_model_classifier(self, optimizer=True, loss=True):
        # Layer 6
        self.net_model.add(Flatten(name='flatten_6'))
        self.net_model.add(FullyConnectedLayer(out_neurons=4096, initializer="xavier", name='fully_connected_6_1'))
        self.net_model.add(ReluLayer(name="relu_6_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        # Layer 7
        self.net_model.add(FullyConnectedLayer(out_neurons=4096, initializer="xavier", name='fully_connected_7_1'))
        self.net_model.add(ReluLayer(name="relu_7_1"))
        self.net_model.add(DropoutLayer(percent=0.5))

        # Layer 8
        self.net_model.add(FullyConnectedLayer(out_neurons=self.output_size, initializer="xavier",
                                               name='fully_connected_8_1'))
        if optimizer:
            self.net_model.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if loss:
            self.net_model.set_loss("cross_entropy")

    def set_optimizer(self, opt_name, **kwargs):
        self.net_model.set_optimizer(opt_name, **kwargs)

    def set_loss(self, loss_name, **kwargs):
        self.net_model.set_loss(loss_name, **kwargs)

    def add(self, layer):
        self.net_model.add(layer)

    def train(self, train_iterator, test_iterator, train_step, test_step, restore_model=False, epochs=300,
              embedding_num=None, early_stop=None, sample_per_epoch=391):
        if early_stop is not None:
            early_stop = {"accuracy": early_stop}
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=train_step, test_iter=test_iterator,
                             test_step=test_step, sample_per_epoch=sample_per_epoch, epochs=epochs,
                             embedding_num=embedding_num,
                             early_stop=early_stop)

    def restore(self):
        self.net_model.restore()

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
                        self.net_model.sess.run(self.net_model.get_layer_by_name(t_name).bias.assign(t_b))
                        self.net_model.sess.run(self.net_model.get_layer_by_name(t_name).weights.assign(t_w))

    def predict(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)

        img_shape = img.shape
        w_pos = random.randint(0, img_shape[1] - self.input_size[0])
        h_pos = random.randint(0, img_shape[0] - self.input_size[1])
        img = img[h_pos:h_pos + self.input_size[1], w_pos:w_pos + self.input_size[0]]
        batch_matrix = np.zeros((1, self.input_size[0], self.input_size[1], self.input_size[2]), dtype=np.float32)
        batch_matrix[0] = img
        eval_list = self.net_model.sess.run(self.net_model.get_last_layer_prediction(),
                                            feed_dict={self.net_model.input_layer_placeholder: batch_matrix,
                                                       self.net_model.is_training_placeholder: False})
        return eval_list


if __name__ == '__main__':
    from cnn_models.iterators.live_dataset import LIVEDataset
    dataset_path = "/home/phoenix/Datasets/Live2005"
    cnd_train = LIVEDataset(data_path=dataset_path, new_resolution=None, patches="227x227", patches_method='random',
                            no_patches=1, is_train=True)
    cnd_test = LIVEDataset(data_path=dataset_path, new_resolution=None, patches="227x227", patches_method='random',
                           no_patches=1, is_train=False)
    im_net_model = AlexNetModel(input_size=[227, 227, 3], output_size=1,
                                log_path="/home/phoenix/tensor_logs/LiveDb",
                                metrics=["mse", "mae"])
    im_net_model.build_model(loss=False, optimizer=False, model_classifier=False)

    im_net_model.net_model.add(Flatten(name='flatten_6'))
    im_net_model.net_model.add(FullyConnectedLayer(out_neurons=512, initializer="xavier", name='fully_connected_6_1'))
    im_net_model.net_model.add(ReluLayer(name="relu_6_1"))
    im_net_model.net_model.add(DropoutLayer(percent=0.5))
    im_net_model.net_model.add(FullyConnectedLayer(out_neurons=512, initializer="xavier", name='fully_connected_7_1'))
    im_net_model.net_model.add(ReluLayer(name="relu_7_1"))
    im_net_model.net_model.add(DropoutLayer(percent=0.5))
    im_net_model.net_model.add(FullyConnectedLayer(out_neurons=im_net_model.output_size, initializer="xavier",
                                                   name='fully_connected_8_1'))
    im_net_model.set_optimizer("SGD")
    im_net_model.set_loss("mae")
    im_net_model.model_compile(0.0002)
    # del ALEXNET_MAPPING_TWO_STREAMS["fc6"]
    # del ALEXNET_MAPPING_TWO_STREAMS["fc7"]
    # del ALEXNET_MAPPING_TWO_STREAMS["fc8"]
    # im_net_model.load_initial_weights("/home/phoenix/Weights/bvlc_alexnet.npy", ALEXNET_MAPPING_TWO_STREAMS)
    im_net_model.train(cnd_train, cnd_test, train_step=4, test_step=49, epochs=246)
