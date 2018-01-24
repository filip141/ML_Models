import cv2
import random
import numpy as np
import tensorflow as tf
from simple_network.models import NetworkParallel
from simple_network.layers import ConvolutionalLayer, MaxPoolingLayer, FullyConnectedLayer, \
     Flatten, LeakyReluLayer, DropoutLayer, LocalResponseNormalization


class AlexNetModel(object):

    def __init__(self, input_size, output_size, log_path, metrics=None, stdev=0.012, conv_filters=16,
                 clear_temp=False):
        self.input_summary = {"img_number": 30}
        self.input_size = input_size
        self.output_size = output_size
        self.stdev = stdev
        self.conv_filters = conv_filters
        log_details = {"author": "Filip Bachura", "subject": "AlexNet for Assessing Image Quality",
                       "keywors": "NR-IQA, Deep Learning, Neural Networks, Non-Reference"}

        # Define model
        self.net_model = NetworkParallel(input_size, metric=metrics, input_summary=self.input_summary,
                                         summary_path=log_path, log_details=log_details, clear_temp=clear_temp)

    def build_model(self, model_classifier=True, optimizer=True, loss=True):
        # Layer 1
        standard_dev = self.stdev
        self.net_model.add(ConvolutionalLayer([11, 11, self.conv_filters], initializer="normal", name='convo_layer_1_1',
                                              stride=4, padding="valid", stddev=standard_dev))
        self.net_model.add(LeakyReluLayer(name="leaky_relu_1_1"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_1_1"))
        self.net_model.add(LocalResponseNormalization(depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0,
                                                      name="lrn_1_1"))

        # Layer 2
        self.net_model.add(ConvolutionalLayer([5, 5, self.conv_filters], initializer="normal", name='convo_layer_2_1',
                                              stride=1, padding="valid", stddev=standard_dev))
        self.net_model.add(LeakyReluLayer(name="leaky_relu_2_1"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_2_1"))
        self.net_model.add(LocalResponseNormalization(depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0,
                                                      name="lrn_2_1"))

        # Layer 3
        self.net_model.add(ConvolutionalLayer([3, 3, self.conv_filters], initializer="normal", name='convo_layer_3_1',
                                              stride=1, stddev=standard_dev))
        self.net_model.add(LeakyReluLayer(name="leaky_relu_3_1"))
        self.net_model.add(ConvolutionalLayer([3, 3, self.conv_filters], initializer="normal", name='convo_layer_3_2',
                                              stride=1, stddev=standard_dev))
        self.net_model.add(LeakyReluLayer(name="leaky_relu_3_2"))
        self.net_model.add(ConvolutionalLayer([3, 3, self.conv_filters], initializer="normal", name='convo_layer_3_3',
                                              stride=1, stddev=standard_dev))
        self.net_model.add(LeakyReluLayer(name="leaky_relu_3_2"))
        self.net_model.add(MaxPoolingLayer(pool_size=[3, 3], stride=2, padding="valid", name="pooling_3_1"))
        self.net_model.add(LocalResponseNormalization(depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0,
                                                      name="lrn_3_1"))

        if model_classifier:
            self.add_model_classifier(optimizer, loss)

    def model_compile(self, learning_rate, decay=None, decay_steps=100000, regularization=None, reg_lambda=0.001,
                      decay_type="exponential_decay"):
        self.net_model.build_model(learning_rate, decay, decay_steps, regularization=regularization,
                                   reg_lambda=reg_lambda, decay_type=decay_type)

    def add_model_classifier(self, optimizer=True, loss=True):
        # Layer 6
        self.net_model.add(Flatten(name='flatten_6'))
        self.net_model.add(FullyConnectedLayer(out_neurons=1000, initializer="normal",
                                               name='fully_connected_4_1', stddev=0.05))
        self.net_model.add(LeakyReluLayer(name="leaky_relu_4_1"))
        self.net_model.add(DropoutLayer(name="dropout_4_1"))
        self.net_model.add(FullyConnectedLayer(out_neurons=self.output_size, initializer="normal",
                                               name='fully_connected_5_1', stddev=0.05))
        if optimizer:
            self.net_model.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if loss:
            self.net_model.set_loss("mse")

    def set_optimizer(self, opt_name, **kwargs):
        self.net_model.set_optimizer(opt_name, **kwargs)

    def set_loss(self, loss_name, **kwargs):
        self.net_model.set_loss(loss_name, **kwargs)

    def add(self, layer):
        self.net_model.add(layer)

    def train(self, train_iterator, test_iterator, train_step, test_step, restore_model=False, epochs=300,
              embedding_num=None, early_stop=None, sample_per_epoch=391, summary_step=20, discrete_metric=None,
              d_metric_steps=5):
        if early_stop is not None:
            early_stop = {"accuracy": early_stop}
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=train_step, test_iter=test_iterator,
                             test_step=test_step, sample_per_epoch=sample_per_epoch, epochs=epochs,
                             embedding_num=embedding_num, discrete_metric=discrete_metric,
                             d_metric_steps=d_metric_steps, early_stop=early_stop, summary_step=summary_step)

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
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)

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
    from cnn_models.iterators.tools import ImageIterator
    from cnn_models.iterators.tid2013 import TID2013Dataset
    dataset_path = "/home/phoenix/Datasets/tid2013"
    cnd_train = TID2013Dataset(data_path=dataset_path, new_resolution=None, patches="227x227", patches_method='random',
                               no_patches=1, is_train=True)
    cnd_test = TID2013Dataset(data_path=dataset_path, new_resolution=None, patches="227x227", patches_method='random',
                              no_patches=1, is_train=False)
    cnd_train = ImageIterator(cnd_train, preprocess='lcn-lecun')
    cnd_test = ImageIterator(cnd_test, preprocess='lcn-lecun')
    im_net_model = AlexNetModel(input_size=[227, 227, 3], output_size=1,
                                log_path="/home/phoenix/tensor_logs/TID2013Db_LCN_LRN_64_FC",
                                metrics=["mse"], conv_filters=64, clear_temp=True)
    im_net_model.build_model(loss=False, optimizer=False, model_classifier=True)
    im_net_model.set_optimizer("Momentum", use_nesterov=True)
    im_net_model.set_loss("mse")
    im_net_model.model_compile(0.005, decay=0.9, decay_steps=94, regularization="L2-bias", reg_lambda=0.0005,
                               decay_type="inverse_time_decay")
    im_net_model.train(cnd_train, cnd_test, train_step=32, test_step=32, sample_per_epoch=94, restore_model=False,
                       discrete_metric="SROCC", d_metric_steps=20, summary_step=10)
