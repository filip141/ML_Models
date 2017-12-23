from simple_network.models import NetworkParallel, ResidualNode
from simple_network.layers import ConvolutionalLayer, ReluLayer, FullyConnectedLayer, \
    Flatten, BatchNormalizationLayer, GlobalAveragePoolingLayer, LeakyReluLayer, MaxPoolingLayer


class ResNet50(object):

    def __init__(self, input_size, output_size, metrics, log_path):
        self.input_summary = {"img_number": 5}
        self.input_size = input_size
        self.output_size = output_size

        # Define model
        self.net_model = NetworkParallel(input_size, metric=metrics,
                                         input_summary=self.input_summary, summary_path=log_path)

    @staticmethod
    def bottleneck_block(l_idx, letter, b_stride=1):
        bottleneck_node = ResidualNode(name="res{}{}".format(l_idx + 2, letter), ntimes=1)
        bottleneck_node.add(ConvolutionalLayer([1, 1, 64 * 2 ** l_idx], initializer="xavier",
                                               name='res{}{}_branch2a'.format(l_idx + 2, letter),
                                               summaries=False, stride=b_stride))
        bottleneck_node.add(BatchNormalizationLayer(name="bn{}{}_branch2a".format(l_idx + 2, letter),
                                                    summaries=False))
        bottleneck_node.add(ReluLayer(name="res{}{}_branch2a_relu".format(l_idx + 2, letter)))

        # Bottleneck node layers 2
        bottleneck_node.add(ConvolutionalLayer([1, 1, 64 * 2 ** l_idx], initializer="xavier",
                                               name='res{}{}_branch2b'.format(l_idx + 2, letter),
                                               summaries=False, stride=1))
        bottleneck_node.add(BatchNormalizationLayer(name="bn{}{}_branch2b".format(l_idx + 2, letter),
                                                    summaries=False))
        bottleneck_node.add(ReluLayer(name="res{}{}_branch2b_relu".format(l_idx + 2, letter)))

        # Bottleneck node layers 3
        bottleneck_node.add(ConvolutionalLayer([1, 1, 256 * 2 ** l_idx], initializer="xavier",
                                               name='res{}{}_branch2c'.format(l_idx + 2, letter),
                                               summaries=False, stride=1))
        bottleneck_node.add(BatchNormalizationLayer(name="bn{}{}_branch2c".format(l_idx + 2, letter),
                                                    summaries=False))
        bottleneck_node.add(ReluLayer(name="res{}{}_branch2c_relu".format(l_idx + 2, letter)))

        # Bottleneck residual
        bottleneck_node.add_residual(ConvolutionalLayer([1, 1, 256 * 2 ** l_idx], initializer="xavier",
                                                        stride=b_stride,
                                                        name='res{}{}_branch1'.format(l_idx + 2, letter),
                                                        summaries=False))
        bottleneck_node.add_residual(BatchNormalizationLayer(name="bn{}{}_branch1".format(l_idx + 2, letter),
                                                             summaries=False))
        return bottleneck_node

    @staticmethod
    def residual_block(l_idx, letters, ntimes=2):
        # Add residual node
        residual_node = ResidualNode(name="res{}{}".format(l_idx + 2, letters), ntimes=ntimes)

        # Residual node layers 1
        residual_node.add(ConvolutionalLayer([1, 1, 64 * 2 ** l_idx], initializer="xavier",
                                             name='res{}{}_branch2a'.format(l_idx + 2, letters), summaries=False))
        residual_node.add(BatchNormalizationLayer(name="bn{}{}_branch2a".format(l_idx + 2, letters), summaries=False))
        residual_node.add(ReluLayer(name="res{}{}_branch2a_relu".format(l_idx + 2, letters), summaries=False))

        # Residual node layers 2
        residual_node.add(ConvolutionalLayer([3, 3, 64 * 2 ** l_idx], initializer="xavier",
                                             name='res{}{}_branch2b'.format(l_idx + 2, letters), summaries=False))
        residual_node.add(BatchNormalizationLayer(name="bn{}{}_branch2b".format(l_idx + 2, letters), summaries=False))
        residual_node.add(ReluLayer(name="res{}{}_branch2b_relu".format(l_idx + 2, letters), summaries=False))

        # Residual node layers 3
        residual_node.add(ConvolutionalLayer([1, 1, 256 * 2 ** l_idx], initializer="xavier",
                                             name='res{}{}_branch2c'.format(l_idx + 2, letters), summaries=False))
        residual_node.add(BatchNormalizationLayer(name="bn{}{}_branch2c".format(l_idx + 2, letters), summaries=False))
        residual_node.add(ReluLayer(name="res{}{}_branch2c_relu".format(l_idx + 2, letters), summaries=False))
        return residual_node

    def build_model(self):
        self.net_model.add(ConvolutionalLayer([7, 7, 64], initializer="xavier", name='conv1', stride=2))
        self.net_model.add(BatchNormalizationLayer(name="bn_conv_1"))
        self.net_model.add(ReluLayer(name="conv1_relu"))
        self.net_model.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="valid", name="pool1"))

        # block 1
        bottleneck_node = self.bottleneck_block(l_idx=0, letter="a", b_stride=1)
        self.net_model.add(bottleneck_node)
        residual_node = self.residual_block(l_idx=0, letters="b", ntimes=2)
        self.net_model.add(residual_node)

        # block 2
        bottleneck_node = self.bottleneck_block(l_idx=1, letter="c", b_stride=2)
        self.net_model.add(bottleneck_node)
        residual_node = self.residual_block(l_idx=1, letters="f", ntimes=3)
        self.net_model.add(residual_node)

        # block 3
        bottleneck_node = self.bottleneck_block(l_idx=2, letter="e", b_stride=2)
        self.net_model.add(bottleneck_node)
        residual_node = self.residual_block(l_idx=2, letters="f", ntimes=5)
        self.net_model.add(residual_node)

        # block 4
        bottleneck_node = self.bottleneck_block(l_idx=2, letter="g", b_stride=2)
        self.net_model.add(bottleneck_node)
        residual_node = self.residual_block(l_idx=2, letters="h", ntimes=2)
        self.net_model.add(residual_node)

        self.net_model.add(GlobalAveragePoolingLayer(name="pool5"))
        self.net_model.add(FullyConnectedLayer(out_neurons=self.output_size, initializer="xavier",
                                               name='fc_end'))

    def model_compile(self, learning_rate, decay=None, decay_steps=100000):
        self.net_model.build_model(learning_rate, decay, decay_steps)

    def set_optimizer(self, opt_name, **kwargs):
        self.net_model.set_optimizer(opt_name, **kwargs)

    def set_loss(self, loss_name, **kwargs):
        self.net_model.set_loss(loss_name, **kwargs)

    def train(self, train_iterator, test_iterator, batch_size, batch_size_test=None, restore_model=False, epochs=300,
              summary_step=10, sample_per_epoch=10000):
        if batch_size_test is None:
            batch_size_test = batch_size
        if restore_model:
            self.net_model.restore()
        self.net_model.train(train_iter=train_iterator, train_step=batch_size, test_iter=test_iterator,
                             test_step=batch_size_test, sample_per_epoch=sample_per_epoch, epochs=epochs,
                             summary_step=summary_step)


if __name__ == '__main__':
    from cnn_models.iterators.cifar import CIFARDataset
    cifar_train_path = "/home/filip141/Datasets/cifar/train"
    cifar_test_path = "/home/filip141/Datasets/cifar/test"
    cifar_train = CIFARDataset(data_path=cifar_train_path, resolution="224x224", force_overfit=False)
    cifar_test = CIFARDataset(data_path=cifar_test_path, resolution="224x224", force_overfit=False)
    wrnnet = ResNet50(input_size=[224, 224, 3], output_size=10,
                      metrics=["accuracy", "cross_entropy"],
                      log_path="/home/filip141/tensor_logs/ResNet50_CIFAR")
    wrnnet.build_model()
    wrnnet.set_optimizer("Adam")
    wrnnet.set_loss("cross_entropy")
    wrnnet.model_compile(0.003, decay=0.96, decay_steps=1562)
    wrnnet.train(cifar_train, cifar_test, batch_size=32, batch_size_test=100, epochs=15, sample_per_epoch=1562,
                 summary_step=80, restore_model=False)
