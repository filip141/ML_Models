from cnn_models.models.vgg16 import VGG16Model
from cnn_models.models.alexnet import AlexNetModel
from cnn_models.iterators.iceberg import IcebergDataset
from cnn_models.iterators.tools import ImageIterator
from simple_network.layers import Flatten, DropoutLayer, FullyConnectedLayer, ReluLayer, LinearLayer


def iceberg_vgg16_model(iceberg_db_train, iceberg_db_test):
    iceberg_model = VGG16Model(input_size=[75, 75, 3], output_size=1, log_path="/home/filip/tensor_logs",
                               metrics=["binary_accuracy", "cross_entropy_sigmoid"])
    iceberg_model.build_model(model_classifier=False, loss=False, optimizer=False)

    # Add Dense Layers
    iceberg_model.add(Flatten(name='flatten_6'))
    iceberg_model.add(DropoutLayer(percent=0.4))
    iceberg_model.add(FullyConnectedLayer([4608, 512],
                                          initializer="xavier", name='fully_connected_6_1'))
    iceberg_model.add(ReluLayer(name="relu_6_1"))
    iceberg_model.add(DropoutLayer(percent=0.4))

    iceberg_model.add(FullyConnectedLayer([512, 512],
                                          initializer="xavier", name='fully_connected_7_1'))
    iceberg_model.add(ReluLayer(name="relu_7_1"))
    iceberg_model.add(DropoutLayer(percent=0.2))

    iceberg_model.add(FullyConnectedLayer([512, 1], initializer="xavier",
                                          name='fully_connected_8_1'))
    iceberg_model.set_optimizer("SGD")
    iceberg_model.set_loss("cross_entropy", activation="sigmoid")
    iceberg_model.model_compile(learning_rate=0.0001)

    # Train model
    iceberg_model.train(iceberg_db_train, iceberg_db_test, 32, epochs=300)


def iceberg_alex_net_model(iceberg_db_train, iceberg_db_test):
    iceberg_model = AlexNetModel(input_size=[75, 75, 1], output_size=1, log_path="/home/filip/tensor_logs",
                                 metrics=["binary_accuracy", "cross_entropy_sigmoid"])
    iceberg_model.build_model(model_classifier=False, loss=False, optimizer=False)

    # Layer 6
    iceberg_model.add(Flatten(name='flatten_6'))
    iceberg_model.add(FullyConnectedLayer([256, 256],
                                          initializer="xavier", name='fully_connected_6_1'))
    iceberg_model.add(ReluLayer(name="relu_6_1"))
    iceberg_model.add(DropoutLayer(percent=0.5))

    # Layer 7
    iceberg_model.add(FullyConnectedLayer([256, 128], initializer="xavier", name='fully_connected_7_1'))
    iceberg_model.add(ReluLayer(name="relu_7_1"))
    iceberg_model.add(DropoutLayer(percent=0.5))

    # Layer 8
    iceberg_model.add(FullyConnectedLayer([128, 1], initializer="xavier",
                                          name='fully_connected_8_1'))
    iceberg_model.add(LinearLayer(name="linear_layer_8_1"))

    iceberg_model.set_optimizer("SGD")
    iceberg_model.set_loss("cross_entropy", activation="sigmoid")
    iceberg_model.model_compile(learning_rate=0.00001)

    # Train model
    iceberg_model.train(iceberg_db_train, iceberg_db_test, 32, epochs=300)

if __name__ == '__main__':
    json_data_train = "/home/filip/Datasets/Iceberg_data/train/processed/train.json"
    iceberg_db_train = ImageIterator(IcebergDataset(json_path=json_data_train, batch_out="mean_dim"), rotate=360)
    iceberg_db_test = ImageIterator(IcebergDataset(json_path=json_data_train, is_test=True, batch_out="mean_dim"))
    iceberg_alex_net_model(iceberg_db_train, iceberg_db_test)