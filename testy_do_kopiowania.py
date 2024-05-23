import tensorflow as tf
import tensorflow_datasets as tfds

class KerasNNModel:
    def __init(self, hidden_layer_size, activation_fn):
        model = tf.keras.Sequental([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(hidden_layer_size, activation=activation_fn),
            tf.keras.layers.Dense(10)
        ])
        self.model = model

    def get_keras_nn_model(self):
        return self.model
    
    def tran_model(self, train_images, train_labels, optimizen_type, metric_list, num_epochs):
        self.model.compile(optimizer=optimizen_type,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=metric_list)
        self.model.fit(train_images, train_labels, epochs=num_epochs)

    def evaluate_model(self, test_images, test_labels):
        metrics_dict = self.model.evaluate(test_images, test_labels)
        return metrics_dict

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
