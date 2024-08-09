"""
For custom callbacks
"""


import tensorflow as tf
import numpy as np
import keras


class ActivationLogger(keras.callbacks.Callback):
    """
    Logs the activations of a neural network during training / inference, if needed
    """
    def __init__(self, layers_to_log, data, to_print=True):
        super().__init__()
        # super().set_model(model)
        self._layers_to_log = layers_to_log
        self._activations = {}
        self._data = data
        self._to_print = to_print

    def on_epoch_end(self, epoch, logs=None):
        for layer_name in self._layers_to_log:
            x, _ = self._data  # Unpack the data (x_train, y_train)
            intermediate_layer_model = keras.models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
            intermediate_output = intermediate_layer_model.predict(x)
            self._activations[layer_name] = intermediate_output
            if self._to_print:
                print(f'Activations from layer {layer_name} after epoch {epoch+1}: {intermediate_output}')

    def on_predict_batch_end(self, batch, logs=None):
        for layer_name in self._layers_to_log:
            intermediate_layer_model = keras.models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
            intermediate_output = intermediate_layer_model.predict(self._data)
            if layer_name not in self._activations:
                self._activations[layer_name] = intermediate_output
            else:
                self._activations[layer_name] = np.vstack((self._activations[layer_name], intermediate_output))
            print(f'Activations from layer {layer_name}: {intermediate_output.shape}')

    """
    Getter, Setter, and Helper functions
    """
    def set_data(self, data):
        self._data = data

    def erase_activations_cache(self):
        self._activations = {}

    def get_activations(self):
        return self._activations


if __name__ == "__main__":
    """
    Demo / Testing code for module
    """
    # Example usage:
    # Define your model
    input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
    conv_layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    flatten_layer = tf.keras.layers.Flatten()(conv_layer)
    output_layer = tf.keras.layers.Dense(10, activation='softmax')(flatten_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generate Random Data
    x_train = np.random.rand(20, 28, 28, 1)
    y_train = np.random.randint(0, 10, 20)
    selected_idxs = np.random.choice(20, 5, replace=False)
    x_test = x_train[selected_idxs]
    y_test = y_train[selected_idxs]

    # Instantiate the custom callback
    activation_logger = ActivationLogger(layers_to_log=['conv2d', 'flatten'], data=(x_train, y_train))

    # Train the model with the custom callback
    model.fit(x_train, y_train, epochs=5, callbacks=[activation_logger])

    # Get and store activations during inference; but before that, set data and clear activation logger
    activation_logger.set_data(x_test)
    model.predict(x_test, callbacks=[activation_logger])
    activations = activation_logger.get_activations()

    # Access stored activations
    conv_layer_activation = activations['conv2d']
    flatten_layer_activation = activations['flatten']

    # Print activations
    print(conv_layer_activation)
    print(flatten_layer_activation)
