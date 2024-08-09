"""
Fully implements a transformer encoder layer
"""


import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, Add, Dropout, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np


class TransformerEncoderLayer(Layer):
    def __init__(self, head_size, num_heads, ff_dim, ff_regularizer=None, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self._attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self._feed_forward = tf.keras.Sequential([
            Dense(ff_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_regularizer=ff_regularizer),
            Dense(head_size),
        ])
        # print("Has layer normalization:", head_size > 2)
        self._layernorm1 = LayerNormalization(epsilon=1e-6) if head_size > 2 else tf.keras.layers.Identity()
        self._layernorm2 = LayerNormalization(epsilon=1e-6) if head_size > 2 else tf.keras.layers.Identity()

    def call(self, inputs):
        attn_output = self._attention(inputs, inputs)
        out1 = self._layernorm1(inputs + attn_output)
        ff_output = self._feed_forward(out1)
        return self._layernorm2(out1 + ff_output)


def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, regularizer=None):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        x = TransformerEncoderLayer(head_size, num_heads, ff_dim, ff_regularizer=regularizer)(x)
    x = GlobalAveragePooling1D()(x)  # Average the outputs of the encoder layer
    outputs = Dense(1, activation="sigmoid", kernel_regularizer=regularizer)(x)  # Logistic regression
    return Model(inputs, outputs)


if __name__ == "__main__":
    """
    Demo / Testing code for module
    Testing done with random data
    """
    input_shape = (100, 1)  # For example, sequence length of 100 and each token represented by a 2-dimensional vector
    head_size = input_shape[0]
    num_heads = 4
    ff_dim = 3
    num_layers = 1
    dropout = 0.1

    model = build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, regularizer=l2(1e-4))

    model.compile(
        loss="binary_crossentropy",  # Change this based on your task
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"]  # Change this based on your task
    )

    # Dummy data for demo
    X_train = np.random.random((1000, input_shape[0], input_shape[1]))
    y_train = np.random.randint(0, 2, size=(1000,))  # Adjust this based on your task
    idxs = np.random.randint(0, 1000, size=(100,))
    X_test = X_train[idxs]
    y_test = y_train[idxs]

    model.fit(X_train, y_train, epochs=20, batch_size=64)
    model.evaluate(X_test, y_test)

