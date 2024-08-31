"""
File for Gan Autoencoder class 'GanNetwork'
"""


import keras
import tensorflow as tf
import os
import json
import numpy as np


class GanNetwork(keras.Model):
    """
    Implements autoencoder + critic combination, similar to in a Generative Adversarial Network (GAN)
    For future, can replace autoencoder in Gan with actual generator that learns to generate PU from noise
    """
    # Static fields
    bce_loss = keras.losses.BinaryCrossentropy(from_logits=False)
    mse_loss = keras.losses.MeanSquaredError()

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self._autoencoder = GanNetwork.build_autoencoder(input_dim, latent_dim)
        self._discriminator = GanNetwork.build_discriminator(input_dim)
        self._d_optimizer = None
        self._a_optimizer = None
        # self.seed_generator = keras.random.SeedGenerator(1337)
        self._aut_loss_tracker = keras.metrics.Mean(name="autoencoder_loss")
        self._disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self._aut_loss_tracker, self._disc_loss_tracker]

    @staticmethod
    def build_autoencoder(input_dim, latent_dim):
        autoencoder = keras.Sequential(
            [
                keras.layers.InputLayer((input_dim,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(latent_dim, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(input_dim, activation="relu"),
            ],
            name="autoencoder",
        )
        return autoencoder

    @staticmethod
    def build_discriminator(input_dim):
        discriminator = keras.Sequential(
            [
                keras.layers.InputLayer((input_dim,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        return discriminator

    @staticmethod
    def discriminator_loss(real_outputs, fake_outputs):
        real_loss = GanNetwork.bce_loss(tf.ones_like(real_outputs), real_outputs)
        fake_loss = GanNetwork.bce_loss(tf.zeros_like(fake_outputs), fake_outputs)
        total_loss = (real_loss + fake_loss) * 0.5
        return total_loss

    @staticmethod
    def autoencoder_loss(inputs, reconstructions, fake_outputs):
        reco_loss = GanNetwork.mse_loss(inputs, reconstructions)
        critic_loss = GanNetwork.bce_loss(tf.ones_like(fake_outputs), fake_outputs)
        alpha = 0.5
        total_loss = alpha * reco_loss + (1 - alpha) * critic_loss
        return total_loss

    def compile(self, d_optimizer, a_optimizer):
        super().compile()
        self._d_optimizer = d_optimizer
        self._a_optimizer = a_optimizer

    def train_step(self, data):
        X_data, _trash = data  # throw away y_data, which is a duplicate of X_data as we are training an autoencoder

        # Train the discriminator
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            reconstructions = self._autoencoder(X_data, training=True)

            real_output = self._discriminator(X_data, training=True)
            fake_output = self._discriminator(reconstructions, training=True)

            gen_loss = self.autoencoder_loss(X_data, reconstructions, fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # Calculate gradients
        auto_grad = gen_tape.gradient(gen_loss, self._autoencoder.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, self._discriminator.trainable_variables)

        # Apply gradients
        self._a_optimizer.apply_gradients(zip(auto_grad, self._autoencoder.trainable_variables))
        self._d_optimizer.apply_gradients(zip(disc_grad, self._discriminator.trainable_variables))

        # Update metrics
        self._aut_loss_tracker.update_state(gen_loss)
        self._disc_loss_tracker.update_state(disc_loss)

        return {
            "autoencoder_loss": self._aut_loss_tracker.result(),
            "discriminator_loss": self._disc_loss_tracker.result(),
        }

    def get_reconstruction_errors(self, input_data):
        """
        Calculates and returns the reconstruction errors of the autoencoder for given input data.

        :param input_data: Input data to be reconstructed
        :return: Reconstruction errors as a numpy array
        """
        reconstructions = self._autoencoder.predict(input_data)
        reconstruction_errors = np.sum((input_data - reconstructions) ** 2, axis=1)
        return reconstruction_errors

    def get_discriminator_predictions(self, input_data):
        """
        Calculates and returns the discriminator predictions for given input data.

        :param input_data: Input data to be evaluated by the discriminator
        :return: Discriminator predictions as a numpy array
        """
        return self._discriminator.predict(input_data)

    def save_model(self, filepath):
        """
        Save the GanNetwork model components to a directory.

        :param filepath: Directory path to save the model components
        """
        os.makedirs(filepath, exist_ok=True)

        # Save autoencoder
        self._autoencoder.save(os.path.join(filepath, 'autoencoder.keras'))

        # Save discriminator
        self._discriminator.save(os.path.join(filepath, 'discriminator.keras'))

        # Save optimizer configurations
        with open(os.path.join(filepath, 'optimizer_config.json'), 'w') as f:
            json.dump({
                'd_optimizer': {
                    'class_name': self._d_optimizer.__class__.__name__,
                    'config': self._d_optimizer.get_config()
                },
                'a_optimizer': {
                    'class_name': self._a_optimizer.__class__.__name__,
                    'config': self._a_optimizer.get_config()
                }
            }, f)

        print(f"Model components saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved GanNetwork model from a directory.

        :param filepath: Directory path to the saved model components
        :return: Loaded GanNetwork instance
        """

        # Load autoencoder
        autoencoder = keras.models.load_model(filepath + "/autoencoder.keras")

        # Load discriminator
        discriminator = keras.models.load_model(filepath + '/discriminator.keras')

        # Create a new instance
        input_dim = autoencoder.input_shape[1]
        latent_dim = autoencoder.layers[2].output.shape[1]  # Assuming the latent layer is the 3rd layer
        gan = cls(input_dim, latent_dim)

        # Replace the components
        gan._autoencoder = autoencoder
        gan._discriminator = discriminator

        # Load and set optimizer configurations
        with open(filepath + '/optimizer_config.json', 'r') as f:
            optimizer_config = json.load(f)

        d_optimizer = keras.optimizers.get(optimizer_config['d_optimizer'])
        a_optimizer = keras.optimizers.get(optimizer_config['a_optimizer'])
        gan.compile(d_optimizer, a_optimizer)

        print(f"Model components loaded from {filepath}")
        return gan