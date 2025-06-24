import os
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers
from sklearn.utils import compute_class_weight
import constantes
# Declare a global variable for the dynamic KL loss factor
dynamic_kl_factor = 0.0  # Start from 0

# Sampling layer to include dynamic KL loss factor, only if the model is a client
class Sampling(layers.Layer):
   """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit and applies a dynamic factor to the KL loss only if it is a client."""

   def __init__(self, isServer=False, **kwargs):
       super().__init__()
       self.isServer = isServer  

   def call(self, inputs):
       z_mean, z_log_var = inputs
       batch = ops.shape(z_mean)[0]
       dim = ops.shape(z_mean)[1]
       epsilon = keras.random.normal(shape=(batch, dim))
       
       # Compute the KL divergence
       kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
       kl_loss = ops.mean(ops.sum(kl_loss, axis=1))

       # If the model is a client, add the KL loss
       if not self.isServer:
           dynamic_kl_loss = dynamic_kl_factor * kl_loss
           self.add_loss(dynamic_kl_loss)

      
       return z_mean + ops.exp(0.5 * z_log_var) * epsilon




class AutoencoderWithClassifier(tf.keras.Model):
    def __init__(
        self,
        input_dim=None,
        isServer=True,
        encoder_layer_sizes=constantes.ENCODER_LAYERS,
        decoder_layer_sizes=constantes.DECODER_LAYERS,
        vae=False,
        kl_target=1,
       
    ):
        super().__init__()
        input_layer = keras.Input(shape=(input_dim,))
  
        self.kl_target = kl_target

        if vae:
            # Initialize the KL loss factor for VAE
            self.kl_loss_factor = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        else:
            self.kl_loss_factor = None

        if vae:
            z, latent_mean, latent_log_var = self.build_vae_encoder(input_layer, encoder_layer_sizes, isServer)
            self.latent_mean = latent_mean
            self.latent_log_var = latent_log_var
            encoded = z
            encoder = keras.Model(input_layer, [latent_mean, latent_log_var, z], name="encoder")
            self.encoder = encoder
        else:
            encoded = self.build_encoder(input_layer, encoder_layer_sizes, isServer)
            encoder = keras.Model(input_layer, encoded, name="encoder")
            self.encoder = encoder

       
        x = encoded
        for i, layer_size in enumerate(decoder_layer_sizes):
            x = layers.Dense(
                layer_size,
                activation="relu",
                trainable=(not isServer),
                name=f"decoder{i + 1}",
            )(x)
           
        x = layers.Dense(
            input_dim,
            activation="relu",
            name="decoder",
            trainable=(not isServer),
        )(x)
        decoded = x

       
        classification_layer = layers.Dense(
            constantes.NUM_CLASSES,
            activation="softmax",
            name="classification_layer",
            trainable=isServer,
        )(encoded)

        loss_weights = [1.0, 0.0] if isServer else [0.0, 1.0]
        classification_model = tf.keras.models.Model(inputs=input_layer, outputs=[classification_layer, decoded])

        classification_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=["categorical_crossentropy", "mean_squared_error"],
            loss_weights=loss_weights,
        )
        self.model = classification_model
        self.vae = vae
        self.isServer = isServer

    def call(self, inputs):
        return self.model(inputs)

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def train(self, train_data, train_labels, epochs):
        if self.vae:
            kl_loss_callback = GradualKLLossCallback(self.kl_loss_factor, self.kl_target, epochs)
            callbacks = [kl_loss_callback]
        else:
            callbacks = []

        self.model.fit(
            train_data,
            [train_labels, train_data],
            epochs=epochs,
            batch_size=constantes.BATCH_SIZE,
            shuffle=False,
            callbacks=callbacks, 
        )

    def build_encoder(self, x, encoder_layer_sizes, isServer):
        for i, layer_size in enumerate(encoder_layer_sizes):
            x = layers.Dense(
                layer_size,
                activation="relu",
                trainable=True,
                name=f"encoder{i + 1}",
            )(x)
          
        return x

    def build_vae_encoder(self, x, encoder_layer_sizes,isServer):
        for i, layer_size in enumerate(encoder_layer_sizes[:-1]):
            x = layers.Dense(
                layer_size,
                activation="relu",
                trainable=True,
                name=f"encoder{i + 1}",
            )(x)
        latent_mean = layers.Dense(encoder_layer_sizes[-1], name="latent_mean", trainable=True, kernel_initializer=initialiser)(x)
        latent_log_var = layers.Dense(encoder_layer_sizes[-1], name="latent_log_var", trainable=True, kernel_initializer=initialiser)(x)
        z = Sampling(isServer=isServer)([latent_mean, latent_log_var])
        return z, latent_mean, latent_log_var



class GradualKLLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, kl_loss_factor, target_kl_factor, epochs):
        self.kl_loss_factor = kl_loss_factor
        self.target_kl_factor = target_kl_factor
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        new_kl_factor = self.target_kl_factor * (epoch + 1) / self.epochs
        self.kl_loss_factor= new_kl_factor
        print(f"Epoch {epoch + 1}: KL loss factor updated to {self.kl_loss_factor}")

