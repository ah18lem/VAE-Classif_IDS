import tensorflow as tf
import keras
from keras import layers, ops
import constantes

dynamic_kl_factor = 0.0

class Sampling(layers.Layer):
    def __init__(self, isServer=False, **kwargs):
        super().__init__(**kwargs)
        self.isServer = isServer

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))

        # KL client
        if not self.isServer:
            kl_loss = -0.5 * (
                1.0 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var)
            )
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            self.add_loss(dynamic_kl_factor * kl_loss)

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

        self.vae = vae
        self.isServer = isServer
        self.kl_target = kl_target
        self.kl_loss_factor = tf.Variable(0.0, trainable=False, dtype=tf.float32) if vae else None

        initializer = keras.initializers.GlorotUniform()
        inputs = keras.Input(shape=(input_dim,))

        # Bloc encodeur
        if vae:
            encoded = self.build_vae_encoder(inputs, encoder_layer_sizes, initializer, isServer)[0]
        else:
            encoded = self.build_encoder(inputs, encoder_layer_sizes, initializer, isServer)

        # Bloc décodeur
        decoded = self.build_decoder(encoded, decoder_layer_sizes, input_dim, initializer, isServer)

        # Sortie classe
        outputs = layers.Dense(
            constantes.NUM_CLASSES,
            activation="softmax",
            name="classification_layer",
            kernel_initializer=initializer,
            trainable=isServer,
        )(encoded)

        self.model = keras.Model(inputs=inputs, outputs=[outputs, decoded])

        # Compilation modèle
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=constantes.LEARNING_RATE),
            loss=["categorical_crossentropy", "mean_squared_error"],
            loss_weights=[1.0, 0.0] if isServer else [0.0, 1.0],
        )

    def call(self, inputs):
        return self.model(inputs)

    def get_parameters(self):
        # Poids modèle
        return self.model.get_weights()

    def set_parameters(self, parameters):
        # Mise à jour
        self.model.set_weights(parameters)

    def train(self, train_data, train_labels, epochs):
        # Entraînement local
        callbacks = []

        if self.vae:
            callbacks = [
                GradualKLLossCallback(
                    self.kl_loss_factor,
                    self.kl_target,
                    epochs,
                )
            ]

        return self.model.fit(
            train_data,
            [train_labels, train_data],
            epochs=epochs,
            batch_size=constantes.BATCH_SIZE,
            shuffle=False,
            callbacks=callbacks,
            verbose=0,
        )

    def build_encoder(self, x, sizes, initializer, isServer):
        # Encodeur simple
        for i, size in enumerate(sizes):
            x = layers.Dense(
                size,
                activation="relu",
                trainable=True,
                name=f"encoder{i + 1}",
                kernel_initializer=initializer,
            )(x)

        return x

    def build_vae_encoder(self, x, sizes, initializer, isServer):
        # Encodeur VAE
        for i, size in enumerate(sizes[:-1]):
            x = layers.Dense(
                size,
                activation="relu",
                trainable=True,
                name=f"encoder{i + 1}",
                kernel_initializer=initializer,
            )(x)

        z_mean = layers.Dense(
            sizes[-1],
            name="latent_mean",
            trainable=True,
            kernel_initializer=initializer,
        )(x)

        z_log_var = layers.Dense(
            sizes[-1],
            name="latent_log_var",
            trainable=True,
            kernel_initializer=initializer,
        )(x)

        z = Sampling(isServer=isServer)([z_mean, z_log_var])
        return z, z_mean, z_log_var

    def build_decoder(self, x, sizes, input_dim, initializer, isServer):
        # Décodeur local
        for i, size in enumerate(sizes):
            x = layers.Dense(
                size,
                activation="relu",
                trainable=(not isServer),
                name=f"decoder{i + 1}",
                kernel_initializer=initializer,
            )(x)

        return layers.Dense(
            input_dim,
            activation="relu",
            name="decoder",
            trainable=(not isServer),
            kernel_initializer=initializer,
        )(x)

class GradualKLLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, kl_loss_factor, target_kl_factor, epochs):
        super().__init__()
        self.kl_loss_factor = kl_loss_factor
        self.target_kl_factor = target_kl_factor
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        # Annealing KL
        self.kl_loss_factor = self.target_kl_factor * (epoch + 1) / self.epochs
