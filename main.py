import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Encoder(layers.Layer):
    def __init__(self, latent_dim, filters=128, num_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.filters = filters
        self.num_layers = num_layers

        self.layers = [
            layers.Conv2D(
                filters=filters if i == 0 else filters * i,
                kernel_size=3,
                strides=2,
                activation="relu",
                padding="same",
            )
            for i in range(num_layers)
        ]
        self.flatten = layers.Flatten()
        self.latent_proj = layers.Dense(2 * latent_dim)

    def call(self, inputs, one_hot_labels, training=False):
        """Forward Pass."""
        input_shape = tf.shape(inputs)
        one_hot_labels = one_hot_labels[:, tf.newaxis, tf.newaxis, :]
        one_hot_labels = tf.tile(one_hot_labels, (1, input_shape[1], input_shape[2], 1))
        out = tf.concat([inputs, one_hot_labels], axis=-1)

        for layer in self.layers:
            out = layer(out)

        out = self.flatten(out)
        out = self.latent_proj(out)

        mean, logvar = tf.split(out, num_or_size_splits=2, axis=1)
        eps = tf.random.normal((tf.shape(mean)))
        z = eps * tf.exp(logvar * 0.5) + mean
        return z, mean, logvar


class Decoder(layers.Layer):
    def __init__(self, latent_dim, filters=32, num_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.filters = filters
        self.num_layers = num_layers

        self.latent_proj = layers.Dense(7 * 7 * 64, activation="relu")
        self.layers = [
            layers.Conv2DTranspose(
                filters=filters if i == 0 else filters * i,
                kernel_size=3,
                strides=2,
                activation="relu",
                padding="same",
            )
            for i in reversed(range(num_layers))
        ]
        self.proj = layers.Conv2DTranspose(
            1,
            kernel_size=3,
            activation=None,
            padding="same",
        )

    def call(self, inputs, one_hot_labels):
        """Forward Pass."""
        out = tf.concat([inputs, one_hot_labels], axis=-1)
        out = self.latent_proj(out)
        out = tf.reshape(out, (tf.shape(out)[0], 7, 7, 64))

        for layer in self.layers:
            out = layer(out)

        out = self.proj(out)
        return out


class ConditionalVAE(tf.keras.Model):
    """Conditional Variational Autoencoder."""

    def __init__(self, num_labels, latent_dim=16, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.encoder = Encoder(latent_dim=latent_dim, num_layers=num_layers)
        self.decoder = Decoder(latent_dim=latent_dim, num_layers=num_layers)

    def call(self, inputs, training=False):
        """Forward Pass."""
        images, labels = inputs
        one_hot_labels = tf.one_hot(labels, self.num_labels)
        z, mean, logvar = self.encoder(images, one_hot_labels, training=training)
        logits = self.decoder(z, one_hot_labels, training=training)
        return mean, logvar, logits

    @tf.function(input_signature=[tf.TensorSpec((None,), tf.int32)])
    def sample(self, labels, eps=None):
        """Sample from (z|labels)"""
        one_hot_labels = tf.one_hot(labels, self.num_labels)

        if eps is None:
            eps = tf.random.normal(shape=(tf.shape(labels)[0], self.latent_size))

        return self.decoder(eps, one_hot_labels)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.total_loss_tracker = tf.keras.metrics.Mean("total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean("reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean("kl_loss")

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            mean, logvar, logits = self([images, labels])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(images, tf.nn.sigmoid(logits)),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        images, labels = data
        mean, logvar, logits = self([images, labels], training=False)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(images, tf.nn.sigmoid(logits)),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def main():
    """Main"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255
    x_train = np.concatenate([x_train, x_test], 0)
    y_train = np.concatenate([y_train, y_test], 0).astype(np.int32)

    print(x_train.shape, y_train.shape)

    model = ConditionalVAE(num_labels=10)
    model.compile(optimizer=tf.optimizers.Adam(1e-4))
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=30,
        verbose=2,
        validation_split=0.2,
    )


if __name__ == "__main__":
    main()
