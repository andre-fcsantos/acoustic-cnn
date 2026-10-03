"""Original v2 convolutional architecture and weighted loss."""
import numpy as np


def build_model(config):
    import tensorflow as tf
    layers = tf.keras.layers
    regularizer = tf.keras.regularizers.l2(config.l2)
    inputs = tf.keras.Input(config.input_shape, name="mel_db")
    x = inputs
    for filters, pool in ((16, True), (32, True), (64, False)):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              kernel_regularizer=regularizer)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        if pool:
            x = layers.MaxPooling2D(2)(x)
        x = layers.SpatialDropout2D(0.10)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, use_bias=False, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(len(config.classes), activation="sigmoid", name="probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="acoustic_multilabel_cnn_v2")


def positive_weights(labels):
    labels = np.asarray(labels)
    positive = labels.sum(axis=0)
    negative = len(labels) - positive
    if (positive == 0).any() or (negative == 0).any():
        raise ValueError("Training requires positive and negative samples for every class.")
    return np.sqrt(negative / positive).astype(np.float32)


def weighted_loss(weights):
    import tensorflow as tf
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        probabilities = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        return tf.reduce_mean(-(weights * y_true * tf.math.log(probabilities)
            + (1 - y_true) * tf.math.log(1 - probabilities)))
    return loss
