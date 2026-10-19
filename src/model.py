"""CNN architecture for CIFAR-10 image classification."""
from tensorflow import keras
from tensorflow.keras import layers

from data import augmentation_layer


def conv_block(x, filters: int, dropout: float):
    """Two 3x3 convolutions with BatchNorm, then max-pooling and dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_model(input_shape=(32, 32, 3), num_classes: int = 10) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = augmentation_layer()(inputs)

    x = conv_block(x, 32, 0.2)   # 32x32 -> 16x16
    x = conv_block(x, 64, 0.3)   # 16x16 -> 8x8
    x = conv_block(x, 128, 0.4)  # 8x8   -> 4x4

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cifar10_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_model().summary()
