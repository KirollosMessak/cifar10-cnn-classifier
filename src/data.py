"""Data loading and preprocessing for CIFAR-10."""
import numpy as np
from tensorflow import keras

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_data(val_split: float = 0.1, seed: int = 42):
    """Load CIFAR-10, scale pixels to [0, 1] and split off a validation set.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train))
    n_val = int(len(x_train) * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    return (
        (x_train[train_idx], y_train[train_idx]),
        (x_train[val_idx], y_train[val_idx]),
        (x_test, y_test),
    )


def augmentation_layer():
    """Light data augmentation applied only during training."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomRotation(0.05),
        ],
        name="augmentation",
    )
