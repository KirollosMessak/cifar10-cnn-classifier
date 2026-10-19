"""Train the CNN on CIFAR-10.

Usage:
    python src/train.py --epochs 50 --batch-size 64
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from data import load_data
from model import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "cifar10_cnn.keras")
RESULTS_DIR = os.path.join(ROOT, "results")


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (x_train, y_train), (x_val, y_val), _ = load_data()
    print(f"Train: {x_train.shape}, Validation: {x_val.shape}")

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    plot_history(history, os.path.join(RESULTS_DIR, "training_curves.png"))
    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    print(f"Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
