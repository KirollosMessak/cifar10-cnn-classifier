"""Evaluate the trained model on the CIFAR-10 test set.

Usage:
    python src/evaluate.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

from data import CLASS_NAMES, load_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "cifar10_cnn.keras")
RESULTS_DIR = os.path.join(ROOT, "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _, _, (x_test, y_test) = load_data()
    model = keras.models.load_model(MODEL_PATH)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3)
    print(report)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\nTest loss: {loss:.4f}\n\n{report}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # Grid of sample predictions (green = correct, red = wrong)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.5))
    for ax, i in zip(axes.flat, np.random.default_rng(0).choice(len(x_test), 18, replace=False)):
        ax.imshow(x_test[i])
        ok = y_pred[i] == y_test[i]
        ax.set_title(f"{CLASS_NAMES[y_pred[i]]}\n(true: {CLASS_NAMES[y_test[i]]})",
                     fontsize=8, color="green" if ok else "red")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sample_predictions.png"), dpi=120)
    plt.close(fig)

    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
