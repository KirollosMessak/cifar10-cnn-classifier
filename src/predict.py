"""Predict the class of your own image.

Usage:
    python src/predict.py path/to/image.jpg
"""
import argparse
import os

import numpy as np
from PIL import Image
from tensorflow import keras

from data import CLASS_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "cifar10_cnn.keras")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--top", type=int, default=3, help="How many top classes to show")
    args = parser.parse_args()

    model = keras.models.load_model(MODEL_PATH)
    img = Image.open(args.image).convert("RGB").resize((32, 32))
    x = np.asarray(img, dtype="float32")[None] / 255.0

    probs = model.predict(x, verbose=0)[0]
    for i in np.argsort(probs)[::-1][: args.top]:
        print(f"{CLASS_NAMES[i]:<12} {probs[i] * 100:5.1f}%")


if __name__ == "__main__":
    main()
