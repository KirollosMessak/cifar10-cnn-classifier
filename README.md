# CIFAR-10 Image Classification with a CNN

A Convolutional Neural Network (CNN) built with TensorFlow/Keras that classifies small 32×32 color images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

## Project overview

| | |
|---|---|
| **Task** | Multi-class image classification |
| **Dataset** | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 images (50,000 train / 10,000 test), 10 classes |
| **Model** | Custom CNN: 3 convolutional blocks + global average pooling + dense classifier |
| **Framework** | TensorFlow 2 / Keras |
| **Expected test accuracy** | ~85% (varies slightly per run) |

## Model architecture

```
Input (32×32×3)
  └─ Data augmentation (random flip, shift, rotation) — training only
  └─ Conv block 1: [Conv 3×3 (32) → BatchNorm → ReLU] ×2 → MaxPool → Dropout 0.2
  └─ Conv block 2: [Conv 3×3 (64) → BatchNorm → ReLU] ×2 → MaxPool → Dropout 0.3
  └─ Conv block 3: [Conv 3×3 (128) → BatchNorm → ReLU] ×2 → MaxPool → Dropout 0.4
  └─ Global Average Pooling
  └─ Dense (128, ReLU) → Dropout 0.5
  └─ Dense (10, Softmax)
```

**Why these choices**

- **Stacked 3×3 convolutions** learn edges first, then textures, then object parts, while keeping the parameter count low.
- **Batch normalization** stabilizes training and lets the model learn faster.
- **Dropout and data augmentation** reduce overfitting, which is the main problem on a small dataset like CIFAR-10.
- **Global average pooling** replaces a large flatten + dense layer, cutting parameters sharply.

**Training setup:** Adam optimizer (lr = 0.001), sparse categorical cross-entropy loss, batch size 64, up to 50 epochs with early stopping (patience 10) and learning-rate reduction on plateau. 10% of the training set is held out for validation.

## Project structure

```
cifar10-cnn-classifier/
├── src/
│   ├── data.py        # Loading, normalization, train/val split, augmentation
│   ├── model.py       # CNN architecture
│   ├── train.py       # Training loop, callbacks, training curves
│   ├── evaluate.py    # Test accuracy, classification report, confusion matrix
│   └── predict.py     # Classify your own image
├── models/            # Saved model (created by train.py, ignored by git)
├── results/           # Plots and reports
├── requirements.txt
└── README.md
```

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Train (the dataset downloads automatically on first run)
python src/train.py --epochs 50

# 4. Evaluate on the test set
python src/evaluate.py

# 5. Predict your own image
python src/predict.py path/to/your_image.jpg
```

> No GPU? Training on CPU takes roughly 1–2 hours. You can also run it free on [Google Colab](https://colab.research.google.com/) with a GPU runtime in about 10 minutes.

## Results

After training and evaluation, these files appear in `results/`:

| File | What it shows |
|---|---|
| `training_curves.png` | Accuracy and loss for training vs. validation per epoch |
| `confusion_matrix.png` | Which classes the model confuses with each other |
| `sample_predictions.png` | Example test images with predicted vs. true labels |
| `classification_report.txt` | Precision, recall and F1 score per class |

<!-- After you train, replace this section with your real numbers and images:

**Test accuracy: XX.X%**

![Training curves](results/training_curves.png)
![Confusion matrix](results/confusion_matrix.png)
![Sample predictions](results/sample_predictions.png)
-->

### Observations

- Vehicles (automobile, ship, truck, airplane) are usually classified most accurately because their shapes are distinctive.
- The most frequent confusion is between **cat and dog**, which share similar textures and poses at 32×32 resolution.

## Possible improvements

- Use **transfer learning** with a pretrained network (e.g. ResNet50 or EfficientNet) to push accuracy above 93%.
- Try stronger augmentation such as Cutout or MixUp.
- Add a simple web demo with Streamlit or Gradio.

## License

MIT
