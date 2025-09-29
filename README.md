# EmergingTech_Vehicles

🚀 Vehicle Image Classification using CIFAR-100 and CNN
📌 Overview

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) using TensorFlow and TensorFlow Datasets (TFDS).
We filter the CIFAR-100 dataset to focus only on specific vehicle classes, preprocess the data, train a deep CNN model, and visualize its performance.

📂 Workflow
Step 1: Load and Filter the Dataset

Load CIFAR-100 dataset from tensorflow_datasets.

Focus only on 5 vehicle categories:

8 → bicycle

13 → bus

48 → motorcycle

58 → pickup truck

90 → train

Filter dataset to keep only these classes.

Step 2: Data Preprocessing and Pipeline

Map original CIFAR-100 labels to new labels (0–4).

Normalize images to [0, 1].

Build efficient TensorFlow pipelines:

Shuffle training data.

Batch size = 32.

Prefetch for performance.

Step 3: CNN Model Architecture

3 Convolutional Blocks with filters 32 → 64 → 128.

MaxPooling layers for downsampling.

Flatten layer to prepare for dense layers.

Dense layer with 128 neurons (ReLU).

Dropout (50%) to prevent overfitting.

Output layer with 5 neurons (softmax).

✅ Compiled with:

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Metric: Accuracy

Step 4: Training

Train for 30 epochs (extended for better performance).

Track both training and validation accuracy/loss.

Step 5: Evaluation

Evaluate the trained model on test data.

Report final test accuracy and loss.

Step 6: Visualization

Plot Training vs. Validation Accuracy.

Plot Training vs. Validation Loss.

📊 These plots help diagnose overfitting/underfitting.

✅ Key Features

CIFAR-100 subset classification (vehicles only).

Clean and optimized TensorFlow data pipeline.

Deeper CNN with dropout regularization.

Visual performance tracking with Matplotlib.

🔧 Requirements

Python 3.8+

TensorFlow 2.x

TensorFlow Datasets

Matplotlib

NumPy

Install dependencies:

pip install tensorflow tensorflow-datasets matplotlib numpy

🚀 How to Run

Clone the repository and run:

python vehicle_classification.py

📊 Example Output

Final Test Accuracy (e.g., ~85% depending on training).

Training & Validation Accuracy/Loss plots.
