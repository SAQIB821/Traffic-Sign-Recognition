"""
Traffic Sign Recognition Package
==================================

A CNN-based traffic sign recognition system using TensorFlow/Keras.

This package contains modules for training and testing a deep learning
model to classify traffic signs from the German Traffic Sign Recognition
Benchmark (GTSRB) dataset.

Modules:
--------
    traffic_sign_recognition : Main module for model training and testing
        Contains functions for:
        - Loading and preprocessing images
        - Building CNN architecture
        - Training the model
        - Evaluating performance
        - Generating visualizations

Features:
---------
    - Multi-threaded image loading for fast data processing
    - CNN with 2 convolutional blocks
    - Dropout regularization to prevent overfitting
    - Comprehensive visualization (accuracy, loss, confusion matrix)
    - Support for both Google Colab and local environments

Usage:
------
    from src import traffic_sign_recognition
    
    # Or run directly:
    python src/traffic_sign_recognition.py

Requirements:
-------------
    See requirements.txt for full list of dependencies

Author: Your Name
Email: your.email@example.com
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Package metadata
__all__ = [
    "traffic_sign_recognition",
]

# Package information
PACKAGE_NAME = "Traffic Sign Recognition"
DESCRIPTION = "CNN-based traffic sign classification system"
DATASET = "German Traffic Sign Recognition Benchmark (GTSRB)"
NUM_CLASSES = 43
IMAGE_SIZE = (30, 30, 3)

# Model architecture info
MODEL_INFO = {
    "architecture": "CNN",
    "layers": [
        "Conv2D (32 filters, 5x5)",
        "Conv2D (32 filters, 5x5)",
        "MaxPooling2D (2x2)",
        "Dropout (0.25)",
        "Conv2D (64 filters, 3x3)",
        "Conv2D (64 filters, 3x3)",
        "MaxPooling2D (2x2)",
        "Dropout (0.25)",
        "Flatten",
        "Dense (256 units)",
        "Dropout (0.5)",
        "Dense (43 units, softmax)"
    ],
    "optimizer": "Adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"]
}

def get_version():
    """Return the version of the package."""
    return __version__

def get_model_info():
    """Return information about the model architecture."""
    return MODEL_INFO

def print_info():
    """Print package information."""
    print(f"{PACKAGE_NAME} v{__version__}")
    print(f"Description: {DESCRIPTION}")
    print(f"Dataset: {DATASET}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")