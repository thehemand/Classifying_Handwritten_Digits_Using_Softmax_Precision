# Classifying_Handwritten_Digits_Using_Softmax_Precision

This project from the course Introduction to Computer Vision and Image Processing by IBM implements a Softmax classifier using PyTorch to classify handwritten digits from the MNIST dataset. The model is trained on the dataset and analyzed based on its accuracy and loss. Visualizations of model parameters and predictions are also included.

# Objectives

- Download and preprocess the MNIST dataset.
- Build a Softmax classifier in PyTorch.
- Create the criterion, optimizer, and data loaders for training and validation.
- Train the model and evaluate its performance.
- Visualize model parameters and analyze prediction probabilities for both correct and misclassified samples.

# Installation
To run this project, ensure you have the following Python packages installed:
- pip install torch torchvision numpy matplotlib

# Required Libraries
- torch: PyTorch framework for building and training neural networks.
- torchvision: For downloading and transforming the MNIST dataset.
- numpy: For numerical operations and data manipulation.
- matplotlib: For visualizing model parameters and results.

# Project Overview
The classifier predicts the digits (0-9) based on the pixel values in the 28x28 images. The model uses the Softmax function to output probabilities for each class, and the class with the highest probability is chosen as the predicted digit.

# Key components include:

- Model Architecture: A single-layer Softmax classifier with input size of 28x28 (flattened image) and output size of 10 (digits).
- Training: Cross-entropy loss is used, and the model is optimized with Stochastic Gradient Descent (SGD).
- Evaluation: Accuracy and loss are tracked over epochs, with visualizations for the parameters and misclassified samples.

# Visualization
The parameters of the Softmax function are visualized to understand the model's learned features.
Correctly and incorrectly classified samples are plotted, along with their prediction probabilities.

# Results
The model achieves consistent accuracy on the MNIST dataset, with higher prediction probabilities for correctly classified samples compared to misclassified ones.
