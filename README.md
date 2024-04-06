# Testing ANN and CNN on CIFAR-10 Dataset

This repository contains Jupyter Notebook files that demonstrate the implementation and evaluation of Artificial Neural Network (ANN) and Convolutional Neural Network (CNN) models on the CIFAR-10 dataset.

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images. The goal is to correctly classify these images into their respective categories. We approach this problem by implementing and comparing two models: ANN and CNN.

## Models

### Artificial Neural Network (ANN)

The ANN model is a simple feedforward neural network with the following architecture:

- Input layer with flattening
- Dense layer with 3000 neurons and ReLU activation
- Dense layer with 1000 neurons and ReLU activation
- Output layer with 10 neurons and sigmoid activation

The ANN model is compiled with the SGD optimizer and sparse categorical crossentropy loss function.

### Convolutional Neural Network (CNN)

The CNN model includes convolutional layers and max pooling layers, followed by dense layers:

- Conv2D layer with 32 filters and ReLU activation
- MaxPooling2D layer
- Conv2D layer with 64 filters and ReLU activation
- MaxPooling2D layer
- Flatten layer
- Dense layer with 64 neurons and ReLU activation
- Output layer with 10 neurons and softmax activation

The CNN model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.

## Training

Both models are trained on the CIFAR-10 training dataset. The ANN model is trained for 5 epochs, while the CNN model is trained for 10 epochs.

## Results

After training, the models achieved the following results:

- ANN: Accuracy of 48% and F1 score of 0.47 after 5 epochs
- CNN: Accuracy of 92% and F1 score of 0.71 after 10 epochs

## Conclusion

The CNN model significantly outperforms the ANN model, showcasing the effectiveness of convolutional layers in image classification tasks.

