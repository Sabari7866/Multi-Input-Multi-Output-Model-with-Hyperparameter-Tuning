
# Multi-Input Multi-Output Neural Network with Hyperparameter Tuning

## Overview
This project demonstrates a deep learning model that accepts multiple input features and produces multiple outputs. The model is implemented using TensorFlow and Keras.

## Objective
To design and implement a neural network that takes inputs such as study hours and attendance percentage and predicts:
- Exam score
- Pass or fail result

Hyperparameter tuning is performed to determine the optimal number of neurons and learning rate.

## Technologies Used
- Python
- TensorFlow / Keras
- Keras Tuner
- NumPy

## Model Features
- Multi-input neural network architecture
- Multi-output prediction
- Hyperparameter tuning using Random Search

## How to Run
1. Install required libraries:
   pip install tensorflow keras-tuner numpy

2. Run the program:
   python multi_input_model.py

## Expected Output
The program predicts:
- Student exam score
- Probability of passing the exam

## Learning Outcome
This project demonstrates how neural networks can handle multiple inputs and outputs while optimizing model performance using hyperparameter tuning.
