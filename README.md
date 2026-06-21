# MNIST MLP - PyTorch

A multilayer perceptron (MLP) for classifying handwritten digits from the MNIST dataset, built step by step with PyTorch: from a single neuron up to a full training pipeline with validation, checkpoints, dropout, batch normalization, gradient clipping, and a learning rate scheduler.

## What's inside

- Custom `MLP` model: 784 - 128 - 64 - 10, with BatchNorm and Dropout on each hidden layer
- Custom `Dataset` and `DataLoader` for batching and shuffling
- Train / validation split, with per-epoch train and validation accuracy
- Checkpointing (resume training after a crash or interruption, including optimizer and scheduler state)
- Gradient clipping and a `StepLR` learning rate scheduler
- Final evaluation on a held-out test set

## Results

~97% accuracy on the held-out test set after 10 epochs.

## Getting the data

This project uses the MNIST dataset in CSV format from Kaggle:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Download `mnist_train.csv` and `mnist_test.csv` (or their `.zip` versions) and place them wherever the notebook expects them (`/content/` if running in Google Colab).

## Running it

1. Open `MNIST_MLP_full_pipeline.ipynb` in Google Colab or Jupyter.
2. Upload the MNIST CSV files to the same environment.
3. Run all cells, in order.

## Requirements

- torch
- pandas
