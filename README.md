# MNIST PyTorch Model

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) to classify images from the MNIST dataset. The model is designed to have less than 25,000 parameters, and includes techniques like Batch Normalization and Dropout for improved training and generalization.

## Project Structure

```
mnist_pytorch/
│
├── mnist_model/
│   ├── __init__.py
│   ├── model.py              # Model definition with CNN, BatchNorm, and Dropout
│
├── training/
│   ├── __init__.py
│   ├── train.py              # Training script
│
├── data/                     # MNIST data (automatically downloaded)
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Test script for model verification
│
├── .github/
│   └── workflows/
│       └── test_model.yml    # GitHub Actions workflow for automated testing
│
├── requirements.txt          # Required Python packages
├── README.md                 # Project documentation
└── main.py                   # Entry point for training the model
```

## Features
- **Model Size**: The model is designed to have less than 25,000 parameters.
- **Batch Normalization**: Added after each convolutional layer to improve training stability.
- **Dropout**: Added before the fully connected layer to reduce overfitting.
- **Training & Validation**: The script trains the model and evaluates its performance on the validation set, printing both accuracies.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/mnist_pytorch.git
   cd mnist_pytorch
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Training Script

To train the model and evaluate it, run:
```sh
python main.py
```
This script will:
- Train the model for one epoch.
- Print the model size, training accuracy, and validation accuracy.
- Save the trained model to `mnist_model.pth`.

## Running Tests

To verify the model has fewer than 25,000 parameters, run the test script:
```sh
python tests/test_model.py
```

Or you can run all tests using `pytest`:
```sh
pytest tests/
```

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow for continuous integration. The workflow tests:
- The model has fewer than 25,000 parameters.
- The training process runs without errors.

The workflow can be found at `.github/workflows/test_model.yml`.

## Requirements
- Python 3.8+
- PyTorch 2.0.0
- torchvision 0.15.0
- numpy

Install all dependencies via `pip`:
```sh
pip install -r requirements.txt
```

## Model Overview

The model is a compact CNN with the following architecture:
- **Conv2D Layer 1**: 16 filters, 3x3 kernel, followed by Batch Normalization and ReLU activation.
- **MaxPooling Layer**: 2x2 pooling.
- **Conv2D Layer 2**: 32 filters, 3x3 kernel, followed by Batch Normalization and ReLU activation.
- **MaxPooling Layer**: 2x2 pooling.
- **Fully Connected Layer 1**: 64 units, ReLU activation, followed by Dropout.
- **Fully Connected Layer 2**: 10 units (output).

## License

This project is licensed under the MIT License.

## Author

- **Your Name** - [GitHub Profile](https://github.com/yourusername)

Feel free to contact me if you have any questions or suggestions!

