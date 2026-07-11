# MNIST CNN From Scratch

This project trains a small convolutional neural network from scratch using NumPy, pandas, and SciPy. It reads local MNIST CSV files, keeps only digits `0` and `1`, trains a binary classifier, and reports test accuracy.

## How It Works

The data loader reads:

```text
data/mnist_train.csv
data/mnist_test.csv
```

Each CSV row is expected to contain:

```text
label, pixel_0, pixel_1, ..., pixel_783
```

By default, `main.py` uses 1000 training samples per class and up to 1000 test samples per class.

The model is:

```text
1x28x28 input
-> 5 convolution kernels
-> ReLU
-> flatten
-> dense 100
-> ReLU
-> dense 2
-> softmax
```

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate:

```bash
python main.py
```

## Project Structure

```text
data/                 Local MNIST CSV files
main.py               CNN MNIST training entry point
requirements.txt      Python dependencies
src/dataset.py        pandas CSV loading and binary preprocessing
src/model.py          CNN architecture
src/train.py          Predict, train, and accuracy helpers
src/layer.py          Base layer interface
src/convolutional.py  Convolution layer
src/flatten.py        Flatten conv features for dense layers
src/dense.py          Dense layer
src/activations.py    ReLU and softmax activations
src/losses.py         Categorical cross-entropy
```
