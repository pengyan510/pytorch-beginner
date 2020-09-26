# pytorch-beginner

Implement a toy CNN model on the MNIST dataset. The purpose is to arrange code in a structured way.

## Files
	- data/mnist contains the MNIST dataset
	- mnist_cnn is the module
	- mnist_cnn/config.py specifies the data path
	- mnist_cnn/dataload.py transfomrs the data from numpy array to DataLoader
	- mnist_cnn/fit.py fits the model
	- mnist_cnn/model.py stores the CNN model
	- mnist_cnn/read.py reads in the data

## Setup
To run, use

```
python run.py
```

User can specify the following attributes:
	* batch_size
	* epoch
	* models
	* loss function
	* optimizer
