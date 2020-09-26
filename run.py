import torch.nn.functional as F
from torch import optim
import mnist_cnn
from mnist_cnn.dataload import *
from mnist_cnn.model import CNN


batch_size = 32
epochs = 10
model = CNN(1)
loss_func = F.cross_entropy
opt = optim.Adam(
    model.parameters(),
    lr=0.001
)


def main():
	x_train, y_train, x_valid, y_valid = mnist_cnn.read()
	train_dl = WrappedDataLoader(x_train, y_train, x_transform, batch_size=batch_size, shuffle=True)
	valid_dl = WrappedDataLoader(x_valid, y_valid, x_transform, batch_size=batch_size * 2)

	mnist_cnn.fit(train_dl, valid_dl, model, loss_func, opt, epochs)


if __name__ == "__main__":
	main()
