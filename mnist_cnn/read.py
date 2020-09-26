import pickle
import gzip
from .config import PATH


def read():
	with gzip.open((PATH).as_posix(), "rb") as f:
		((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

	return x_train, y_train, x_valid, y_valid
