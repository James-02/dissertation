import numpy as np

DATA_PATH = 'data/'

def load_mnist():
    data = np.load(DATA_PATH + 'mnist.npz')
    # Split out testing and training sets into inputs and targets
    return data['x_train'][:5000], data['y_train'][:5000], data['x_test'][:5000], data['y_test'][:5000]
