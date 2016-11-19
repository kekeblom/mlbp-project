import random
import numpy as np
import theano

class Dataset(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def shuffle(self):
        mask = np.random.choice(np.arange(len(self.x)), len(self.x), replace=False)
        return Dataset(self.x[mask], self.y[mask])

    def sample(self, n):
        mask = np.random.choice(np.arange(len(self.x)), n, replace=False)
        return Dataset(self.x[mask], self.y[mask])

    def split(self, ratio):
        total = len(self.x)
        validation_examples = int(total * (1 - ratio))
        training_examples = total - validation_examples
        training_data = Dataset(self.x[0:training_examples], self.y[0:training_examples])
        validation_data = Dataset(self.x[-validation_examples:], self.y[-validation_examples:])
        return training_data, validation_data

