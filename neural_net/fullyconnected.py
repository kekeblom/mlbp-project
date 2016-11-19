import numpy as np
import theano
from theano import tensor

class FullyConnected(object):
    def __init__(self, input_dimensions=None, output_dimensions=None, activation=tensor.nnet.relu):
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions

        weights = np.asarray(np.random.normal(loc=0.0, scale=0.5, size=(input_dimensions, output_dimensions)), dtype=theano.config.floatX)

        biases = np.zeros((output_dimensions), dtype=theano.config.floatX)

        self.weights = theano.shared(value=weights, name='W', borrow=True)
        self.biases = theano.shared(value=biases, name='b', borrow=True)
        self.activation = activation

    def build(self, input, batch_size):
        reshaped = input.reshape((batch_size, self.input_dimensions))
        output = self.activation(tensor.dot(reshaped, self.weights) + self.biases)
        return output

    def get_params(self):
        return [self.weights, self.biases]

    def get_weights(self):
        return [self.weights]

    def accuracy(self, labels, output):
        return tensor.mean(tensor.eq(labels, tensor.argmax(output, axis=1)))

