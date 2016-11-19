import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

class ConvPoolLayer(object):
    def __init__(self, filter_shape, input_shape, pool_size=(2, 2), activation=T.nnet.relu):
        """
        input_shape: (number of input channels, image height, image width)
        filter_shape: (number of output channels, number of input channels, filter height, filter width)
        """
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.pool_size = pool_size

        fan_in = np.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size))

        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        self.weights = theano.shared( np.asarray(
                    np.random.uniform(
                        low=-w_bound,
                        high=w_bound,
                        size=filter_shape
                        )
                ))
        bias_values = np.zeros((filter_shape[0]), dtype=theano.config.floatX)
        self.biases = theano.shared(value=bias_values, borrow=True)
        self.activation = activation

    def build(self, input, batch_size):
        input = input.reshape((batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        convolution_output = conv2d(
                input=input,
                filters=self.weights,
                filter_shape=self.filter_shape,
                input_shape=(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        pooled = pool.pool_2d(
                input=convolution_output,
                ds=self.pool_size,
                ignore_border=True)

        return self.activation(pooled + self.biases.dimshuffle('x', 0, 'x', 'x'))

    def get_params(self):
        return [self.weights, self.biases]

    def get_weights(self):
        return [self.weights]


