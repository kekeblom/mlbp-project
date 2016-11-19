import numpy as np
import theano
from theano import tensor as T

class SimpleRecurrent(object):
    def __init__(self, hidden_units, input_dimensions, output_dimensions, activation=T.nnet.relu):
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.hidden_dimensions = hidden_units
        self.sequence_length = input_dimensions # not necessarily the case for non mnist but let's go with it for now.
        self.activation = activation

        weight_shape = (self.hidden_dimensions, self.hidden_dimensions)
        initial_weights = np.random.normal(loc=0.0, scale=0.1, size=weight_shape)
        self.W = theano.shared(initial_weights, name='simple_recurrent_weights')

        u_shape = (self.input_dimensions, self.hidden_dimensions)
        initial_u = np.random.normal(loc=0.0, scale=0.1, size=u_shape)
        self.U = theano.shared(initial_u, name='simple_recurrent_u')

        v_shape = (self.hidden_dimensions, self.output_dimensions)
        initial_v = np.random.normal(loc=0.0, scale=0.1, size=v_shape)
        self.V = theano.shared(initial_v, name='simple_recurrent_v')

        initial_biases = np.zeros(self.output_dimensions)
        self.biases = theano.shared(initial_biases, name='simple_recurrent_biases')


    def build(self, input, batch_size):
        input = input.reshape((batch_size, self.input_dimensions, self.sequence_length))

        def step(x_t, o_t_previous, s_t_previous):
            s_t = T.tanh(T.dot(x_t, self.U) + T.dot(s_t_previous, self.W))
            o_t = self.activation(T.dot(s_t, self.V)) + self.biases
            return [o_t, s_t]

        def outer_step(batch_item):
            initial_o_t = np.zeros(self.output_dimensions)
            initial_s_t = np.zeros(self.hidden_dimensions)

            output, sequences = theano.scan(step,
                    outputs_info=[initial_o_t, initial_s_t],
                    sequences=batch_item)
            return output[0][-1]

        outputs, _ = theano.scan(outer_step, sequences=input)

        return outputs

    def get_params(self):
        return [self.W, self.U, self.V, self.biases]

    def get_weights(self):
        return [self.W, self.U, self.V]

