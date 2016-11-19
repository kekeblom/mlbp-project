import theano
import numpy as np
from theano import tensor as T

class BaseOptimizer(object):
    def __init__(self, learning_rate, clip_gradient_norm=None):
        self.learning_rate = learning_rate
        self.clip_value = clip_gradient_norm

    def get_gradients(self, fn, parameters, minibatch_size):
        gradient = [g / float(minibatch_size) for g in T.grad(fn, parameters)]
        norm = T.sqrt(T.sum([T.sum(g ** 2) for g in gradient]))
        return [self._clipnorm(g, norm) for g in gradient]

    def _clipnorm(self, gradient, norm):
        if self.clip_value is not None:
            return T.switch(gradient > self.clip_value, gradient * self.clip_value / norm, gradient)
        else:
            return gradient



class VanillaGradientDescent(BaseOptimizer):

    def updates(self, cost_function, parameters, minibatch_size):
        gradients = self.get_gradients(cost_function, parameters, minibatch_size)
        updates = []
        for parameter, gradient in zip(parameters, gradients):
            updated_parameter = parameter - self.learning_rate * gradient
            updates.append((parameter, updated_parameter))
        return updates

class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate, clip_gradient_norm=None):
        super().__init__(learning_rate, clip_gradient_norm=clip_gradient_norm)
        self.learning_rate = learning_rate
        self.rho = 0.9
        self.epsilon = 1e-8
        self.r = theano.shared(0.0)

    def updates(self, cost_function, parameters, minibatch_size):
        shapes = [param.get_value().shape for param in parameters]
        accumulators = [theano.shared(np.zeros(shape)) for shape in shapes]
        gradients = self.get_gradients(cost_function, parameters, minibatch_size)
        updates = []
        for parameter, gradient, accumulator in zip(parameters, gradients, accumulators):
            new_accumulator = self.rho * accumulator + (1. - self.rho) * (gradient**2)
            updates.append((accumulator, new_accumulator))

            new_parameter = parameter - self.learning_rate * (
                    gradient / (np.sqrt(new_accumulator) + self.epsilon)
                )

            updates.append((parameter, new_parameter))

        return updates

