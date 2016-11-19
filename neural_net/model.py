import theano
from theano import tensor as T
import numpy as np
from neural_net.optimizers import VanillaGradientDescent

class Model(object):

    def __init__(self,
            minibatch_size=10,
            learning_rate=0.1,
            cost=None,
            lmbd=0.0,
            optimizer=VanillaGradientDescent,
            clipnorm=None,
            input_type=T.matrix('x'),
            output_type=T.ivector('y')):
        self.layers = []
        self.model = None
        self.cost = None
        self.minibatch_size = minibatch_size
        self.x = input_type
        self.y = output_type
        self.CostFunctionClass = cost
        self.lmbd = lmbd
        self.clipnorm = clipnorm
        self.optimizer = optimizer(learning_rate, clip_gradient_norm=clipnorm)

    def add(self, layer):
        self.layers.append(layer)

    def compile(self):
        self.output, self.cost = self._compile(self.minibatch_size)

    def _compile(self, batch_size):
        output = self.layers[0].build(self.x, batch_size)
        prev_output = output
        for layer in self.layers[1:]:
            output = layer.build(prev_output, batch_size)
            prev_output = output

        output = prev_output

        cost = self.CostFunctionClass(output, self.y, model=self).get_function()
        return output, cost

    def eval_cost(self, dataset):
        x, y = dataset.x, dataset.y
        cost = theano.function([self.x, self.y], self.cost)
        costs = []
        for i in range(0, int(len(y) / self.minibatch_size)):
            costs.append(cost(
                x[i * self.minibatch_size:(i+1) * self.minibatch_size],
                y[i * self.minibatch_size:(i+1) * self.minibatch_size]
            ))
        return np.mean(costs)

    def predict(self, x):
        output, _ = self._compile(len(x))
        predict = theano.function([self.x], output)
        return predict(x)

    def test_accuracy(self, dataset):
        x = dataset.x
        y = dataset.y

        accuracy = theano.function([self.x, self.y], self.layers[-1].accuracy(self.y, self.output))
        mb_accuracy = []
        for i in range(0, int(len(dataset.y) / self.minibatch_size)):
            mb_accuracy.append(accuracy(
                dataset.x[i * self.minibatch_size:(i+1) * self.minibatch_size],
                dataset.y[i * self.minibatch_size:(i+1) * self.minibatch_size]
                ))

        return np.mean(mb_accuracy)

    def minibatch_SGD(self, training_data):
        training_data = training_data.shuffle()
        batch_index = T.lscalar()

        minibatch_size = self.minibatch_size

        updates = self.optimizer.updates(self.cost, self.get_params(), minibatch_size)

        training_x = theano.shared(training_data.x)
        training_y = theano.shared(training_data.y)

        self.train = theano.function([batch_index], self.cost, updates=updates,
            givens={
                self.x: training_x[batch_index * minibatch_size:(batch_index+1) * minibatch_size],
                self.y: training_y[batch_index * minibatch_size:(batch_index+1) * minibatch_size]
            })

        training_batch_count = int(len(training_data.x) / self.minibatch_size)
        costs = []
        for batch in range(0, training_batch_count):
            cost = self.train(batch)
            costs.append(np.mean(cost))

        return costs

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params

    def get_weights(self):
        weights = []
        for layer in self.layers:
            layer_weights = layer.get_weights()
            for weight in layer_weights:
                weights.append(weight.get_value())

        return weights

