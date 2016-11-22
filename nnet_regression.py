import data
import numpy as np
import theano
from theano import tensor as T
from neural_net import Model, FullyConnected
from neural_net.dataset import Dataset
from matplotlib import pyplot
from neural_net import optimizers
from neural_net.cost import ClassificationCost
from sklearn import metrics

STOPPING_CRITERIA = 6

_, training_data, training_labels = data.load_training_data("./regression_dataset_training.csv")
training_data = Dataset(
        training_data.astype(np.float32),
        training_labels.astype(np.float32)).shuffle()
training_data, validation_data = training_data.split(0.9)

_, test_data = data.load_test_data('./regression_dataset_testing.csv')
_, test_labels = data.load_test_labels('./regression_dataset_testing_solution.csv')

test_data = Dataset(test_data.astype(np.float32), test_labels.astype(np.float32))

class RegularizedCost(object):
    def __init__(self, output, y, model=None):
        self.output = output
        self.y = y
        self.model = model

    def get_function(self):
        weights = np.concatenate([np.asarray(m).reshape(-1) for m in self.model.get_weights()])
        l2_norm = (weights ** 2).sum()
        l2_weight = 1.0
        return T.mean((self.y - T.sum(self.output, axis=1)) ** 2) + l2_weight * l2_norm

def score(model, dataset):
    return metrics.mean_squared_error(dataset.y, model.predict(dataset.x))

class NNetRegressor(object):
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Model(learning_rate=0.0005,
                cost=RegularizedCost,
                optimizer=optimizers.RMSProp,
                minibatch_size=500,
                lmbd=0.0,
                input_type=T.fmatrix('x'),
                output_type=T.fvector('y'))

        model.add(FullyConnected(
            input_dimensions=50,
            output_dimensions=19,
            activation=T.nnet.relu))

        model.add(FullyConnected(
            input_dimensions=19,
            output_dimensions=10,
            activation=T.nnet.relu))

        model.compile()
        return model

    def predict(self, dataset):
        predictions = self.model.predict(dataset).sum(axis=1)
        # Round the predictions as we are regressing to integers.
        return np.round(predictions)

    def fit(self, training_data):
        best_mean_squared_error = 99999.0
        epoch = 0
        epochs_without_improvement = 0
        costs_on_training = []
        while 1:
            epoch += 1

            print("Epoch", epoch)

            training_data = training_data.shuffle()

            cost = np.mean(self.model.minibatch_SGD(training_data))
            costs_on_training.append(cost)

            print('cost:', cost)

            validation_mean_squared_error = score(self, validation_data)
            print("Mean squared error:", validation_mean_squared_error)

            if epochs_without_improvement > STOPPING_CRITERIA:
                break

            elif validation_mean_squared_error < best_mean_squared_error:
                best_mean_squared_error = validation_mean_squared_error
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1


regressor = NNetRegressor()
regressor.fit(training_data)

import ipdb; ipdb.set_trace()
mean_squared_error = score(regressor, test_data)

print("Mean squared error: ", mean_squared_error)

