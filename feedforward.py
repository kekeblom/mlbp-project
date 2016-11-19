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

STOPPING_CRITERIA = 5

_, training_data, training_labels = data.load_training_data("./classification_dataset_training.csv")
training_data = Dataset(
        training_data.astype(np.int32),
        training_labels.astype(np.int32))
training_data, validation_data = training_data.split(0.8)

_, test_data = data.load_test_data('./classification_dataset_testing.csv')
_, test_labels = data.load_test_labels('./classification_dataset_testing_solution.csv')

test_data = Dataset(test_data.astype(np.int32), test_labels.astype(np.int32))

input = T.dmatrix('input')

class RegularizedCost(object):
    def __init__(self, output, y, model=None):
        self.output = output
        self.y = y
        self.model = model

    def get_function(self):
        weights = np.concatenate([np.asarray(m).reshape(-1) for m in self.model.get_weights()])
        l2_norm = (weights ** 2).sum()
        l2_weight = 0.5
        return T.mean(T.nnet.categorical_crossentropy(self.output, self.y)) + l2_weight * l2_norm

model = Model(learning_rate=0.01,
        cost=RegularizedCost,
        optimizer=optimizers.RMSProp,
        minibatch_size=100,
        lmbd=0.0,
        input_type=T.imatrix('x'))

model.add(FullyConnected(
    input_dimensions=50,
    output_dimensions=12,
    activation=T.nnet.sigmoid))

model.add(FullyConnected(
    input_dimensions=12,
    output_dimensions=10,
    activation=T.nnet.softmax))

model.compile()

best_accuracy = 0.0
epoch = 0
epochs_without_improvement = 0

def accuracy_score(model, dataset):
    return metrics.accuracy_score(model.predict(dataset.x).argmax(axis=1), dataset.y)

costs_on_training = []
while 1:
    epoch += 1

    print("Epoch", epoch)

    training_data = training_data.shuffle()

    cost = np.mean(model.minibatch_SGD(training_data))
    costs_on_training.append(cost)

    print('cost:', cost)

    accuracy_on_validation = accuracy_score(model, validation_data)
    print("Accuracy on validation:", accuracy_on_validation)
    print("Accuracy on test:", accuracy_score(model, test_data))

    if epochs_without_improvement > STOPPING_CRITERIA:
        break

    elif accuracy_on_validation > best_accuracy:
        best_accuracy = accuracy_on_validation
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1


model.minibatch_SGD(validation_data)

accuracy_on_test_set = model.test_accuracy(test_data)

print("Accuracy: ", accuracy_on_test_set)

pyplot.plot(range(1, epoch + 1), costs_on_training)
pyplot.show()

