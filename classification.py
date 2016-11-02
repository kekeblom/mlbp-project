from data import load_table

import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import KFold

headers, table, labels = load_table("./classification_dataset_training.csv")


class KFoldCrossValidator(object):
    def __init__(self,
            classifier,
            data=None,
            labels=None,
            cost=None,
            fold_count=5):
        self.data = data
        self.new_classifier = classifier
        self.fold_count = fold_count
        self.cost = cost
        self.labels = labels

    def fit(self):
        costs = []
        kfold = KFold(n_splits=self.fold_count)
        for train, validation in kfold.split(self.data):
            training_data, training_labels = self.data[train], self.labels[train]
            validation_data, validation_labels = self.data[validation], self.labels[validation]

            classifier = self.new_classifier()
            classifier.fit(training_data, training_labels)

            predictions = classifier.predict_proba(validation_data)
            cost = self.cost(validation_labels, predictions)

            costs.append(cost)
        return np.mean(costs)


validator = KFoldCrossValidator(
        classifier=lambda: tree.DecisionTreeClassifier(max_depth=3),
        data=table,
        labels=labels,
        cost=metrics.log_loss)
mean_cost = validator.fit()


