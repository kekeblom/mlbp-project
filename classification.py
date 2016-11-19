import data

import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
from sklearn import decomposition
from sklearn.pipeline import Pipeline


headers, training_data, training_labels = data.load_training_data("./classification_dataset_training.csv")


_, test_data = data.load_test_data("./classification_dataset_testing.csv")
_, test_labels = data.load_test_labels("./classification_dataset_testing_solution.csv")

random_forests = map(lambda n: ensemble.RandomForestClassifier(min_samples_leaf=n, n_estimators=100), range(1, 10))

pipelines = []
for classifier in random_forests:
    for i in range(5, 50):
        pca = decomposition.PCA(n_components=i)
        pipelines.append(Pipeline(steps=[
            ('pca', pca),
            ('fit', classifier)
        ]))


cross_validation_scores = []
for classifier in pipelines:
    score = model_selection.cross_val_score(classifier, training_data, training_labels,
            cv=model_selection.KFold(n_splits=5))
    cross_validation_scores.append(np.mean(score))

best_classifier = classifiers[np.array(cross_validation_scores).argmax()]
print("best classifier:", best_classifier)

best_classifier.fit(training_data, training_labels)
print("Test accuracy:", metrics.accuracy_score(best_classifier.predict(test_data), test_labels))

import ipdb; ipdb.set_trace()
