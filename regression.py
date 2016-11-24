import data

import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
from sklearn import decomposition
from sklearn.pipeline import Pipeline


headers, training_data, training_labels = data.load_training_data("./regression_dataset_training.csv")


_, test_data = data.load_test_data("./regression_dataset_testing.csv")
_, test_labels = data.load_test_labels("./regression_dataset_testing_solution.csv")

regressors = []
for min_samples in range(1, 10):
    for max_depth in range(1, 50, 3):
            regressors.append(ensemble.RandomForestRegressor(
                min_samples_leaf=min_samples,
                max_depth=max_depth,
                n_estimators=100))

cross_validation_scores = []
for regressor in regressors:
    score = model_selection.cross_val_score(regressor, training_data, training_labels,
            cv=model_selection.KFold(n_splits=5))
    cross_validation_scores.append(np.mean(score))

best_classifier = regressors[np.array(cross_validation_scores).argmax()]
print("best regressor:", best_classifier)

best_classifier.fit(training_data, training_labels)
print("Mean squared error:", metrics.mean_squared_error(test_labels, best_classifier.predict(test_data)))

import ipdb; ipdb.set_trace()
