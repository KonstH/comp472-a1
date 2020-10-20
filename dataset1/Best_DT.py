from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from utils import split_feats_targs, capture_features, capture_targets, export_results

(train_features, train_targets) = split_feats_targs('train_1.csv')  # pass training set with targets
test_features = capture_features('test_no_label_1.csv', False)  # pass test set without targets
actual_targets = capture_targets('test_with_label_1.csv')  # pass test set with targets

"""
Parameter options to tune:
  • splitting criterion: gini and entropy
  • maximum depth of the tree: 10 and no maximum
  • minimum number of samples to split an internal node: experiment with values of your choice
  • minimum impurity decrease: experiment with values of your choice
  • class weight: None and balanced
"""

print("Finding best hyperparameters for DT....")
best_dt = GridSearchCV(DecisionTreeClassifier(), {
  'criterion': ['gini', 'entropy'],
  'max_depth': [10, None],
  'min_samples_split': [2,3,5],
  'min_impurity_decrease': [0.0, 1e-250, 1e-900],
  'class_weight': [None, 'balanced']
}, return_train_score = False, n_jobs = -1)

best_dt.fit(train_features, train_targets)
best_params = best_dt.best_params_    # records best found params from gridsearch
print("Best hyperparameters for DT:")
print(best_params)
print("\n")

best_dt = DecisionTreeClassifier(criterion = best_params['criterion'], max_depth = best_params['max_depth'], min_samples_split=best_params['min_samples_split'], min_impurity_decrease=best_params['min_impurity_decrease'], class_weight = best_params['class_weight'])
fitted_dt = best_dt.fit(train_features, train_targets)   # fits model with training set values
predicted_targets = list(fitted_dt.predict(test_features))   # gets predictions from model and record them
export_results(actual_targets, predicted_targets, 'Best-DT-DS1.csv')

