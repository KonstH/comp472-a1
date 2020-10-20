"""Importing all necessary modules"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from utils import split_feats_targs, capture_features, capture_targets, export_results

(train_features, train_targets) = split_feats_targs('train_2.csv')  # pass training set with targets
test_features = capture_features('test_no_label_2.csv', False)

def find_best_params_DT():
  (val_values, val_classes) = split_feats_targs('val_2.csv')

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
    'min_samples_split': [2,3,4,5],
    'min_impurity_decrease': [0.0, 1e-06, 1e-90, 1e-900],
    'class_weight': [None, 'balanced']
  }, return_train_score = False, n_jobs = -1)

  best_dt.fit(val_values, val_classes)
  best_params = best_dt.best_params_
  print("Best hyperparameters for DT:")
  print(best_params)
  print("\n")
  return best_params

best_params = find_best_params_DT()

fitted_dt = DecisionTreeClassifier(criterion = best_params['criterion'], max_depth = best_params['max_depth'], min_samples_split=best_params['min_samples_split'], min_impurity_decrease=best_params['min_impurity_decrease'], class_weight = best_params['class_weight'])
fitted_dt.fit(train_features, train_targets)

predicted_targets = list(fitted_dt.predict(test_features)) # Displays our predictions in a nice format
actual_targets = capture_targets('test_with_label_2.csv')  # pass test set with targets

export_results(actual_targets, predicted_targets, 'Best-DT-DS2.csv')

