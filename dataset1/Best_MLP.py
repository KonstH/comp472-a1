"""Importing all necessary modules"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from utils import split_feats_targs, capture_features, capture_targets,export_results

(train_features, train_targets) = split_feats_targs('train_1.csv')
test_features = capture_features('test_no_label_1.csv', False)


def find_best_params_MLP():
  (train_features, train_targets) = split_feats_targs('train_1.csv')
  """
  Parameter options to tune:
    • activation function: sigmoid, tanh, relu and identity
    • 2 network architectures of your choice: for eg 2 hidden layers with 30+50 nodes, 3 hidden layers with 10+10
    • solver: Adam and stochastic gradient descent
  """
  print("Finding best hyperparameters for MLP....")
  best_mlp = GridSearchCV(MLPClassifier(), {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30,50), (10,10,10)],
    'solver': ['sgd', 'adam']
  }, return_train_score = False, n_jobs = -1)

  best_mlp.fit(train_features, train_targets)
  best_params = best_mlp.best_params_
  print("Best hyperparameters for MLP:")
  print(best_params)
  print("\n")
  return best_params

best_params = find_best_params_MLP()

fitted_mlp = MLPClassifier(activation=best_params['activation'],hidden_layer_sizes=best_params['hidden_layer_sizes'] ,solver=best_params['solver']).fit(train_features, train_targets)
predicted_targets = list(fitted_mlp.predict(test_features)) # Displays our predictions in a nice format
actual_targets = capture_targets('test_with_label_1.csv')  # pass test set with targets

export_results(actual_targets, predicted_targets, 'Best-MLP-DS1.csv')
