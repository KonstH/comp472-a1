"""
Implementation notes:
  • Since assignment guidelines did not specify the amount of max iterations, the default of 200 is given. MLP will therefore not converge
  • We use the training set to tune the hyperparameters, because the GridSearchCV library functions based on cross-validation. It can therefore
    only take in 1 dataset. Since the training set has more data, it is best to use it instead of the validation set.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from utils import split_feats_targs, capture_features, capture_targets,export_results

(train_features, train_targets) = split_feats_targs('train_1.csv')  # pass training set with targets
test_features = capture_features('test_no_label_1.csv', False)  # pass test set without targets
actual_targets = capture_targets('test_with_label_1.csv')  # pass test set with targets

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
best_params = best_mlp.best_params_   # records best found params from gridsearch
print("Best hyperparameters for MLP:")
print(best_params)
print("\n")

best_mlp = MLPClassifier(activation=best_params['activation'],hidden_layer_sizes=best_params['hidden_layer_sizes'] ,solver=best_params['solver'])
fitted_mlp = best_mlp.fit(train_features, train_targets)    # fits model with training set values
predicted_targets = list(fitted_mlp.predict(test_features))   # gets predictions from model and record them
export_results(actual_targets, predicted_targets, 'Best-MLP-DS1.csv')
