"""
DATASET 1
Running this file will run every single estimator for the given training and test set

Make sure that in this directory, you have included:
    • training set 1 csv file
    • test set with no labels 1 csv file
    • test set with labels 1 csv file
    • validation set 1 csv file

These files are necessary for the code to properly run.
"""
from utils import split_feats_targs, capture_features, capture_targets, export_results
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


"""
Store the training/test set values into variables
"""
(train_features, train_targets) = split_feats_targs('train_1.csv')  # pass training set with targets
test_features = capture_features('test_no_label_1.csv', False) # pass test set w/o targets
actual_targets = capture_targets('test_with_label_1.csv')  # pass test set with targets

"""
Run GNB model
"""
fitted_gnb = GaussianNB().fit(train_features, train_targets)  # Fit model with training set values
predicted_targets = list(fitted_gnb.predict(test_features)) # Displays our predictions in a nice format
export_results(actual_targets, predicted_targets, 'GNB-DS1.csv')

"""
Run PER model
"""
fitted_per = Perceptron().fit(train_features, train_targets)  # Fit model with training set values
predicted_targets = list(fitted_per.predict(test_features)) # Displays our predictions in a nice format  
export_results(actual_targets, predicted_targets, 'PER-DS1.csv')

"""
Run BaseDT model
"""
fitted_baseDT = DecisionTreeClassifier(criterion='entropy').fit(train_features, train_targets)  # Fit model with training set values
predicted_targets = list(fitted_baseDT.predict(test_features))   #   
export_results(actual_targets, predicted_targets, 'Base-DT-DS1.csv')

"""
Find best hyperparameters and run BestDT model

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

best_dt.fit(train_features, train_targets)
best_params = best_dt.best_params_
print("Best hyperparameters for DT:")
print(best_params)
print("\n")

best_dt = DecisionTreeClassifier(criterion = best_params['criterion'], max_depth = best_params['max_depth'], min_samples_split=best_params['min_samples_split'], min_impurity_decrease=best_params['min_impurity_decrease'], class_weight = best_params['class_weight'])
fitted_dt = best_dt.fit(train_features, train_targets)
predicted_targets = list(fitted_dt.predict(test_features))  # Displays our predictions in a nice format  
export_results(actual_targets, predicted_targets, 'Best-DT-DS1.csv')

"""
Run BaseMLP model
"""
fitted_mlp = MLPClassifier(activation='logistic', solver='sgd').fit(train_features, train_targets)
predicted_targets = list(fitted_mlp.predict(test_features)) # Displays our predictions in a nice format  
export_results(actual_targets, predicted_targets, 'Base-MLP-DS1.csv')

"""
Find best hyperparameters and run BestMLP model

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

fitted_mlp = MLPClassifier(activation=best_params['activation'],hidden_layer_sizes=best_params['hidden_layer_sizes'] ,solver=best_params['solver']).fit(train_features, train_targets)
predicted_targets = list(fitted_mlp.predict(test_features)) # Displays our predictions in a nice format  
export_results(actual_targets, predicted_targets, 'Best-MLP-DS1.csv')

