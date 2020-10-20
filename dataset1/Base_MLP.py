"""Importing all necessary modules"""
from sklearn.neural_network import MLPClassifier
from utils import split_feats_targs, capture_features, capture_targets, export_results

(train_features, train_targets) = split_feats_targs('train_1.csv')  # pass training set with targets
test_features = capture_features('test_no_label_1.csv', False) # pass test set w/o targets

fitted_mlp = MLPClassifier(activation='logistic', solver='sgd').fit(train_features, train_targets)
predicted_targets = list(fitted_mlp.predict(test_features)) # Displays our predictions in a nice format
actual_targets = capture_targets('test_with_label_1.csv')  # pass test set with targets

export_results(actual_targets, predicted_targets, 'Base-MLP-DS1.csv')