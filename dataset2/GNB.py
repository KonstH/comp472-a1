"""Importing all necessary modules"""
from sklearn.naive_bayes import GaussianNB
from utils import split_feats_targs, capture_features, capture_targets, export_results

(train_features, train_targets) = split_feats_targs('train_2.csv')  # pass training set with targets
test_features = capture_features('test_no_label_2.csv', False) # pass test set w/o targets

fitted_clf = GaussianNB().fit(train_features, train_targets)  # We fit the model with our training set values
predicted_targets = list(fitted_clf.predict(test_features)) # Displays our predictions in a nice format
actual_targets = capture_targets('test_with_label_2.csv')  # pass test set with targets

export_results(actual_targets, predicted_targets, 'GNB-DS2.csv')