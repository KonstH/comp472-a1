"""Importing all necessary modules"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from utils import split_feats_targs, capture_features, capture_targets

(train_values, train_classes) = split_feats_targs('train_1.csv')
test_values = capture_features('test_no_label_1.csv', False)

"""
Parameter options to tune:
  • activation function: sigmoid, tanh, relu and identity
  • 2 network architectures of your choice: for eg 2 hidden layers with 30+50 nodes, 3 hidden layers with 10+10
  • solver: Adam and stochastic gradient descent
"""

# fitted_mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(100,), solver='sgd').fit(train_values, train_classes)

# predicted_targets = list(map(int, fitted_mlp.predict(test_values))) # Displays our predictions in a nice format
# actual_targets = capture_targets('test_with_label_1.csv')  # pass test set with targets
