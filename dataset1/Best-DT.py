"""Importing all necessary modules"""
from sklearn.tree import DecisionTreeClassifier
from utils import split_feats_targs, capture_features, capture_targets

(train_features, train_targets) = split_feats_targs('train_1.csv')
test_features = capture_features('test_no_label_1.csv', False)

"""
Parameter options to tune:
  • splitting criterion: gini and entropy
  • maximum depth of the tree: 10 and no maximum
  • minimum number of samples to split an internal node: experiment with values of your choice
  • minimum impurity decrease: experiment with values of your choice
  • class weight: None and balanced
"""

# TEST 1: Gini, None, 2, 0, None (accuracy: 0.4375)
# dt = DecisionTreeClassifier(criterion = 'gini', max_depth = None, min_samples_split = 2, min_impurity_decrease = 0, class_weight = None)

# # TEST 2: Gini, 10, 2, 0, None (accuracy: 0.4)
# dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, min_samples_split = 2, min_impurity_decrease = 0, class_weight = None)

# # TEST 3: Gini, None, 2, 0, Balanced (accuracy: )
# dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, min_samples_split = 2, min_impurity_decrease = 0, class_weight = None)

# # TEST 4: Gini, 10, 2, 0, Balanced (accuracy: )
# dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, min_samples_split = 2, min_impurity_decrease = 0, class_weight = None)

# # TEST 5: Gini, 10, 2, 0, None
# dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, min_samples_split = 2, min_impurity_decrease = 0, class_weight = None)

# # TEST 6: Gini, 10, 2, 0, None
# dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, min_samples_split = 2, min_impurity_decrease = 0, class_weight = None)

# # TEST 7: Gini, 10, 2, 0, None
# dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, min_samples_split = 2, min_impurity_decrease = 0, class_weight = None)

# # TEST 8: Gini, 10, 2, 0, None
# dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, min_samples_split = 2, min_impurity_decrease = 0, class_weight = None)

# val_targets = capture_targets('val_1.csv')  # pass validation set with targets