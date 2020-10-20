from sklearn.tree import DecisionTreeClassifier
from utils import split_feats_targs, capture_features, capture_targets, export_results

(train_features, train_targets) = split_feats_targs('train_1.csv')  # pass training set with targets
test_features = capture_features('test_no_label_1.csv', False)  # pass test set without targets
actual_targets = capture_targets('test_with_label_1.csv')  # pass test set with targets

fitted_dt = DecisionTreeClassifier(criterion='entropy').fit(train_features, train_targets)  # fits model with training set values
predicted_targets = list(fitted_dt.predict(test_features))  # gets predictions from model and record them
export_results(actual_targets, predicted_targets, 'Base-DT-DS1.csv')