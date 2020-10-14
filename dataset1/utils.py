import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
curr_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) # Store current directory path


"""
Function which splits csv file contents into features and classes, and returns a tuple of this pair
  arg: file_name (name of the csv file we want to split)
"""
def split_feats_targs(file_name):
  values = []
  classes = []
  # We convert the entries of the csv file into array indices to manipulate it in python
  with open(os.path.join(curr_dir, file_name), 'r') as f:
    lines = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    for line in lines:
      values.append(line)
      classes.append(values[-1].pop())

  return (values, classes)
    
"""
Function which takes csv file containing features with targets, and only returns targets
  arg: file_name (name of the csv file we want to split)
  arg: include_classes (if true, we return values with their observed class. If false, only return values)
"""
def capture_targets(file_name):
  targets = []
  # We convert the entries of the csv file into array indices to manipulate it in python
  with open(os.path.join(curr_dir, file_name), 'r') as f:
    lines = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    for line in lines:
      targets.append(line[-1])
    
  return (targets)

"""
Function which takes csv file containing features with/without targets, and only returns features
  arg: file_name (name of the csv file we want to split)
  arg: has_targets (does csv file include targets)
"""
def capture_features(file_name, has_targets):
  features = []
  # We convert the entries of the csv file into array indices to manipulate it in python
  with open(os.path.join(curr_dir, file_name), 'r') as f:
    lines = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)

    if(has_targets):
      for line in lines:
        line.pop()
        features.append(line)
    else:
      for line in lines:
        features.append(line)
    
  return (features)


"""
Function which creates array of entries where each entry an instance's row followed by its predicted class
  arg: predictions (predicted classes)
"""
def preds_summary(real, predictions):
  i = 1
  results = []

  # 3a: Append row num of instance with its predicted class
  for target in predictions:
    results.append("%d,%d" % (i, target))
    i = i + 1

  # return results object
  return(results)


def export_results(real, pred, file_name):
  results = preds_summary(real, pred)

  f = open(file_name,'w')
  print("Exporting predicted classes...")
  f.write('3A - Predictions:\n\n')
  for result in results:
    f.write(result + '\n') 

  f.close()
  f = open(file_name,'a')

  # 3B: Append confusion matrix
  print("Exporting confusion matrix...")
  f.write('\n\n3B - Confusion Matrix:\n\n')
  cm = np.array(confusion_matrix(real, pred))
  np.savetxt(f, cm, fmt='%i', delimiter=",")

  # 3C: Append precision scores
  print("Exporting precision scores...")
  f.write('\n\n3C - Precision Scores:\n\n')
  ps = np.array(precision_score(real, pred, average=None, zero_division=1))
  np.savetxt(f, ps, fmt='%1.5f', delimiter=",")

  # 3C: Append recall scores
  print("Exporting recall scores...")
  f.write('\n\n3C - Recall Scores:\n\n')
  rs = np.array(recall_score(real, pred, average=None, zero_division=1))
  np.savetxt(f, rs, fmt='%1.5f', delimiter=",")

  # 3C: Append f1 measures
  print("Exporting f1 measures...")
  f.write('\n\n3C - F1 Scores:\n\n')
  f1 = np.array(f1_score(real, pred, average=None, zero_division=1))
  np.savetxt(f, f1, fmt='%1.5f', delimiter=",")

  # 3D: Append accuracy
  print("Exporting accuracy measure...")
  f.write('\n\n3D - Accuracy:\n\n')
  accuracy = ("%1.5f" % accuracy_score(real, pred))
  f.write(accuracy + '\n')

  # 3D: Append Macro Average f1 measure
  print("Exporting Macro Average measure...")
  f.write('\n\n3D - Macro Average F1:\n\n')
  # f1 = np.array(f1_score(real, pred, average='macro', zero_division=1))
  ma_f1 = ("%1.5f" % f1_score(real, pred, average='macro', zero_division=1))
  f.write(ma_f1 + '\n')

  # 3D: Append Weighted Average f1 measure
  print("Exporting Weighted Average measure...")
  f.write('\n\n3D - Weighted Average F1:\n\n')
  wa_f1 = ("%1.5f" % f1_score(real, pred, average='weighted', zero_division=1))
  f.write(wa_f1 + '\n')

  print("All metrics have been successfully exported to " + file_name + "!")

  f.close()


  
