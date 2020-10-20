## COMP 472 - Assignment 1

https://github.com/KonstH/comp472-a1.git

How to run program
---

### Step 1 - Setup
Directly inside of the `dataset1` folder, include the following files:
- `train_1.csv`
- `test_no_label_1.csv`
- `test_with_label_1.csv`

Directly inside of the `dataset2` folder, include the following files:
- `train_2.csv`
- `test_no_label_2.csv`
- `test_with_label_2.csv`
- `val_2.csv`

Python modules used/required for this project:
- `os`
- `csv`
- `numpy`
- `scikit-learn`

### Step 2 - Run Models / Generate files (multiple options)
- To generate all outputs for `dataset1` at once, run the file: `dataset1/main.py`
- To generate an output file for a single model, run the appropriate model file under `dataset1` folder
  * **ex:** to run and generate output file for only GNB model, run: `dataset1/GNB.py`
  
- To generate all outputs for `dataset2` at once, run the file: `dataset2/main.py`
- To generate an output file for a single model, run the appropriate model file under `dataset2` folder
  * **ex:** to run and generate output file for only GNB model, run: `dataset2/GNB.py`
