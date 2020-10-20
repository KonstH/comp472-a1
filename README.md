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
- To generate all output files for `dataset1` at once, run the file: `dataset1/main.py`
- To generate a single output file for a specific model, run the appropriate model file under the `dataset1` folder
  * **Ex:** to run and generate output file for only the GNB model, run: `dataset1/GNB.py`
  
- To generate all output files for `dataset2` at once, run the file: `dataset2/main.py`
- To generate a single output file for a specific model, run the appropriate model file under the `dataset2` folder
  * **Ex:** to run and generate output file for only the GNB model, run: `dataset2/GNB.py`
  
Additional Notes
---

- You might have noticed the folder dataset_demo. This is reserved for running the models on specific files provided during the assignment demo, and exporting their results
- The `Best` type models all have some specific implementation notes added as comments in their respective code files
- Points `2.3.a`, `2.3.b`, `2.3.c` and `2.3.d` in the assignment guidelines are all present in the output csv files
