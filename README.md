# Long Documents Classification

This project provides scripts for long documents classification using various methods, including simple methods (keyword-based by count and TF-IDF vectorization), intermediate methods (TF-IDF + Logistic Regression, SVM, MLP, and XGBoost) and complex methods (TF-IDF + DistilBERT, BERT and RoBERTa). The main scripts are:

- `run_simple_methods.py`: Script for running simple classification methods (keyword-based and TF-IDF vectorization).
- `run_intermediate_methods.py`: Main script for running simple and intermediate classification methods (keyword-based, TF-IDF, Logistic Regression, SVM, MLP, XGBoost).
- `run_complex_methods.py`: Script for running complex classification methods (TF-IDF + DistilBERT, BERT, RoBERTa).
- `utils.py`: Utility functions for data loading, preprocessing, and evaluation.
- `complex_utils.py`: Utility functions for complex classification methods.
- `intermediate_utils.py`: Utility functions for intermediate classification methods.
- `simple_utils.py`: Utility functions for simple classification methods.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```


## Download and Prepare Dataset

[Download](https://github.com/LiqunW/Long-document-dataset) the dataset.

Organize your dataset as follows:

```
dataset/Long-document-dataset/
├── label1/
│   ├── file1.txt
│   ├── file2.txt
│   └── ...
├── label2/
│   ├── file3.txt
│   └── ...
└── ...
```

Each name like cs.AI is a class label, and each file is a document.

In our experiments we use 8 cs. classes and 3 math. classes.


## Usage


### 1. Run Simple Methods

Run the script for simple classification methods (Count and TF-IDF vectorization are used for keyword-based classification):

```bash
python3 run_simple_methods.py
```

#### Arguments

- `--dataset`: Path to the dataset directory (default: `dataset/Long-document-dataset`)
- `--log`: Path to the log file (default: `log/simple_methods.log`)
- `--method`: Choosing a method for vectorization (default: `tdidf`)
- `--minimum_count_samples`: Number of documents used for training and testing (30% is automatically used for testing). (default: 100)
- `--numkey`: Number of extracted keywords for each category (default: 512)

Example:

```bash
python3 run_simple_methods.py --dataset 'dataset/Long-document-dataset' --method count --minimum_count_samples 20 --numkey 100

```

---

### 2. Run Intermediate Methods

Run the script for simple and intermediate classification methods (keyword-based, TF-IDF, Logistic Regression, SVM, MLP, XGBoost):

```bash
python3 run_intermediate_methods.py
```

#### Arguments

- `--dataset`: Path to the dataset directory (default: `dataset/Long-document-dataset`)
- `--log`: Path to the log file (default: `log/intermediate_methods.log`)
- `--minimum_count_samples`: List of minimum samples per class for balancing (default: `["org", "minlength", "100", "140"]`)
- `--models`: List of models to use (default: `['reg', 'svm', 'mlp', 'xgb']`)

Example:

```bash
python3 run_intermediate_methods.py --log 'log/intermediate_methods_512.log' --minimum_count_samples "org" "100" "140" "minlength" "1000" --models 'reg' 'svm' 'mlp' 'xgb'
```

---

### 3. Run Complex Methods

Run the script for complex classification methods (TF-IDF + DistilBERT, BERT, RoBERTa):

```bash
python3 run_complex_methods.py
```

#### Arguments

- `--dataset`: Path to the dataset directory (default: `dataset/Long-document-dataset`)
- `--log`: Path to the log file (default: `log/complex_methods_{model_name}.log`)
- `--model_name`: Name of the transformer model to use (`distilbert-base-uncased`, `bert-base-uncased`, default: `roberta-base`)
- `--minimum_count_samples`: Minimum samples per class for balancing (default: `100`)
- `--epochs`: Number of model training epochs (default: `100`)
- `--train_bachSize`: Training batch size (default: `32`)
- `--eval_bachSize`: Evaluation batch size (default: `8`)
- `--max_len`: Maximum sequence length (default: 512)
- `--mode`: Mode to run (`train` or `test`, default: `train`)
- `--test_dataset_name`: Path to external test dataset (default: `None`)
- `--model_dir`: Path to load model for testing (default: `None`)

Example:

```bash
python3 run_complex_methods.py --log 'log/complex_methods_out_bert.log' --mode 'train' --model_name 'bert-base-uncased' --mode 'train' --epochs 10 --train_bachSize 64
```


## Logging

Logs, confusion matrix images and tensorboards are saved in the `log/` directory.
Traind complex methods like BERT are saved in `models/` directory. Also, the last 2 checkpoints are saved in `results/` directory

## License

This project, supported by Procycons, aims to explore methods for classifying long documents.
