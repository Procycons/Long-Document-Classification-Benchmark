import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import modin.pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from utils.utils import (
    TextPreprocessor, 
    DatasetManager
)
from sklearn.metrics import (
    classification_report, 
    confusion_matrix
)



class IntermediateClassifier:
    def __init__(self, 
                 args, 
                 logger, 
                 preprocessor=None, 
                 manager=None) -> None:
        self.args = args
        self.logger = logger
        self.classification_reports = {}
        self.training_metrics = {}
        self.prediction_metrics = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=512)
        self.text_preprocessor = preprocessor if hasattr(args, 'preprocessor') else TextPreprocessor()
        self.dataset_manager = manager if hasattr(args, 'manager') else DatasetManager(logger, self.text_preprocessor)

    def _log_memory_time(self, 
                         start_time, 
                         start_memory) -> tuple:
        duration = time.time() - start_time
        memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) - start_memory
        return duration, memory_used

    def _run_model(self, 
                   model: object, 
                   model_name: str, 
                   y_train: pd.Series, 
                   y_test: pd.Series, 
                   X_train_tfidf: pd.DataFrame, 
                   X_test_tfidf: pd.DataFrame, 
                   log_name, 
                   label_encoder=None) -> None:
        self.logger.info(f"Running model: {model_name}")
        
        # Training
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        model.fit(X_train_tfidf, y_train)
        train_time, train_memory = self._log_memory_time(start_time, start_memory)

        self.logger.info(f"Model {model_name} trained in {train_time:.4f}s with memory usage {train_memory:.2f}MB")
        # Save the trained model
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        model_path = os.path.join(results_dir, f"{log_name}_{model_name.replace(' ', '_').lower()}.joblib")
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.logger.info(f"Saving {model_name} model to {model_path}")
        else:
            self.logger.warning(f"Model path {model_path} already exists. Overwriting.")
        joblib.dump(model, model_path)
        self.logger.info(f"Saved {model_name} model to {model_path}")
        # Prediction
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        y_pred = model.predict(X_test_tfidf)
        if label_encoder:
            y_pred = label_encoder.inverse_transform(y_pred)
        pred_time, pred_memory = self._log_memory_time(start_time, start_memory)

        report = classification_report(y_test, y_pred, zero_division=0)
        self.classification_reports[model_name] = report
        self.training_metrics[model_name] = {'time': train_time, 'memory': train_memory}
        self.prediction_metrics[model_name] = {'time': pred_time, 'memory': pred_memory}

        print(f"\n--- {model_name} ---\n{report}")
        print(f"Training time: {train_time:.4f}s | Memory: {train_memory:.2f}MB")
        print(f"Prediction time: {pred_time:.4f}s | Memory: {pred_memory:.2f}MB")

        self.dataset_manager.log_metrics(model_name, 'Training', start_time, start_memory, y_train)
        self.dataset_manager.log_metrics(model_name, 'Prediction', start_time, start_memory, y_test, y_pred)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        self.dataset_manager.save_image(fig, 'log/confusion_matrix', f'confusion_matrix_{log_name}_{model_name.lower()}')
        plt.close(fig)

    def classify(self, 
                 X_train: pd.Series, 
                 y_train: pd.Series, 
                 X_test: pd.Series, 
                 y_test: pd.Series, 
                 log_name: str) -> tuple:
        self.logger.info(f"--- Log Name {log_name} ---")

        try:
            print("Starting TF-IDF + classifier section...")
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = self.tfidf_vectorizer.transform(X_test)

            tfidf_time, tfidf_memory = self._log_memory_time(start_time, start_memory)
            self.logger.info(f"TF-IDF Time: {tfidf_time:.2f}s | Memory: {tfidf_memory:.2f}MB")

            models = [m.lower() for m in self.args.models]

            if "reg" in models:
                self._run_model(LogisticRegression(max_iter=1000), "Logistic Regression", y_train, y_test,
                                X_train_tfidf, X_test_tfidf, log_name)

            if "svm" in models:
                self._run_model(SVC(), "SVM", y_train, y_test,
                                X_train_tfidf, X_test_tfidf, log_name)

            if "mlp" in models:
                self._run_model(MLPClassifier(max_iter=100), "MLP", y_train, y_test,
                                X_train_tfidf, X_test_tfidf, log_name)

            if "xgb" in models or "xgboost" in models:
                label_encoder = LabelEncoder()
                y_train_enc = label_encoder.fit_transform(y_train)
                y_test_enc = label_encoder.transform(y_test)

                self._run_model(XGBClassifier(objective='multi:softmax',
                                              num_class=len(label_encoder.classes_),
                                              eval_metric='mlogloss',
                                              use_label_encoder=False),
                                "XGBoost", y_train_enc, y_test, X_train_tfidf, X_test_tfidf, log_name,
                                label_encoder=label_encoder)

            return self.classification_reports, self.training_metrics, self.prediction_metrics

        except Exception as e:
            self.logger.error("Error in TF-IDF + classifier section: %s", str(e), exc_info=True)
            print(f"Error in TF-IDF classification: {e}")
            return None, None, None

    def process_with_min_count(self, 
                               df: pd.DataFrame, 
                               min_count: int) -> None:
        self.logger.info(f"#####  min count set on {min_count}\n")

        if 'label' not in df.columns:
            raise ValueError("The DataFrame must contain a 'label' column for balancing.")

        actual_min_count = df['label'].value_counts().min()
        if min_count > actual_min_count:
            self.logger.warning(f"Requested min_count {min_count} > actual min {actual_min_count}. Adjusting.")
            min_count = actual_min_count
        self.logger.info(f"Using min_count: {min_count}")

        df_balanced = df.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
        self.logger.info("Balanced dataset created.")

        try:
            print(f"Starting balanced data processing with min_count={min_count}...")
            df_test, df_train_balanced, df_test_balanced = self.dataset_manager.train_test_split_and_process(
                df_balanced, test_size=0.3, preprocess=True, random_state=42
            )

            print(f"Train/Test shapes: {df_train_balanced.shape}, {df_test_balanced.shape}")
            self.classify(df_train_balanced["text"], df_train_balanced["label"],
                          df_test_balanced["text"], df_test_balanced["label"],
                          log_name=f'balanced_data_min_{min_count}')
            print(f"Balanced classification with min_count={min_count} complete.")
        except Exception as e:
            self.logger.error("Error in balanced data classification: %s", str(e), exc_info=True)
            print(f"Error during balanced data processing: {e}")
