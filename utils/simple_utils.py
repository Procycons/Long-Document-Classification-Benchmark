import os
os.environ["MODIN_ENGINE"] = "ray"

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))

import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.simplefilter('ignore')


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import modin.pandas as pd
import seaborn as sns
from collections import Counter
import time
from sklearn.utils import resample
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



class KeywordBasedClassifier:
    def __init__(self, 
                 args, 
                 logging, 
                 stop_words=None):
        self.args = args
        self.logging = logging
        self.num_keywords = args.numkey
        self.method = args.method
        self.stop_words = stop_words or set(stopwords.words('english'))
        self.category_keywords = None

    def extract_keywords(self, 
                         df_train)-> dict:
        if self.method == "count":
            return self._extract_keywords_count(df_train)
        elif self.method == "tfidf":
            return self._extract_keywords_tfidf(df_train)
        else:
            raise ValueError("Unsupported keyword extraction method.")

    def _extract_keywords_count(self, 
                                df_train)-> dict:
        self.logging.info("Using count-based keyword extraction.")
        self.category_keywords = {}
        for category in df_train['label'].unique():
            texts = df_train[df_train['label'] == category]['text']
            words = ' '.join(texts).split()
            word_counts = Counter(w for w in words if w not in self.stop_words)
            self.category_keywords[category] = [w for w, _ in word_counts.most_common(self.num_keywords)]
        self.save_keywords('log/keywords')
        return self.category_keywords

    def _extract_keywords_tfidf(self, 
                                df_train)-> dict:
        if df_train.empty:
            self.logging.error("Training DataFrame is empty. Cannot extract keywords.")
            raise ValueError("Training DataFrame is empty. Cannot extract keywords.")
        self.logging.info("Using TF-IDF-based keyword extraction.")
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_train['text'])
        feature_names = tfidf_vectorizer.get_feature_names_out()

        category_keywords = {}
        for label in df_train['label'].unique():
            idx = df_train['label'] == label
            tfidf_sub = tfidf_matrix[idx.to_numpy()]
            mean_tfidf = np.asarray(tfidf_sub.mean(axis=0)).flatten()
            top_idx = mean_tfidf.argsort()[::-1][:self.num_keywords]
            keywords = [feature_names[i] for i in top_idx]
            category_keywords[label] = keywords
        self.category_keywords = category_keywords
        self.save_keywords('log/tfidf_keywords')
        return category_keywords

    def classify(self, 
                 df_test)-> pd.Series:
        if df_test.empty:
            self.logging.error("Test DataFrame is empty. Cannot classify.")
            raise ValueError("Test DataFrame is empty. Cannot classify.")
        if self.category_keywords is None:
            raise ValueError("Keywords must be extracted before classification.")

        if self.method == "count":
            return self._classify_count(df_test)
        elif self.method == "tfidf":
            return self._classify_tfidf(df_test)
        else:
            raise ValueError("Unsupported classification method.")

    def _classify_count(self, 
                        df_test)-> pd.Series:
        predictions = []
        for text in df_test['text']:
            words = text.split()
            scores = {cat: sum(w in kws for w in words) for cat, kws in self.category_keywords.items()}
            prediction = max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"
            predictions.append(prediction)
        return pd.Series(predictions)

    def _classify_tfidf(self, 
                        df_test)-> pd.Series:
        category_keywords = {k: set(v) for k, v in self.category_keywords.items()}
        predictions = []

        for text in df_test['text']:
            words = set(text.split())
            scores = {cat: len(words & kws) for cat, kws in category_keywords.items()}
            prediction = max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"
            predictions.append(prediction)
        return pd.Series(predictions)
    
    def save_keywords(self, 
                      output_dir)-> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for category, keywords in self.category_keywords.items():
            with open(os.path.join(output_dir, f"{self.method}_{category}_keywords.txt"), 'w') as f:
                f.write('\n'.join(keywords))
        self.logging.info(f"Keywords saved to {output_dir}")
    

    def evaluate(self, 
                 y_true, 
                 y_pred, 
                 labels, 
                 log_name)-> dict:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Logging and plotting
        self.logging.info(f"Accuracy: {acc:.4f}")
        self.logging.info(f"Precision: {prec:.4f}")
        self.logging.info(f"Recall: {rec:.4f}")
        self.logging.info(f"F1 Score: {f1:.4f}")
        self.logging.info("\n%s", conf_matrix)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        self.args.manager.save_image(fig, 'log/confusion_matrix', f'confusion_matrix_{log_name}')
        plt.close(fig)

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }

    def prediction(self, 
                   df_test, 
                   keywords_dir='log/keywords')-> pd.Series:
        if self.category_keywords is None:
            self.logging.info("Loading keywords from directory: %s", keywords_dir)
            self.category_keywords = {}
            try:
                if not os.path.exists(keywords_dir):
                    self.logging.error(f"Keywords directory '{keywords_dir}' does not exist.")
                    raise FileNotFoundError(f"Keywords directory '{keywords_dir}' does not exist.")
                for filename in os.listdir(keywords_dir):
                    if filename.endswith('.txt'):
                        category = filename.split('_')[1]
                        with open(os.path.join(keywords_dir, filename), 'r') as f:
                            keywords = f.read().splitlines()
                        self.category_keywords[category] = keywords
            except Exception as e:
                self.logging.error(f"Error loading keywords from directory: {e}")
                raise e
        else:
            self.logging.info("Using already extracted keywords.")
            
        y_pred = self.classify(df_test)

        return y_pred


    def simple_classification(self,
                            df_train, 
                            df_test, 
                            log_name)-> tuple:
        try:
            self.logging.info(f"\n\n--- Running simple classification [{log_name}] ---\n")
            model = KeywordBasedClassifier(self.args, self.logging)
            start_extract = time.time()
            model.extract_keywords(df_train)
            extract_time = time.time() - start_extract

            start_test = time.time()
            y_pred = model.classify(df_test)
            test_time = time.time() - start_test

            y_true = df_test['label']
            labels = sorted(df_test['label'].unique())
            eval_metrics = model.evaluate(y_true, y_pred, labels, log_name)


            self.logging.info(f"Keyword extraction time: {extract_time:.2f}s")
            self.logging.info(f"Testing time: {test_time:.2f}s")
            return y_true, eval_metrics
        except Exception as e:
            self.logging.error("Error in classification: %s", str(e), exc_info=True)
            raise e


    def process_min_count_simple(self,
                                df: pd.DataFrame)-> pd.DataFrame:
        if self.args.minimum_count_samples is None:
            self.logging.info("No minimum count samples specified, using original data.")
            return df

        min_counts = []
        for val in self.args.minimum_count_samples:
            if isinstance(val, str) and val.lower() == "org":
                continue  # Skip original data processing
            elif isinstance(val, str) and val.lower() == "minlength":
                # Use the minimum count of samples in any class
                min_counts.append(df['label'].value_counts().min())
            else:
                min_counts.append(int(val))

        for min_count in min_counts:
            self.logging.info(f"#####  min count set on {min_count}\n")
            if 'label' not in df.columns:
                raise ValueError("The DataFrame must contain a 'label' column for balancing.")
            else:
                self.logging.info("The DataFrame contains a 'label' column for balancing.")
                actual_min_count = df['label'].value_counts().min()
                if min_count > actual_min_count:
                    self.logging.warning(f"Requested minimum count {min_count} is greater than the actual minimum count {actual_min_count}. Adjusting to {actual_min_count}.")
                    min_count = actual_min_count
                self.logging.info(f"Minimum count of samples in any class: {min_count}")

                # Optimized Data Balancing: Using resample for more efficient undersampling
                # Instead of groupby().apply().sample(), we iterate through classes and resample
                df_balanced_list = []
                for label in df['label'].unique():
                    df_label = df[df['label'] == label]
                    df_label_sampled = resample(df_label,
                                                replace=False,     # Sample without replacement
                                                n_samples=min_count, # To match the minimum class size
                                                random_state=42)   # For reproducible results
                    df_balanced_list.append(df_label_sampled)
                df_balanced = pd.concat(df_balanced_list)

                self.logging.info("Balanced the dataset by undersampling the majority classes using resample.\n")
                self.logging.info("The balanced dataset has been created with equal samples from each class.\n")
            try:
                print(f"Starting balanced data processing min_count={min_count}...")
                self.logging.info(f"Starting balanced data processing min_count={min_count}...\n")
                # Assume train_test_split_and_process and simple_classification are optimized internally
                # to use efficient parameters and n_jobs=-1 where applicable.
                df_balanced, df_train_balanced, df_test_balanced = self.args.manager.train_test_split_and_process(df_balanced, test_size=0.3, preprocess=True, random_state=42)
                self.logging.info("Data split into training and testing sets for balanced data.\n")
                print("Data split into training and testing sets for balanced data.")
                print(f"Calling simple_classification with balanced data min_count={min_count}...")
                self.logging.info(f"Calling simple_classification with balanced data min_count={min_count}...\n")
                # Call the classification function with the balanced data
                self.logging.info("Calling simple_classification function with balanced data...\n")

                self.simple_classification(df_train_balanced, df_test_balanced, log_name=f'balanced_data_min_{min_count}')

                print(f"Balanced data min_count={min_count} processing completed successfully.")
                self.logging.info(f"Balanced data min_count={min_count} processing completed successfully.")
            except Exception as e:
                self.logging.error("Error during balanced data processing: %s", str(e), exc_info=True)
                print(f"An error occurred during balanced data processing: {e}")
            self.logging.info(f"#####  End of processing for min_count set on {min_count}\n")