import nltk
nltk.download('punkt')
nltk.download('stopwords')

import os
from tqdm import tqdm
import modin.pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
import time
import psutil
import json
plt.style.use('ggplot')
stop=set(stopwords.words('english'))



class DataLoader:
    @staticmethod
    def load_data(dataset_path) -> pd.DataFrame:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        data = []
        for root, _, files in os.walk(dataset_path):
            for file in tqdm(files, desc="Loading files"):
                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    label = os.path.basename(root)
                    data.append({'text': text, 'label': label})

        df = pd.DataFrame(data)
        return df.sample(frac=1).reset_index(drop=True)



class TextPreprocessor:
    def __init__(self, 
                 abbrev_path='utils/abbreviations.json'):
        self.stop_words = set(stopwords.words('english'))
        with open(abbrev_path, 'r') as f:
            self.abbreviations = json.load(f)

    def clean(self, 
              text: str) -> str:
        if not isinstance(text, str) or not text:
            raise ValueError("Input must be a non-empty string.")

        text = text.lower()
        text = self.remove_url(text)
        text = self.remove_html(text)
        text = self.remove_emoji(text)
        text = self.remove_stopwords(text)
        text = self.insert_spaces_around_punct(text)
        text = self.remove_all_punct(text)
        text = self.convert_abbreviations(text)
        return text

    def remove_url(self, 
                   text: str) -> str: return re.sub(r'https?://\S+|www\.\S+', '', text)

    def remove_html(self, 
                    text: str) -> str: return re.sub(r'<.*?>', '', text)

    def remove_emoji(self, 
                     text: str) -> str:
        pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols
            u"\U0001F680-\U0001F6FF"  # transport
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
        return pattern.sub('', text)

    def remove_stopwords(self, 
                         text: str) -> str:
        return " ".join([word for word in text.split() if word not in self.stop_words])

    def insert_spaces_around_punct(self, 
                                   text: str) -> str:
        for p in '@#!?+&*[]-%.:/();$=><{}^\'`':
            text = text.replace(p, f' {p} ')
        return text.replace('...', ' ... ').replace('..', ' ... ')

    def remove_all_punct(self, 
                         text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))

    def convert_abbreviations(self, 
                              text: str) -> str:
        tokens = word_tokenize(text)
        expanded = [self.abbreviations.get(word.lower(), word) for word in tokens]
        return ' '.join(expanded)

    def process_dataframe(self, 
                          df: pd.DataFrame) -> pd.DataFrame:
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'text' and 'label' columns.")
        df['text'] = df['text'].apply(self.clean)
        return df


class DatasetManager:
    def __init__(self, 
                 logger, 
                 preprocessor: TextPreprocessor)-> None:
        self.logger = logger
        self.preprocessor = preprocessor

    def train_test_split_and_process(self, 
                                     df: pd.DataFrame, 
                                     test_size=0.3, 
                                     preprocess=True, 
                                     random_state=42)-> tuple:
        if preprocess:
            df = self.preprocessor.process_dataframe(df)

        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
        return df, df_train, df_test

    def save_image(self, 
                   fig, 
                   folder_path, 
                   image_name)-> None:
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{image_name}.png")
        fig.savefig(file_path)
        self.logger.info(f"Saved image to: {file_path}")

    def log_metrics(self, 
                    model_name, 
                    phase, 
                    start_time, 
                    start_memory, 
                    y_true, 
                    y_pred=None)-> None:
        end_time = time.time()
        current_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        time_taken = end_time - start_time
        mem_diff = current_mem - start_memory

        self.logger.info(f"\n--- Metrics for {model_name} ({phase}) ---")
        self.logger.info(f"Time taken: {time_taken:.2f} seconds")
        self.logger.info(f"Memory usage increase: {mem_diff:.2f} MB")

        if phase.lower() == 'prediction' and y_pred is not None:
            report = classification_report(y_true, y_pred, zero_division=0)
            self.logger.info("\nClassification Report:\n" + report)
            return report
        return None
