import os
import json
import time
import logging
import numpy as np
import modin.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import torch
import seaborn as sns
from transformers import (
    AutoTokenizer,
    Trainer, TrainingArguments,
    DistilBertForSequenceClassification, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    BertTokenizer
)
from utils.utils import (
    DataLoader, 
    TextPreprocessor, 
    DatasetManager
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix
)


class TransformerTextClassifier:
    def __init__(self, 
                 args)-> None:
        self.args = args
        self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        self.min_count_samples = args.min_count_samples
        self.model_identifier = f"{self.model_name}_balancedCount{self.min_count_samples}"
        self.save_path = f"models/{self.model_identifier}"
        self.label_path = f"models/{self.model_identifier}.json"
        self.log_path = args.log if args.log else f"log/complex_methods_{self.model_identifier}.log"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.label_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def get_top_keywords_by_tfidf(self, 
                                  data: pd.DataFrame, 
                                  top_n: int = 512) -> pd.DataFrame:
        grouped = data.groupby('label')
        records = []
        for label, group in tqdm(grouped):
            docs = group['text']
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            feature_names = self.vectorizer.get_feature_names_out()
            for doc_idx in range(tfidf_matrix.shape[0]):
                tfidf_scores = tfidf_matrix[doc_idx].toarray().flatten()
                top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
                keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                records.append({
                    'label': label,
                    'keywords': ' '.join(keywords),
                    'text': docs.iloc[doc_idx]
                })
        return pd.DataFrame(records)

    def prepare_dataset(self, 
                        max_len: int) -> pd.DataFrame:
        preprocessor = TextPreprocessor()
        manager = DatasetManager(logging, preprocessor)
        data_org = DataLoader.load_data(self.dataset_name)
        data_balanced = data_org.groupby('label').apply(
            lambda x: x.sample(n=self.min_count_samples, random_state=42)).reset_index(drop=True)
        validation_checks = {
            "empty_dataset": data_balanced.empty,
            "missing_columns": not {'text', 'label'}.issubset(data_balanced.columns),
            "null_values": data_balanced[['text', 'label']].isnull().any().any()
        }
        if any(validation_checks.values()):
            raise ValueError("Dataset validation failed.")
        data_balanced = preprocessor.process_dataframe(data_balanced)
        data = self.get_top_keywords_by_tfidf(data_balanced, max_len)
        if data.empty or not {'text', 'label'}.issubset(data.columns) or data[['text', 'label']].isnull().any().any():
            raise ValueError("Data validation failed after keyword extraction.")
        return data

    def _get_tokenizer(self):
        if "roberta" in self.model_name:
            return RobertaTokenizer.from_pretrained('roberta-base')
        elif "distilbert" in self.model_name:
            return AutoTokenizer.from_pretrained(self.model_name)
        else:
            return BertTokenizer.from_pretrained(self.model_name)

    def _get_model_class(self, 
                         num_labels: int):
        if "roberta" in self.model_name:
            return RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        elif "distilbert" in self.model_name:
            return DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        else:
            return BertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)

    def tokenize_data(self, 
                      df: pd.DataFrame, 
                      tokenizer) -> Dataset:
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=self.args.max_len), batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    def compute_metrics(self, 
                        pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


    def train(self) -> None:
        df = self.prepare_dataset(self.args.max_len)
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["label"])
        with open(self.label_path, 'w', encoding='utf-8') as f:
            json.dump(le.classes_.tolist(), f, ensure_ascii=False, indent=2)
        temp, test_df = train_test_split(df, test_size=0.3, random_state=42)
        train_df, val_df = train_test_split(temp, test_size=0.1, random_state=42)
        tokenizer = self._get_tokenizer()
        train_dataset = self.tokenize_data(train_df, tokenizer)
        val_dataset = self.tokenize_data(val_df, tokenizer)
        test_dataset = self.tokenize_data(test_df, tokenizer)
        model = self._get_model_class(df["label"].nunique()).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        logging.info(f"Training model {self.model_identifier} with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
        logging.info(f"Model will be saved to {self.save_path} and labels to {self.label_path}")

        start_time = time.time()
        training_args = TrainingArguments(
            output_dir="./results",
            logging_steps=500,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.train_bachSize,
            per_device_eval_batch_size=self.args.eval_bachSize,
            learning_rate=2e-5,
            warmup_steps=200,
            weight_decay=0.01,
            metric_for_best_model="accuracy",
            seed=42,
            fp16=True,
            dataloader_num_workers=4
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
        logging.info(f"Saving model to {self.save_path} and tokenizer to {self.save_path}")
        model.save_pretrained(self.save_path)
        tokenizer.save_pretrained(self.save_path)
        eval_results = trainer.evaluate(test_dataset)
        predictions = trainer.predict(test_dataset)
        cm = confusion_matrix(predictions.label_ids, predictions.predictions.argmax(-1))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"log/{self.model_identifier}_confusion_matrix.png")
        plt.close()
        logging.info(f"Confusion Matrix:\n{cm}")
        logging.info(f"Evaluation Results: {json.dumps(eval_results, indent=2)}")
        logging.info(f"Confusion Matrix saved to log/{self.model_identifier}_confusion_matrix.png")
    
    def test(self) -> pd.DataFrame:
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"Model not found at {self.save_path}. Please train the model first.")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label mapping not found at {self.label_path}. Please train the model first.")
        tokenizer = self._get_tokenizer()
        model = self._get_model_class(0).from_pretrained(self.save_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        test_df = DataLoader.load_data(self.args.test_dataset_name)
        test_df = test_df.sample(n=self.min_count_samples, random_state=42)
        test_dataset = self.tokenize_data(test_df, tokenizer)
        trainer = Trainer(model=model, tokenizer=tokenizer)
        predictions = trainer.predict(test_dataset)
        preds = predictions.predictions.argmax(-1)
        with open(self.label_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        pred_labels = [labels[p] for p in preds]
        test_df['predicted_label'] = pred_labels
        test_df.to_csv(f"log/{self.model_identifier}_test_results.csv", index=False)
        logging.info(f"Test results saved to log/{self.model_identifier}_test_results.csv")
        return test_df[['text', 'predicted_label']]