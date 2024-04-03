import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import habana_frameworks.torch.core as htcore
import warnings

warnings.filterwarnings('ignore')


class SentimentDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.train_dataset = None
        self.eval_dataset = None

    def load_and_preprocess(self):
        self.df = pd.read_csv(self.filepath)
        self.df[['text', 'sentiment']] = self.df.apply(lambda row: self._preprocess(row), axis=1).tolist()

        # Encode labels
        label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.df['labels'] = self.df['sentiment'].map(label_dict)

        # Split dataset
        train_df, eval_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_dataset = Dataset.from_pandas(train_df)
        self.eval_dataset = Dataset.from_pandas(eval_df)

    def _preprocess(self, row):
        text = row['Review Text'] if pd.notna(row['Review Text']) else 'No review text provided'
        sentiment = 'positive' if row['Rating'] > 3 else 'neutral' if row['Rating'] == 3 else 'negative'
        return text, sentiment


class SentimentModelTrainer:
    def __init__(self, train_dataset, eval_dataset, use_habana=True):
        self.tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
        self.model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=3)
        self.train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        self.eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)
        self.trainer = None
        self.use_habana = use_habana
        if self.use_habana:
            self.model = self.model.to(htcore.Device("hpu"))

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    def train_model(self, output_dir, num_train_epochs=2):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            report_to="none" 
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )

        self.trainer.train()
        return self.trainer.evaluate()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='The path of the data to be processed.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path where to save the trained model and results.')
    parser.add_argument('--num_train_epochs', type=int, default=2, help='Number of epochs for training.')
    parser.add_argument('--use_habana', action='store_true', help='Use Habana Gaudi accelerator if available.')

    args = parser.parse_args()

    data_processor = SentimentDataProcessor(args.data)
    data_processor.load_and_preprocess()

    sentiment_trainer = SentimentModelTrainer(data_processor.train_dataset, data_processor.eval_dataset, args.use_habana)
    evaluation_results = sentiment_trainer.train_model(args.output_dir, args.num_train_epochs)
    print(evaluation_results)
