import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import argparse
import os
from tqdm.auto import tqdm
import time


# Torch ML libraries
import transformers
from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

#intel Gaudi server
import habana_frameworks.torch.core as htcore

# Misc.
import warnings
warnings.filterwarnings('ignore')

print("Libraries Imported")


class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['Review']
        sentiment = self.data.iloc[index]['sentiment']
        # label = self.label_dict[sentiment]
        
        # Tokenizing the text
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }


class T5SentimentAnalysis:
    def __init__(self, datafile, review_col, rating):
        _, file_ext = os.path.splitext(datafile)
        if file_ext == ".csv":
            self.data_df = pd.read_csv(datafile)
        elif file_ext == ".xslx" or file_ext == ".xls" or file_ext == ".xlsx" or file_ext == ".xlsm" or file_ext == ".xlsb" or file_ext == ".odf" or file_ext == ".ods" or file_ext == ".odt":
            self.data_df = pd.read_excel(datafile)
        elif file_ext == ".json":
            self.data_df = pd.read_json(datafile)
        elif file_ext == ".h5" or file_ext == ".hdf":
            self.data_df = pd.read_hdf(datafile)
        elif file_ext == ".html":
            self.data_df = pd.read_html(datafile)
        else:
            print("File Extension not supported. Please use dataset with file ext. - .csv, .xls, .xlsx, .xlsm, .xlsb, .odf, .ods, .odt, .json, .html, .h5, .hdf")

        self.review_col = review_col
        self.rating = rating  
        self.data_df['Review'] = self.data_df.apply(self.fill_review, axis=1)
        self.data_df['sentiment'] = self.data_df.Rating.apply(self.to_sentiment)
        
    def fill_review(self, row):
        if pd.isna(row['Review Text']):
            if row['Rating'] in [5, 4]:
                return 'excellent'
            elif row['Rating'] == 3:
                return 'good'
            elif row['Rating'] in [2, 1, 0]:
                return 'poor'
            
        return row['Review Text']

    
    def to_sentiment(self, rating):        
        rating = int(rating)
        if rating <= 2:
            return 0
        elif rating == 3:
            return 1
        else:
            return 2

    def extract_features(self):
        df = self.data_df[['Review', 'sentiment']]

        return df

    def split_dataset(self, target_df):
        train_df, val_df = train_test_split(target_df, test_size=0.2)

        return train_df, val_df

    def training_model(self, model, device, optimizer, num_epochs, train_loader):
        start_time = time.time()
        num_training_steps = num_epochs * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))
        model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                # Assuming `labels` is your target sequence for training
                decoder_input_ids = torch.full((labels.size(0), 1), model.config.pad_token_id, dtype=torch.long, device=device)
                decoder_attention_mask = (decoder_input_ids != model.config.pad_token_id).long()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=decoder_attention_mask,
                                        labels=labels)
                loss = outputs.loss
        
                optimizer.zero_grad()
                loss.backward()
                
                htcore.mark_step()
                
                optimizer.step()
                
                htcore.mark_step()
                
                progress_bar.update(1)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        end_time = time.time()

        print("Training Time : ", end_time-start_time)
        return model
    
    def inference(self, model, val_loader, tokenizer, device):
        model.eval()  # Set the model to evaluation mode
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Create dummy decoder_input_ids
                # For T5, the pad token is used as a dummy input for decoder_input_ids during classification
                dummy_decoder_input_ids = torch.full(
                    (input_ids.shape[0], 1), 
                    model.config.decoder_start_token_id, 
                    dtype=torch.long).to(device)
                
                # Pass both encoder and dummy decoder inputs
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    decoder_input_ids=dummy_decoder_input_ids,
                    labels=labels
                )
                
                logits = outputs.logits  # Assuming the model modification returns logits directly
                
                # Apply softmax to logits to get class probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Determine the predicted class with the highest probability
                predicted_classes = torch.argmax(probs, dim=-1)
                
                predictions.extend(predicted_classes.detach().cpu().tolist())
                actuals.extend(labels.detach().cpu().tolist())
                
        return predictions, actuals

    def calc_metrics(self, predictions, actuals):
        accuracy = accuracy_score(actuals, predictions)
        precision, recall, fscore, _ = precision_recall_fscore_support(actuals, predictions, average='weighted')
        
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")
        print(f"F1-score: {fscore*100:.2f}%")

        report = classification_report(actuals, predictions, target_names=['negative', 'neutral', 'positive'])
        print("=========================================================================================")
        print(report)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='The path of the data to be processed.')
    parser.add_argument('--review_col', type=str, help='The review column (text) of the data.')
    parser.add_argument('--rating', type=str, help='The rating column of the data.')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training.')

    args = parser.parse_args()
    
    print("Arguments Parsed...\n", args.data, args.review_col, args.rating, args.num_epochs)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("hpu")
    
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("Tokenizer Loaded...\n")
    print("Model Loaded...")
    
    t5s_a = T5SentimentAnalysis(args.data, args.review_col, args.rating)
    target_df = t5s_a.extract_features()
    print(target_df.head())
    train_df, val_df = t5s_a.split_dataset(target_df)

    train_sent_data = SentimentDataset(train_df, tokenizer, max_length=512)
    val_sent_data = SentimentDataset(val_df, tokenizer, max_length=512)
    
    train_loader = DataLoader(train_sent_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_sent_data, batch_size=16)

    print(len(train_loader), len(val_loader))
    
    trained_model = t5s_a.training_model(model, device, optimizer, args.num_epochs, train_loader)

    trained_model = trained_model.to("cpu")

    torch.save(trained_model.state_dict(), "/mnt/datasets/example-dataset/T5_model.pth")
    print("Saved PyTorch Model State to T5_model.pth")

    preds, actual = t5s_a.inference(trained_model, val_loader, tokenizer, device)
    t5s_a.calc_metrics(preds, actual)
