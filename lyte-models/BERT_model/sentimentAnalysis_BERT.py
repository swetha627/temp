import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import argparse
import os
from tqdm.auto import tqdm
import time

# Torch ML libraries
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#intel Gaudi server
import habana_frameworks.torch.core as htcore

# Misc.
import warnings
warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class GPReviewDataset(Dataset):
    # Constructor Function
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output["pooler_output"]
        output = self.drop(pooled_output)

        return self.out(output)
        

class BERTSentimentAnalysis:
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

    def split_dataset(self, RANDOM_SEED, train_test_size,test_val_size):
        df_train, df_test = train_test_split(self.data_df, test_size=train_test_size, random_state=RANDOM_SEED)
        df_val, df_test = train_test_split(df_test, test_size=test_val_size, random_state=RANDOM_SEED)

        return df_train, df_test, df_val

    def train_epoch(self, model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
        model = model.train()
        losses = []
        correct_predictions = 0
        for d in tqdm(data_loader, total=len(data_loader), desc="Training"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
    
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    
            loss.backward()
            htcore.mark_step()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            htcore.mark_step()
            scheduler.step()
            htcore.mark_step()
            optimizer.zero_grad()
    
        return correct_predictions.double() / n_examples, np.mean(losses)

    def calling_train_eval_func(self, model, train_data_loader, val_data_loader, loss_fn, optimizer, device, scheduler, epochs, best_accuracy, df_train, df_val):
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 10)
            train_acc, train_loss = self.train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(df_train)
            )
            print(f"Train loss {train_loss} accuracy {train_acc}")
            
            val_acc, val_loss = self.eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(df_val)
            )
            print(f"Val   loss {val_loss} accuracy {val_acc}")
            print()
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
        
            if val_acc > best_accuracy:
                torch.save(model.state_dict(), 'best_model_state.bin')
                best_accuracy = val_acc

        end_time = time.time()
        print("Training Time : ", end_time-start_time)

    def eval_model(self, model, data_loader, loss_fn, device, n_examples):
        model = model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
    
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())
    
        return correct_predictions.double() / n_examples, np.mean(losses)

    def get_predictions(self, model, data_loader):
        model = model.eval()
        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []
    
        with torch.no_grad():
            i=0
            for d in data_loader:
                print(i)
                i+=1
                texts = d["review_text"]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
    
                review_texts.extend(texts)
                # predictions.extend(preds)
                # prediction_probs.extend(outputs)
                # real_values.extend(targets)
                predictions.append(preds.detach().cpu())  # Appending to list after detaching and moving to CPU
                prediction_probs.append(outputs.detach().cpu())  # Same for prediction probabilities
                real_values.append(targets.detach().cpu())
        # print("Evaluation Completed...")
        # predictions = torch.stack(predictions)#.cpu()
        # print("Predictions Stack Completed : ", len(predictions))
        # prediction_probs = torch.stack(prediction_probs)#.cpu()
        # print("Predictions Probs Stack Completed : ", len(prediction_probs))
        # real_values = torch.stack(real_values)#.cpu()
        # print("Real Values Stack Completed : ", len(real_values))
        predictions = torch.cat(predictions, dim=0)
        prediction_probs = torch.cat(prediction_probs, dim=0)
        real_values = torch.cat(real_values, dim=0)
    
        return review_texts, predictions, prediction_probs, real_values

    def calc_metrics(self, model, test_data_loader):
        print("Calculating Metrics...")
        y_review_texts, y_pred, y_pred_probs, y_test = self.get_predictions(model, test_data_loader)
        print("**************************************************************")
        print("Classification Report \n")
        y_pred = y_pred.cpu().numpy()
        y_test = y_test.cpu().numpy()
        print(classification_report(y_test, y_pred, target_names=class_names))


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.Review.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='The path of the data to be processed.')
    parser.add_argument('--review_col', type=str, help='The review column (text) of the data.')
    parser.add_argument('--rating', type=str, help='The rating column of the data.')
    parser.add_argument('--batch_size', type=int, help='The batch size of the data.')
    parser.add_argument('--random_seed', type=int, help='The random seed value for the data.', default=42)
    parser.add_argument('--train_test_size', type=float, help='Train test dataset ratio for splitting.')
    parser.add_argument('--test_val_size', type=float, help='Test val dataset ratio for splitting.')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training.')

    args = parser.parse_args()
    
    print("Arguments Parsed...\n")
    device = torch.device("hpu")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MAX_LEN = 160
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    berts_a = BERTSentimentAnalysis(args.data, args.review_col, args.rating)

    # MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # model = BertModel.from_pretrained(MODEL_NAME).to(device)

    print("Tokenizer Loaded...\n")
    print("Model Loaded...")
    
    class_names = ['negative', 'neutral', 'positive']
    model = SentimentClassifier(len(class_names))
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    history = defaultdict(list)
    best_accuracy = 0
    
    df_train, df_test, df_val = berts_a.split_dataset(args.random_seed, args.train_test_size, args.test_val_size)
    
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, args.batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, args.batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, args.batch_size)
    
    print("train_data_loader length : ", len(train_data_loader))
    print("test_data_loader length : ", len(test_data_loader))

    total_steps = len(train_data_loader) * args.num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    # loss_fn = nn.CrossEntropyLoss().to(device)
    loss_fn = FocalLoss().to(device)

    berts_a.calling_train_eval_func(model, train_data_loader, val_data_loader, loss_fn, optimizer, device, scheduler, args.num_epochs, best_accuracy, df_train, df_val)

    berts_a.calc_metrics(model, test_data_loader)
