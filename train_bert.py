import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import clean_text
import kagglehub

class SentimentDataset(Dataset):
    """
    BERT Dataset Class
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """
    Dataset Load
    """
    path = kagglehub.dataset_download("kazanova/sentiment140")
    
    df = pd.read_csv(path + '/training.1600000.processed.noemoticon.csv', 
                     encoding='latin-1', 
                     header=None,
                     names=['target', 'id', 'date', 'flag', 'user', 'text'])
    
    df = df[['target', 'text']]
    df['target'] = df['target'].replace(4, 1)
    df = df.sample(n=5000, random_state=42)
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    return df

def train_bert():
    """
    BERT Model Train
    """
    # Data Load
    df = load_data()
    
    X = df['cleaned_text']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Device Check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tokenizer Load
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Datasets
    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_dataset = SentimentDataset(X_test.tolist(), y_test.tolist(), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Model Load
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Training Loop
    print("\n BERT Training...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/3")
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Average Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluation
    print("\n BERT Evaluation...")
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(actual_labels, predictions)
    print(f"\n BERT Accuracy: {accuracy:.4f}")
    
    return accuracy, predictions, actual_labels

if __name__ == "__main__":
    bert_acc, preds, labels = train_bert()