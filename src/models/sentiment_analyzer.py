import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import Dataset, DataLoader
import fasttext
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AmazonReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentAnalyzer:
    def __init__(self, model_type: str = "bert"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type (str): Type of model to use ('bert' or 'fasttext')
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        if model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=2
            ).to(self.device)
        else:
            self.model = None
            self.tokenizer = None
    
    def train_fasttext(self, train_file: str, model_path: str):
        """
        Train FastText model.
        
        Args:
            train_file (str): Path to training data
            model_path (str): Path to save the model
        """
        logger.info("Training FastText model")
        self.model = fasttext.train_supervised(
            input=train_file,
            lr=0.5,
            epoch=25,
            wordNgrams=2,
            verbose=2
        )
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def train_bert(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                  batch_size: int = 16, epochs: int = 3):
        """
        Train BERT model.
        
        Args:
            train_data (pd.DataFrame): Training data
            val_data (pd.DataFrame): Validation data
            batch_size (int): Batch size
            epochs (int): Number of epochs
        """
        logger.info("Training BERT model")
        
        train_dataset = AmazonReviewsDataset(
            texts=train_data['text'].values,
            labels=train_data['sentiment'].values,
            tokenizer=self.tokenizer
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data (pd.DataFrame): Test data
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        logger.info("Evaluating model")
        
        if self.model_type == "fasttext":
            predictions = []
            for text in test_data['text']:
                pred = self.model.predict(text)[0][0].replace('__label__', '')
                predictions.append(1 if pred == '2' else 0)
        else:
            test_dataset = AmazonReviewsDataset(
                texts=test_data['text'].values,
                labels=test_data['sentiment'].values,
                tokenizer=self.tokenizer
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False
            )
            
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    _, preds = torch.max(outputs.logits, 1)
                    predictions.extend(preds.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(
            test_data['sentiment'],
            predictions,
            output_dict=True
        )
        
        # Create confusion matrix
        cm = confusion_matrix(test_data['sentiment'], predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        return {
            'classification_report': report,
            'confusion_matrix': cm
        }

if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer(model_type="bert")
    
    # Load data
    train_data = pd.read_csv("data/processed/processed_train.csv")
    test_data = pd.read_csv("data/processed/processed_test.csv")
    
    # Train and evaluate
    analyzer.train_bert(train_data, test_data)
    results = analyzer.evaluate(test_data)
    
    # Print results
    print("\nClassification Report:")
    print(pd.DataFrame(results['classification_report']).transpose()) 