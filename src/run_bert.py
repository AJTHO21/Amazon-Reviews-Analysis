import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

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

def load_fasttext_data(file_path):
    """
    Load and parse fastText formatted data.
    
    Args:
        file_path (str): Path to the fastText file
        
    Returns:
        pd.DataFrame: DataFrame containing the parsed data
    """
    logger.info(f"Loading data from {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split the line into label and text
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                label = parts[0].replace('__label__', '')
                text = parts[1]
                data.append({'label': label, 'text': text})
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} reviews")
    return df

def train_bert_model(train_data, val_data, model_path, batch_size=16, epochs=4, max_length=256, learning_rate=1e-5):
    """
    Train a BERT model.
    
    Args:
        train_data (pd.DataFrame): Training data
        val_data (pd.DataFrame): Validation data
        model_path (str): Path to save the model
        batch_size (int): Batch size
        epochs (int): Number of epochs
        max_length (int): Maximum sequence length
        learning_rate (float): Learning rate for the optimizer
        
    Returns:
        tuple: (model, tokenizer, training history)
    """
    logger.info("Training BERT model")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    
    # Create datasets
    train_dataset = AmazonReviewsDataset(
        texts=train_data['text'].values,
        labels=train_data['sentiment'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = AmazonReviewsDataset(
        texts=val_data['text'].values,
        labels=val_data['sentiment'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            _, preds = torch.max(outputs.logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Record learning rate
            history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                _, preds = torch.max(outputs.logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
    
    # Save the model
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, tokenizer, history

def evaluate_bert_model(model, tokenizer, test_data, batch_size=32, max_length=512):
    """
    Evaluate the BERT model on test data.
    
    Args:
        model (BertForSequenceClassification): Trained model
        tokenizer (BertTokenizer): Tokenizer
        test_data (pd.DataFrame): Test data
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (predictions, true labels)
    """
    logger.info("Evaluating BERT model")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and data loader
    test_dataset = AmazonReviewsDataset(
        texts=test_data['text'].values,
        labels=test_data['sentiment'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Evaluation
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions, true_labels

def create_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Create and plot confusion matrix.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        save_path (str, optional): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()
    
    return cm

def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.close()

def main():
    # Set paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    models_dir = Path("models")
    results_dir = Path("results")
    
    # Create directories if they don't exist
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Set file paths
    train_file = raw_dir / "train.ft.txt"
    test_file = raw_dir / "test.ft.txt"
    model_path = models_dir / "bert_model"
    
    # Load data
    logger.info("Loading data...")
    train_df = load_fasttext_data(str(train_file))
    test_df = load_fasttext_data(str(test_file))
    
    # Take a smaller subset for faster training on CPU
    train_size = 10000  # Increased from 5000
    test_size = 2000   # Increased from 1000
    
    train_df = train_df.sample(n=train_size, random_state=42)
    test_df = test_df.sample(n=test_size, random_state=42)
    
    logger.info(f"Using {train_size} training samples and {test_size} test samples")
    
    # Convert labels to binary
    train_df['sentiment'] = train_df['label'].map({'1': 0, '2': 1})
    test_df['sentiment'] = test_df['label'].map({'1': 0, '2': 1})
    
    # Train model with optimized parameters
    model, tokenizer, history = train_bert_model(
        train_data=train_df,
        val_data=test_df,
        model_path=str(model_path),
        batch_size=16,      # Increased from 8
        epochs=4,           # Increased from 2
        max_length=256,     # Increased from 128
        learning_rate=1e-5  # Slightly lower learning rate
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=str(results_dir / "bert_training_history.png")
    )
    
    # Evaluate model
    predictions, true_labels = evaluate_bert_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_df,
        batch_size=32,      # Increased from 16
        max_length=256      # Same as training
    )
    
    # Create confusion matrix
    cm = create_confusion_matrix(
        true_labels,
        predictions,
        save_path=str(results_dir / "bert_confusion_matrix.png")
    )
    
    # Print classification report
    report = classification_report(
        true_labels,
        predictions,
        target_names=['Negative', 'Positive']
    )
    
    logger.info("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(results_dir / "bert_classification_report.txt", 'w') as f:
        f.write(report)
    
    logger.info("BERT analysis completed successfully!")

if __name__ == "__main__":
    main() 