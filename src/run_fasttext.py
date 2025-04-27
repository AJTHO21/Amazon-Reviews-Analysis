import os
import sys
import logging
import fasttext
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def train_fasttext_model(train_file, model_path, epochs=25, lr=0.5, wordNgrams=2):
    """
    Train a FastText model.
    
    Args:
        train_file (str): Path to training data
        model_path (str): Path to save the model
        epochs (int): Number of epochs
        lr (float): Learning rate
        wordNgrams (int): N-gram size
        
    Returns:
        fasttext.FastText._FastText: Trained model
    """
    logger.info(f"Training FastText model with {epochs} epochs, lr={lr}, wordNgrams={wordNgrams}")
    
    model = fasttext.train_supervised(
        input=train_file,
        epoch=epochs,
        lr=lr,
        wordNgrams=wordNgrams,
        verbose=2
    )
    
    # Save the model
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model

def evaluate_model(model, test_file):
    """
    Evaluate the model on test data.
    
    Args:
        model (fasttext.FastText._FastText): Trained model
        test_file (str): Path to test data
        
    Returns:
        tuple: (precision, recall, number of samples)
    """
    logger.info(f"Evaluating model on {test_file}")
    
    result = model.test(test_file)
    precision, recall, n_samples = result
    
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Number of samples: {n_samples}")
    
    return precision, recall, n_samples

def predict_sentiment(model, texts):
    """
    Predict sentiment for a list of texts.
    
    Args:
        model (fasttext.FastText._FastText): Trained model
        texts (list): List of texts
        
    Returns:
        list: Predicted labels
    """
    predictions = []
    for text in texts:
        pred = model.predict(text)[0][0].replace('__label__', '')
        predictions.append(1 if pred == '2' else 0)
    
    return predictions

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
    model_path = models_dir / "fasttext_model.bin"
    
    # Load data
    logger.info("Loading data...")
    train_df = load_fasttext_data(str(train_file))
    test_df = load_fasttext_data(str(test_file))
    
    # Take a smaller subset for faster training on CPU
    train_size = 10000  # Adjust this number based on your CPU capabilities
    test_size = 2000
    
    train_df = train_df.sample(n=train_size, random_state=42)
    test_df = test_df.sample(n=test_size, random_state=42)
    
    logger.info(f"Using {train_size} training samples and {test_size} test samples")
    
    # Save the subset data for training
    subset_train_file = raw_dir / "train_subset.ft.txt"
    subset_test_file = raw_dir / "test_subset.ft.txt"
    
    with open(subset_train_file, 'w', encoding='utf-8') as f:
        for _, row in train_df.iterrows():
            f.write(f"__label__{row['label']} {row['text']}\n")
    
    with open(subset_test_file, 'w', encoding='utf-8') as f:
        for _, row in test_df.iterrows():
            f.write(f"__label__{row['label']} {row['text']}\n")
    
    # Train model
    model = train_fasttext_model(
        train_file=str(subset_train_file),
        model_path=str(model_path),
        epochs=25,
        lr=0.5,
        wordNgrams=2
    )
    
    # Evaluate model
    precision, recall, n_samples = evaluate_model(model, str(subset_test_file))
    
    # Convert labels to binary
    test_df['sentiment'] = test_df['label'].map({'1': 0, '2': 1})
    
    # Predict on test data
    predictions = predict_sentiment(model, test_df['text'].values)
    
    # Create confusion matrix
    cm = create_confusion_matrix(
        test_df['sentiment'].values,
        predictions,
        save_path=str(results_dir / "fasttext_confusion_matrix.png")
    )
    
    # Print classification report
    report = classification_report(
        test_df['sentiment'].values,
        predictions,
        target_names=['Negative', 'Positive']
    )
    
    logger.info("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(results_dir / "fasttext_classification_report.txt", 'w') as f:
        f.write(report)
    
    logger.info("FastText analysis completed successfully!")

if __name__ == "__main__":
    main() 