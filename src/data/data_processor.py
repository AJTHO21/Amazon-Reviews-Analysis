import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AmazonReviewsProcessor:
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the data processor.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_fasttext_data(self, file_path: str) -> pd.DataFrame:
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
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the reviews data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        logger.info("Preprocessing data")
        
        # Convert labels to binary (1 for negative, 2 for positive)
        df['sentiment'] = df['label'].map({'1': 0, '2': 1})
        
        # Basic text cleaning
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].str.replace(r'[^\w\s]', ' ', regex=True)
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
        
        logger.info("Preprocessing completed")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to CSV.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            filename (str): Output filename
        """
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    # Example usage
    processor = AmazonReviewsProcessor()
    
    # Load and process training data
    train_df = processor.load_fasttext_data("train.ft.txt")
    train_df = processor.preprocess_data(train_df)
    processor.save_processed_data(train_df, "processed_train.csv")
    
    # Load and process test data
    test_df = processor.load_fasttext_data("test.ft.txt")
    test_df = processor.preprocess_data(test_df)
    processor.save_processed_data(test_df, "processed_test.csv") 