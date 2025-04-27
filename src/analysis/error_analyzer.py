import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorAnalyzer:
    def __init__(self, model_path, device=None):
        """
        Initialize the error analyzer.
        
        Args:
            model_path (str): Path to the saved BERT model
            device (torch.device, optional): Device to use for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
    
    def predict_with_confidence(self, text, max_length=256):
        """
        Get prediction and confidence score for a text.
        
        Args:
            text (str): Input text
            max_length (int): Maximum sequence length
            
        Returns:
            tuple: (prediction, confidence, probabilities)
        """
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        return prediction, confidence, probabilities[0].cpu().numpy()
    
    def analyze_errors(self, test_data, confidence_threshold=0.8):
        """
        Analyze model errors and identify challenging cases.
        
        Args:
            test_data (pd.DataFrame): Test dataset
            confidence_threshold (float): Threshold for high-confidence predictions
            
        Returns:
            dict: Analysis results
        """
        results = {
            'correct_predictions': [],
            'incorrect_predictions': [],
            'low_confidence_predictions': [],
            'confusion_matrix': None,
            'error_patterns': {}
        }
        
        predictions = []
        true_labels = []
        confidences = []
        
        for _, row in tqdm(test_data.iterrows(), desc="Analyzing predictions"):
            text = row['text']
            true_label = row['sentiment']
            
            pred, conf, probs = self.predict_with_confidence(text)
            predictions.append(pred)
            true_labels.append(true_label)
            confidences.append(conf)
            
            result = {
                'text': text,
                'true_label': true_label,
                'predicted_label': pred,
                'confidence': conf,
                'probabilities': probs
            }
            
            if pred == true_label:
                results['correct_predictions'].append(result)
            else:
                results['incorrect_predictions'].append(result)
            
            if conf < confidence_threshold:
                results['low_confidence_predictions'].append(result)
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        results['confusion_matrix'] = cm
        
        # Analyze error patterns
        error_texts = [r['text'] for r in results['incorrect_predictions']]
        results['error_patterns'] = self._analyze_error_patterns(error_texts)
        
        return results
    
    def _analyze_error_patterns(self, error_texts):
        """
        Analyze patterns in misclassified texts.
        
        Args:
            error_texts (list): List of misclassified texts
            
        Returns:
            dict: Pattern analysis results
        """
        patterns = {
            'length_stats': {
                'mean': np.mean([len(text.split()) for text in error_texts]),
                'std': np.std([len(text.split()) for text in error_texts]),
                'min': min([len(text.split()) for text in error_texts]),
                'max': max([len(text.split()) for text in error_texts])
            },
            'common_words': self._get_common_words(error_texts),
            'sentiment_ambiguity': self._analyze_sentiment_ambiguity(error_texts)
        }
        
        return patterns
    
    def _get_common_words(self, texts, top_n=20):
        """
        Get most common words in the texts.
        
        Args:
            texts (list): List of texts
            top_n (int): Number of top words to return
            
        Returns:
            list: List of (word, count) tuples
        """
        words = []
        for text in texts:
            words.extend(text.lower().split())
        
        word_counts = pd.Series(words).value_counts()
        return list(word_counts.head(top_n).items())
    
    def _analyze_sentiment_ambiguity(self, texts):
        """
        Analyze sentiment ambiguity in texts.
        
        Args:
            texts (list): List of texts
            
        Returns:
            dict: Ambiguity analysis results
        """
        # This is a simple implementation. You can make it more sophisticated
        # by using sentiment lexicons or other NLP tools
        ambiguous_patterns = {
            'negation_words': ['not', "don't", 'never', 'no'],
            'contrast_words': ['but', 'however', 'although', 'yet'],
            'uncertainty_words': ['maybe', 'might', 'could', 'perhaps']
        }
        
        ambiguity_stats = {
            pattern_type: sum(1 for text in texts if any(word in text.lower() for word in words))
            for pattern_type, words in ambiguous_patterns.items()
        }
        
        return ambiguity_stats
    
    def visualize_results(self, results, save_dir=None):
        """
        Visualize analysis results.
        
        Args:
            results (dict): Analysis results
            save_dir (str, optional): Directory to save visualizations
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title('Confusion Matrix')
        if save_dir:
            plt.savefig(save_dir / 'confusion_matrix.png')
        plt.close()
        
        # Plot confidence distribution
        confidences = [r['confidence'] for r in results['correct_predictions'] + results['incorrect_predictions']]
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, alpha=0.7)
        plt.title('Distribution of Prediction Confidences')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        if save_dir:
            plt.savefig(save_dir / 'confidence_distribution.png')
        plt.close()
        
        # Save error analysis report
        if save_dir:
            report = self._generate_report(results)
            with open(save_dir / 'error_analysis_report.txt', 'w') as f:
                f.write(report)
    
    def _generate_report(self, results):
        """
        Generate a detailed error analysis report.
        
        Args:
            results (dict): Analysis results
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append("Error Analysis Report")
        report.append("=" * 50)
        
        # Overall statistics
        total = len(results['correct_predictions']) + len(results['incorrect_predictions'])
        accuracy = len(results['correct_predictions']) / total
        report.append(f"\nOverall Statistics:")
        report.append(f"Total samples: {total}")
        report.append(f"Accuracy: {accuracy:.2%}")
        report.append(f"Number of errors: {len(results['incorrect_predictions'])}")
        
        # Error patterns
        report.append("\nError Patterns:")
        for pattern_type, count in results['error_patterns']['sentiment_ambiguity'].items():
            report.append(f"{pattern_type}: {count} occurrences")
        
        # Common words in errors
        report.append("\nMost Common Words in Errors:")
        for word, count in results['error_patterns']['common_words']:
            report.append(f"{word}: {count}")
        
        # Length statistics
        report.append("\nText Length Statistics in Errors:")
        for stat, value in results['error_patterns']['length_stats'].items():
            report.append(f"{stat}: {value:.2f}")
        
        return "\n".join(report)

def main():
    # Example usage
    model_path = Path("models/bert_model")
    analyzer = ErrorAnalyzer(str(model_path))
    
    # Load test data
    test_file = Path("data/raw/test.ft.txt")
    test_data = pd.read_csv(test_file, sep=' ', header=None, names=['label', 'text'])
    test_data['sentiment'] = test_data['label'].map({'1': 0, '2': 1})
    
    # Take a subset for analysis
    test_subset = test_data.sample(n=1000, random_state=42)
    
    # Analyze errors
    results = analyzer.analyze_errors(test_subset)
    
    # Visualize results
    analyzer.visualize_results(results, save_dir="results/error_analysis")

if __name__ == "__main__":
    main() 