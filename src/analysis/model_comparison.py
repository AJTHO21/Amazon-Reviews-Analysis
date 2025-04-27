import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import fasttext
import time
import psutil
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
import shap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))

# Custom Dataset for LSTM
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ModelEvaluator:
    def __init__(self, model_path, model_name, device=None, model_params=None):
        """
        Initialize the model evaluator.
        
        Args:
            model_path (str): Path to the saved model
            model_name (str): Name of the model (e.g., 'BERT', 'FastText', 'DistilBERT', 'RoBERTa', 'LSTM')
            device (torch.device, optional): Device to use for computation
            model_params (dict, optional): Parameters for model initialization
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_params = model_params or {}
        
        if model_name.lower() == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
        elif model_name.lower() == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
        elif model_name.lower() == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
        elif model_name.lower() == 'fasttext':
            self.model = fasttext.load_model(model_path)
        elif model_name.lower() == 'lstm':
            # Initialize LSTM model with parameters
            vocab_size = self.model_params.get('vocab_size', 30522)  # Default BERT vocab size
            embedding_dim = self.model_params.get('embedding_dim', 300)
            hidden_dim = self.model_params.get('hidden_dim', 256)
            output_dim = self.model_params.get('output_dim', 2)
            n_layers = self.model_params.get('n_layers', 2)
            dropout = self.model_params.get('dropout', 0.5)
            
            self.model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(self.device)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            
            # Use BERT tokenizer for LSTM as well
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            raise ValueError(f"Model type {model_name} not supported yet")
    
    def evaluate(self, test_data, batch_size=32):
        """
        Evaluate model performance on test data.
        
        Args:
            test_data (pd.DataFrame): Test dataset
            batch_size (int): Batch size for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = []
        true_labels = []
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Create batches
        n_batches = len(test_data) // batch_size + (1 if len(test_data) % batch_size != 0 else 0)
        
        for i in tqdm(range(n_batches), desc=f"Evaluating {self.model_name}"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_data))
            batch = test_data.iloc[start_idx:end_idx]
            
            batch_predictions = self._predict_batch(batch)
            predictions.extend(batch_predictions)
            true_labels.extend(batch['sentiment'].tolist())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        
        # Calculate performance metrics
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'inference_time': end_time - start_time,
            'memory_usage': end_memory - start_memory
        }
        
        return metrics
    
    def _predict_batch(self, batch):
        """
        Make predictions for a batch of texts.
        
        Args:
            batch (pd.DataFrame): Batch of test data
            
        Returns:
            list: Predictions
        """
        if self.model_name.lower() in ['bert', 'distilbert', 'roberta']:
            texts = batch['text'].tolist()
            inputs = self.tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                max_length=256,
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
            
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            return predictions.tolist()
        
        elif self.model_name.lower() == 'fasttext':
            texts = batch['text'].tolist()
            predictions = []
            for text in texts:
                pred = self.model.predict(text, k=1)
                # FastText returns labels with '__label__' prefix
                label = int(pred[0][0].replace('__label__', ''))
                predictions.append(label)
            return predictions
        
        elif self.model_name.lower() == 'lstm':
            dataset = TextDataset(batch['text'].tolist(), batch['sentiment'].tolist(), self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=len(dataset))
            
            predictions = []
            with torch.no_grad():
                for batch_data in dataloader:
                    input_ids = batch_data['input_ids'].to(self.device)
                    outputs = self.model(input_ids)
                    predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            
            return predictions
        
        return []

class ModelComparison:
    def __init__(self, models_config):
        """
        Initialize model comparison.
        
        Args:
            models_config (list): List of dictionaries containing model configurations
        """
        self.models_config = models_config
        self.evaluators = {}
        self.results = {}
    
    def load_models(self):
        """Load all models specified in the configuration."""
        for config in self.models_config:
            model_name = config['name']
            model_path = config['path']
            model_params = config.get('params', {})
            self.evaluators[model_name] = ModelEvaluator(model_path, model_name, model_params=model_params)
            logger.info(f"Loaded model: {model_name}")
    
    def compare_models(self, test_data, batch_size=32):
        """
        Compare performance of all models.
        
        Args:
            test_data (pd.DataFrame): Test dataset
            batch_size (int): Batch size for evaluation
        """
        for model_name, evaluator in self.evaluators.items():
            logger.info(f"Evaluating {model_name}...")
            metrics = evaluator.evaluate(test_data, batch_size)
            self.results[model_name] = metrics
            logger.info(f"Results for {model_name}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.4f}")
    
    def cross_validate(self, data, n_splits=5, batch_size=32):
        """
        Perform cross-validation for all models.
        
        Args:
            data (pd.DataFrame): Dataset for cross-validation
            n_splits (int): Number of folds
            batch_size (int): Batch size for evaluation
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = {model_name: [] for model_name in self.evaluators.keys()}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            logger.info(f"Fold {fold+1}/{n_splits}")
            val_data = data.iloc[val_idx]
            
            for model_name, evaluator in self.evaluators.items():
                metrics = evaluator.evaluate(val_data, batch_size)
                cv_results[model_name].append(metrics)
        
        # Calculate average metrics across folds
        for model_name, fold_results in cv_results.items():
            avg_metrics = {}
            for metric in fold_results[0].keys():
                if isinstance(fold_results[0][metric], float):
                    avg_metrics[metric] = np.mean([r[metric] for r in fold_results])
            
            self.results[model_name].update({
                'cv_accuracy': avg_metrics['accuracy'],
                'cv_f1': avg_metrics['f1'],
                'cv_std_accuracy': np.std([r['accuracy'] for r in fold_results]),
                'cv_std_f1': np.std([r['f1'] for r in fold_results])
            })
    
    def hyperparameter_tuning(self, model_name, data, param_space, n_trials=50):
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name (str): Name of the model to tune
            data (pd.DataFrame): Dataset for tuning
            param_space (dict): Parameter space to explore
            n_trials (int): Number of trials
        """
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Create model with sampled parameters
            evaluator = ModelEvaluator(
                self.models_config[model_name]['path'],
                model_name,
                model_params=params
            )
            
            # Evaluate model
            metrics = evaluator.evaluate(data)
            return metrics['f1']  # Optimize for F1 score
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Update model with best parameters
        best_params = study.best_params
        self.models_config[model_name]['params'] = best_params
        self.evaluators[model_name] = ModelEvaluator(
            self.models_config[model_name]['path'],
            model_name,
            model_params=best_params
        )
        
        return best_params, study.best_value
    
    def interpret_model(self, model_name, data, n_samples=100):
        """
        Generate interpretability analysis for a model.
        
        Args:
            model_name (str): Name of the model to interpret
            data (pd.DataFrame): Dataset for interpretation
            n_samples (int): Number of samples to analyze
        """
        evaluator = self.evaluators[model_name]
        
        if model_name.lower() in ['bert', 'distilbert', 'roberta']:
            # Use SHAP for transformer models
            explainer = shap.Explainer(evaluator.model, evaluator.tokenizer)
            shap_values = explainer(data['text'].head(n_samples))
            
            # Save SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, data['text'].head(n_samples), show=False)
            plt.savefig(f"results/model_comparison/{model_name}_shap_summary.png")
            plt.close()
            
            return shap_values
        else:
            logger.warning(f"Interpretability analysis not supported for {model_name}")
            return None
    
    def visualize_comparison(self, save_dir=None):
        """
        Visualize model comparison results.
        
        Args:
            save_dir (str, optional): Directory to save visualizations
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(self.results.keys())
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            values = [self.results[model_name][metric] for metric in metrics]
            plt.bar(x + i * width, values, width, label=model_name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x + width * (len(model_names) - 1) / 2, metrics)
        plt.legend()
        
        if save_dir:
            plt.savefig(save_dir / 'model_comparison.png')
        plt.close()
        
        # Create performance comparison plot
        plt.figure(figsize=(10, 6))
        performance_metrics = ['inference_time', 'memory_usage']
        x = np.arange(len(performance_metrics))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            values = [self.results[model_name][metric] for metric in performance_metrics]
            plt.bar(x + i * width, values, width, label=model_name)
        
        plt.xlabel('Performance Metrics')
        plt.ylabel('Value')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * (len(model_names) - 1) / 2, ['Inference Time (s)', 'Memory Usage (MB)'])
        plt.legend()
        
        if save_dir:
            plt.savefig(save_dir / 'performance_comparison.png')
        plt.close()
        
        # Create cross-validation plot if available
        if any('cv_accuracy' in results for results in self.results.values()):
            plt.figure(figsize=(10, 6))
            x = np.arange(len(model_names))
            width = 0.35
            
            accuracies = [self.results[model_name]['cv_accuracy'] for model_name in model_names]
            std_accuracies = [self.results[model_name]['cv_std_accuracy'] for model_name in model_names]
            
            plt.bar(x, accuracies, width, yerr=std_accuracies, label='Cross-validation Accuracy')
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Cross-validation Results')
            plt.xticks(x, model_names)
            plt.legend()
            
            if save_dir:
                plt.savefig(save_dir / 'cross_validation.png')
            plt.close()
        
        # Save detailed results
        if save_dir:
            with open(save_dir / 'model_comparison_results.json', 'w') as f:
                json.dump(self.results, f, indent=4)
    
    def generate_report(self, save_dir=None):
        """
        Generate a detailed comparison report.
        
        Args:
            save_dir (str, optional): Directory to save the report
        """
        report = []
        report.append("Model Comparison Report")
        report.append("=" * 50)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall comparison
        report.append("Overall Comparison")
        report.append("-" * 20)
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            report.append(f"\n{metric.upper()}:")
            for model_name, results in self.results.items():
                report.append(f"{model_name}: {results[metric]:.4f}")
        
        # Performance comparison
        report.append("\nPerformance Comparison")
        report.append("-" * 20)
        for model_name, results in self.results.items():
            report.append(f"\n{model_name}:")
            report.append(f"Inference Time: {results['inference_time']:.4f} seconds")
            report.append(f"Memory Usage: {results['memory_usage']:.2f} MB")
        
        # Cross-validation results if available
        if any('cv_accuracy' in results for results in self.results.values()):
            report.append("\nCross-validation Results")
            report.append("-" * 20)
            for model_name, results in self.results.items():
                if 'cv_accuracy' in results:
                    report.append(f"\n{model_name}:")
                    report.append(f"CV Accuracy: {results['cv_accuracy']:.4f} ± {results['cv_std_accuracy']:.4f}")
                    report.append(f"CV F1: {results['cv_f1']:.4f} ± {results['cv_std_f1']:.4f}")
        
        # Best model for each metric
        report.append("\nBest Model for Each Metric")
        report.append("-" * 20)
        for metric in metrics:
            best_model = max(self.results.items(), key=lambda x: x[1][metric])[0]
            best_score = self.results[best_model][metric]
            report.append(f"{metric.upper()}: {best_model} ({best_score:.4f})")
        
        report_text = "\n".join(report)
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / 'model_comparison_report.txt', 'w') as f:
                f.write(report_text)
        
        return report_text

def main():
    # Example usage
    models_config = [
        {
            'name': 'BERT',
            'path': 'models/bert_model'
        },
        {
            'name': 'FastText',
            'path': 'models/fasttext_model.bin'
        },
        {
            'name': 'DistilBERT',
            'path': 'models/distilbert_model'
        },
        {
            'name': 'RoBERTa',
            'path': 'models/roberta_model'
        },
        {
            'name': 'LSTM',
            'path': 'models/lstm_model.pt',
            'params': {
                'vocab_size': 30522,
                'embedding_dim': 300,
                'hidden_dim': 256,
                'output_dim': 2,
                'n_layers': 2,
                'dropout': 0.5
            }
        }
    ]
    
    # Initialize comparison
    comparison = ModelComparison(models_config)
    comparison.load_models()
    
    # Load test data
    test_file = Path("data/raw/test.ft.txt")
    test_data = pd.read_csv(test_file, sep=' ', header=None, names=['label', 'text'])
    test_data['sentiment'] = test_data['label'].map({'1': 0, '2': 1})
    
    # Take a subset for comparison
    test_subset = test_data.sample(n=1000, random_state=42)
    
    # Compare models
    comparison.compare_models(test_subset)
    
    # Perform cross-validation
    comparison.cross_validate(test_data.sample(n=5000, random_state=42))
    
    # Hyperparameter tuning for LSTM
    lstm_param_space = {
        'embedding_dim': (100, 500),
        'hidden_dim': (128, 512),
        'n_layers': (1, 3),
        'dropout': (0.1, 0.7)
    }
    best_params, best_score = comparison.hyperparameter_tuning('LSTM', test_subset, lstm_param_space, n_trials=20)
    logger.info(f"Best LSTM parameters: {best_params}")
    logger.info(f"Best LSTM F1 score: {best_score:.4f}")
    
    # Interpret models
    for model_name in ['BERT', 'DistilBERT', 'RoBERTa']:
        comparison.interpret_model(model_name, test_subset, n_samples=50)
    
    # Visualize and save results
    comparison.visualize_comparison(save_dir="results/model_comparison")
    comparison.generate_report(save_dir="results/model_comparison")

if __name__ == "__main__":
    main() 