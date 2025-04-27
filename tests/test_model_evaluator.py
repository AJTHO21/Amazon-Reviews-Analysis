import unittest
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the ModelEvaluator class
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis.model_comparison import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock data
        self.test_data = pd.DataFrame({
            'text': ['This is a positive review', 'This is a negative review'],
            'sentiment': [1, 0]
        })
        
        # Mock model paths
        self.bert_path = os.path.join(self.test_dir, 'bert_model')
        self.fasttext_path = os.path.join(self.test_dir, 'fasttext_model.bin')
        self.distilbert_path = os.path.join(self.test_dir, 'distilbert_model')
        self.roberta_path = os.path.join(self.test_dir, 'roberta_model')
        self.lstm_path = os.path.join(self.test_dir, 'lstm_model.pt')
        
        # Create mock model files
        os.makedirs(self.bert_path, exist_ok=True)
        os.makedirs(self.distilbert_path, exist_ok=True)
        os.makedirs(self.roberta_path, exist_ok=True)
        
        # Create a dummy LSTM model file
        dummy_model = torch.nn.Linear(10, 2)
        torch.save(dummy_model.state_dict(), self.lstm_path)
        
        # Create a dummy FastText model file
        with open(self.fasttext_path, 'w') as f:
            f.write('dummy fasttext model')
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
    
    @patch('src.analysis.model_comparison.BertTokenizer.from_pretrained')
    @patch('src.analysis.model_comparison.BertForSequenceClassification.from_pretrained')
    def test_bert_initialization(self, mock_model, mock_tokenizer):
        """Test BERT model initialization."""
        # Set up mocks
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self.bert_path, 'BERT')
        
        # Check that the model and tokenizer were initialized
        mock_tokenizer.assert_called_once_with(self.bert_path)
        mock_model.assert_called_once_with(self.bert_path)
        
        # Check that the model was moved to the correct device
        mock_model.return_value.to.assert_called_once()
        mock_model.return_value.eval.assert_called_once()
    
    @patch('src.analysis.model_comparison.fasttext.load_model')
    def test_fasttext_initialization(self, mock_load_model):
        """Test FastText model initialization."""
        # Set up mock
        mock_load_model.return_value = MagicMock()
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self.fasttext_path, 'FastText')
        
        # Check that the model was loaded
        mock_load_model.assert_called_once_with(self.fasttext_path)
    
    @patch('src.analysis.model_comparison.DistilBertTokenizer.from_pretrained')
    @patch('src.analysis.model_comparison.DistilBertForSequenceClassification.from_pretrained')
    def test_distilbert_initialization(self, mock_model, mock_tokenizer):
        """Test DistilBERT model initialization."""
        # Set up mocks
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self.distilbert_path, 'DistilBERT')
        
        # Check that the model and tokenizer were initialized
        mock_tokenizer.assert_called_once_with(self.distilbert_path)
        mock_model.assert_called_once_with(self.distilbert_path)
        
        # Check that the model was moved to the correct device
        mock_model.return_value.to.assert_called_once()
        mock_model.return_value.eval.assert_called_once()
    
    @patch('src.analysis.model_comparison.RobertaTokenizer.from_pretrained')
    @patch('src.analysis.model_comparison.RobertaForSequenceClassification.from_pretrained')
    def test_roberta_initialization(self, mock_model, mock_tokenizer):
        """Test RoBERTa model initialization."""
        # Set up mocks
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self.roberta_path, 'RoBERTa')
        
        # Check that the model and tokenizer were initialized
        mock_tokenizer.assert_called_once_with(self.roberta_path)
        mock_model.assert_called_once_with(self.roberta_path)
        
        # Check that the model was moved to the correct device
        mock_model.return_value.to.assert_called_once()
        mock_model.return_value.eval.assert_called_once()
    
    @patch('src.analysis.model_comparison.BertTokenizer.from_pretrained')
    def test_lstm_initialization(self, mock_tokenizer):
        """Test LSTM model initialization."""
        # Set up mock
        mock_tokenizer.return_value = MagicMock()
        
        # Initialize evaluator with LSTM parameters
        model_params = {
            'vocab_size': 1000,
            'embedding_dim': 100,
            'hidden_dim': 128,
            'output_dim': 2,
            'n_layers': 1,
            'dropout': 0.1
        }
        
        evaluator = ModelEvaluator(self.lstm_path, 'LSTM', model_params=model_params)
        
        # Check that the tokenizer was initialized
        mock_tokenizer.assert_called_once_with('bert-base-uncased')
        
        # Check that the model was loaded
        self.assertIsNotNone(evaluator.model)
        self.assertTrue(isinstance(evaluator.model, torch.nn.Module))
    
    def test_invalid_model_type(self):
        """Test that an invalid model type raises an error."""
        with self.assertRaises(ValueError):
            ModelEvaluator(self.test_dir, 'InvalidModel')
    
    @patch('src.analysis.model_comparison.ModelEvaluator._predict_batch')
    def test_evaluate(self, mock_predict_batch):
        """Test the evaluate method."""
        # Set up mock
        mock_predict_batch.return_value = [1, 0]
        
        # Create a mock evaluator
        evaluator = ModelEvaluator(self.bert_path, 'BERT')
        evaluator._predict_batch = mock_predict_batch
        
        # Evaluate the model
        metrics = evaluator.evaluate(self.test_data)
        
        # Check that the metrics were calculated correctly
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('inference_time', metrics)
        self.assertIn('memory_usage', metrics)
        
        # Check that the predict_batch method was called
        mock_predict_batch.assert_called_once()
    
    @patch('src.analysis.model_comparison.ModelEvaluator.tokenizer')
    def test_bert_predict_batch(self, mock_tokenizer):
        """Test BERT batch prediction."""
        # Set up mock
        mock_tokenizer.batch_encode_plus.return_value = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        
        # Create a mock model
        mock_model = MagicMock()
        mock_model.return_value.logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        
        # Create evaluator with mock model
        evaluator = ModelEvaluator(self.bert_path, 'BERT')
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        
        # Predict batch
        predictions = evaluator._predict_batch(self.test_data)
        
        # Check that the predictions are correct
        self.assertEqual(predictions, [1, 0])
        
        # Check that the tokenizer and model were called correctly
        mock_tokenizer.batch_encode_plus.assert_called_once()
        mock_model.assert_called_once()
    
    @patch('src.analysis.model_comparison.ModelEvaluator.model')
    def test_fasttext_predict_batch(self, mock_model):
        """Test FastText batch prediction."""
        # Set up mock
        mock_model.predict.side_effect = [
            (['__label__1'], [0.9]),
            (['__label__0'], [0.8])
        ]
        
        # Create evaluator with mock model
        evaluator = ModelEvaluator(self.fasttext_path, 'FastText')
        evaluator.model = mock_model
        
        # Predict batch
        predictions = evaluator._predict_batch(self.test_data)
        
        # Check that the predictions are correct
        self.assertEqual(predictions, [1, 0])
        
        # Check that the model was called correctly
        self.assertEqual(mock_model.predict.call_count, 2)
    
    @patch('src.analysis.model_comparison.ModelEvaluator.tokenizer')
    def test_lstm_predict_batch(self, mock_tokenizer):
        """Test LSTM batch prediction."""
        # Set up mock
        mock_tokenizer.encode_plus.return_value = {
            'input_ids': torch.tensor([1, 2, 3])
        }
        
        # Create a mock model
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        
        # Create evaluator with mock model
        evaluator = ModelEvaluator(self.lstm_path, 'LSTM')
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        
        # Predict batch
        predictions = evaluator._predict_batch(self.test_data)
        
        # Check that the predictions are correct
        self.assertEqual(predictions, [1, 0])
        
        # Check that the model was called correctly
        self.assertEqual(mock_model.call_count, 1)

if __name__ == '__main__':
    unittest.main() 