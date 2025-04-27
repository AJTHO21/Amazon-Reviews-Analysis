import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import time
import torch

# Import the ErrorAnalyzer class
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis.error_analyzer import ErrorAnalyzer

class TestErrorAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock data
        self.test_data = pd.DataFrame({
            'text': [
                'This is a positive review',
                'This is a negative review',
                'This is an ambiguous review',
                'This is a very long review with many words and complex sentences',
                'This is a short review'
            ],
            'sentiment': [1, 0, 1, 0, 1]
        })
        
        # Mock model path
        self.model_path = os.path.join(self.test_dir, 'model')
        os.makedirs(self.model_path, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_initialization(self, mock_tokenizer, mock_model):
        """Test ErrorAnalyzer initialization."""
        # Set up mocks
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Check that the model and tokenizer were loaded
        mock_model.assert_called_once_with(self.model_path)
        mock_tokenizer.assert_called_once_with(self.model_path)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_predict_with_confidence(self, mock_tokenizer, mock_model):
        """Test predict_with_confidence method."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Set up mock tokenizer output
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Set up mock model output
        mock_model_instance.return_value = MagicMock(
            logits=torch.tensor([[0.8, 0.2]])
        )
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Make prediction
        label, confidence, probs = analyzer.predict_with_confidence('test review')
        
        # Check prediction output
        self.assertIsInstance(label, int)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(len(probs), 2)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_analyze_errors(self, mock_tokenizer, mock_model):
        """Test analyze_errors method."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Set up mock predictions
        def mock_predict(text):
            if 'positive' in text:
                return 1, 0.9, np.array([0.1, 0.9])
            elif 'negative' in text:
                return 0, 0.8, np.array([0.8, 0.2])
            else:
                return 1, 0.6, np.array([0.4, 0.6])
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        analyzer.predict_with_confidence = mock_predict
        
        # Analyze errors
        results = analyzer.analyze_errors(self.test_data)
        
        # Check results structure
        self.assertIn('correct', results)
        self.assertIn('incorrect', results)
        self.assertIn('low_confidence', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('error_patterns', results)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_analyze_error_patterns(self, mock_tokenizer, mock_model):
        """Test _analyze_error_patterns method."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create sample error data
        error_data = pd.DataFrame({
            'text': [
                'This is a very long review with many words',
                'This is another long review',
                'This is a short review'
            ],
            'true_label': [1, 0, 1],
            'predicted_label': [0, 1, 0],
            'confidence': [0.8, 0.7, 0.6]
        })
        
        # Analyze error patterns
        patterns = analyzer._analyze_error_patterns(error_data)
        
        # Check patterns structure
        self.assertIn('text_length_stats', patterns)
        self.assertIn('common_words', patterns)
        self.assertIn('sentiment_ambiguity', patterns)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    @patch('src.analysis.error_analyzer.plt')
    def test_visualize_results(self, mock_plt, mock_tokenizer, mock_model):
        """Test visualize_results method."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create sample results
        results = {
            'correct': pd.DataFrame({'text': ['correct1'], 'confidence': [0.9]}),
            'incorrect': pd.DataFrame({'text': ['incorrect1'], 'confidence': [0.8]}),
            'low_confidence': pd.DataFrame({'text': ['low_conf1'], 'confidence': [0.6]}),
            'confusion_matrix': np.array([[10, 5], [3, 12]]),
            'error_patterns': {
                'text_length_stats': {'mean': 100, 'std': 20},
                'common_words': ['word1', 'word2'],
                'sentiment_ambiguity': 0.3
            }
        }
        
        # Create visualization directory
        viz_dir = os.path.join(self.test_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualize results
        analyzer.visualize_results(results, viz_dir)
        
        # Check that the figures were created
        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.savefig.called)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_generate_report(self, mock_tokenizer, mock_model):
        """Test _generate_report method."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create sample results
        results = {
            'correct': pd.DataFrame({'text': ['correct1'], 'confidence': [0.9]}),
            'incorrect': pd.DataFrame({'text': ['incorrect1'], 'confidence': [0.8]}),
            'low_confidence': pd.DataFrame({'text': ['low_conf1'], 'confidence': [0.6]}),
            'confusion_matrix': np.array([[10, 5], [3, 12]]),
            'error_patterns': {
                'text_length_stats': {'mean': 100, 'std': 20},
                'common_words': ['word1', 'word2'],
                'sentiment_ambiguity': 0.3
            }
        }
        
        # Generate report
        report = analyzer._generate_report(results)
        
        # Check report content
        self.assertIsInstance(report, str)
        self.assertIn('Error Analysis Report', report)
        self.assertIn('Overall Statistics', report)
        self.assertIn('Error Patterns', report)
        self.assertIn('Common Words', report)
        self.assertIn('Text Length Statistics', report)

    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_empty_dataset(self, mock_tokenizer, mock_model):
        """Test analyze_errors with empty dataset."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create empty dataset
        empty_data = pd.DataFrame(columns=['text', 'sentiment'])
        
        # Analyze errors
        results = analyzer.analyze_errors(empty_data)
        
        # Check results structure
        self.assertIn('correct', results)
        self.assertIn('incorrect', results)
        self.assertIn('low_confidence', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('error_patterns', results)
        
        # Check that results are empty
        self.assertEqual(len(results['correct']), 0)
        self.assertEqual(len(results['incorrect']), 0)
        self.assertEqual(len(results['low_confidence']), 0)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_single_sample(self, mock_tokenizer, mock_model):
        """Test analyze_errors with single sample."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create single sample dataset
        single_data = pd.DataFrame({
            'text': ['This is a test review'],
            'sentiment': [1]
        })
        
        # Set up mock predictions
        def mock_predict(text):
            return 1, 0.9, np.array([0.1, 0.9])
        
        analyzer.predict_with_confidence = mock_predict
        
        # Analyze errors
        results = analyzer.analyze_errors(single_data)
        
        # Check results structure
        self.assertIn('correct', results)
        self.assertIn('incorrect', results)
        self.assertIn('low_confidence', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('error_patterns', results)
        
        # Check that results contain the single sample
        self.assertEqual(len(results['correct']) + len(results['incorrect']) + len(results['low_confidence']), 1)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_all_correct_predictions(self, mock_tokenizer, mock_model):
        """Test analyze_errors with all correct predictions."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create dataset
        data = pd.DataFrame({
            'text': ['Positive review 1', 'Positive review 2', 'Negative review 1'],
            'sentiment': [1, 1, 0]
        })
        
        # Set up mock predictions to always be correct
        def mock_predict(text):
            if 'Positive' in text:
                return 1, 0.9, np.array([0.1, 0.9])
            else:
                return 0, 0.9, np.array([0.9, 0.1])
        
        analyzer.predict_with_confidence = mock_predict
        
        # Analyze errors
        results = analyzer.analyze_errors(data)
        
        # Check that all predictions are correct
        self.assertEqual(len(results['correct']), 3)
        self.assertEqual(len(results['incorrect']), 0)
        self.assertEqual(len(results['low_confidence']), 0)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_all_incorrect_predictions(self, mock_tokenizer, mock_model):
        """Test analyze_errors with all incorrect predictions."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create dataset
        data = pd.DataFrame({
            'text': ['Positive review 1', 'Positive review 2', 'Negative review 1'],
            'sentiment': [1, 1, 0]
        })
        
        # Set up mock predictions to always be incorrect
        def mock_predict(text):
            if 'Positive' in text:
                return 0, 0.9, np.array([0.9, 0.1])
            else:
                return 1, 0.9, np.array([0.1, 0.9])
        
        analyzer.predict_with_confidence = mock_predict
        
        # Analyze errors
        results = analyzer.analyze_errors(data)
        
        # Check that all predictions are incorrect
        self.assertEqual(len(results['correct']), 0)
        self.assertEqual(len(results['incorrect']), 3)
        self.assertEqual(len(results['low_confidence']), 0)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_different_confidence_thresholds(self, mock_tokenizer, mock_model):
        """Test analyze_errors with different confidence thresholds."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create dataset
        data = pd.DataFrame({
            'text': ['Review 1', 'Review 2', 'Review 3'],
            'sentiment': [1, 0, 1]
        })
        
        # Set up mock predictions with varying confidence
        def mock_predict(text):
            if 'Review 1' in text:
                return 1, 0.95, np.array([0.05, 0.95])
            elif 'Review 2' in text:
                return 0, 0.85, np.array([0.85, 0.15])
            else:
                return 1, 0.55, np.array([0.45, 0.55])
        
        analyzer.predict_with_confidence = mock_predict
        
        # Test with high confidence threshold
        analyzer.confidence_threshold = 0.9
        results_high = analyzer.analyze_errors(data)
        
        # Test with low confidence threshold
        analyzer.confidence_threshold = 0.5
        results_low = analyzer.analyze_errors(data)
        
        # Check that high threshold results in more low confidence predictions
        self.assertGreaterEqual(
            len(results_high['low_confidence']),
            len(results_low['low_confidence'])
        )
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_special_characters_and_languages(self, mock_tokenizer, mock_model):
        """Test analyze_errors with special characters and different languages."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create dataset with special characters and different languages
        data = pd.DataFrame({
            'text': [
                'Review with special chars: !@#$%^&*()',
                'Review with emojis: 😊 😢 👍',
                'Review in Spanish: Excelente producto',
                'Review in French: Très bon produit',
                'Review with numbers: 12345'
            ],
            'sentiment': [1, 0, 1, 1, 0]
        })
        
        # Set up mock predictions
        def mock_predict(text):
            return 1, 0.8, np.array([0.2, 0.8])
        
        analyzer.predict_with_confidence = mock_predict
        
        # Analyze errors
        results = analyzer.analyze_errors(data)
        
        # Check that the analysis handles special characters and different languages
        self.assertIn('text_length_stats', results['error_patterns'])
        self.assertIn('common_words', results['error_patterns'])
        self.assertIn('sentiment_ambiguity', results['error_patterns'])

    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_large_dataset_performance(self, mock_tokenizer, mock_model):
        """Test analyze_errors with a large dataset to measure performance."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create a large dataset (1000 samples)
        large_data = pd.DataFrame({
            'text': [f'Review {i} with some content' for i in range(1000)],
            'sentiment': [1 if i % 2 == 0 else 0 for i in range(1000)]
        })
        
        # Set up mock predictions
        def mock_predict(text):
            return 1, 0.8, np.array([0.2, 0.8])
        
        analyzer.predict_with_confidence = mock_predict
        
        # Measure execution time
        start_time = time.time()
        results = analyzer.analyze_errors(large_data)
        execution_time = time.time() - start_time
        
        # Check that the analysis completes within a reasonable time (e.g., less than 10 seconds)
        self.assertLess(execution_time, 10.0)
        
        # Check that the results structure is maintained
        self.assertIn('correct', results)
        self.assertIn('incorrect', results)
        self.assertIn('low_confidence', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('error_patterns', results)
        
        # Check that all samples are processed
        total_samples = len(results['correct']) + len(results['incorrect']) + len(results['low_confidence'])
        self.assertEqual(total_samples, 1000)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_corrupted_model_file(self, mock_tokenizer, mock_model):
        """Test error handling with a corrupted model file."""
        # Set up mocks to simulate a corrupted model file
        mock_model.side_effect = Exception("Failed to load model")
        mock_tokenizer.return_value = MagicMock()
        
        # Initialize analyzer with a corrupted model file
        with self.assertRaises(Exception):
            ErrorAnalyzer(self.model_path)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_malformed_input_data(self, mock_tokenizer, mock_model):
        """Test error handling with malformed input data."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create malformed dataset (missing 'sentiment' column)
        malformed_data = pd.DataFrame({
            'text': ['Review 1', 'Review 2']
        })
        
        # Analyze errors with malformed data
        with self.assertRaises(KeyError):
            analyzer.analyze_errors(malformed_data)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_invalid_confidence_threshold(self, mock_tokenizer, mock_model):
        """Test error handling with invalid confidence threshold."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Set invalid confidence threshold
        analyzer.confidence_threshold = 1.5  # Should be between 0 and 1
        
        # Create dataset
        data = pd.DataFrame({
            'text': ['Review 1'],
            'sentiment': [1]
        })
        
        # Set up mock predictions
        def mock_predict(text):
            return 1, 0.8, np.array([0.2, 0.8])
        
        analyzer.predict_with_confidence = mock_predict
        
        # Analyze errors with invalid confidence threshold
        with self.assertRaises(ValueError):
            analyzer.analyze_errors(data)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    @patch('src.analysis.error_analyzer.plt')
    def test_visualize_empty_results(self, mock_plt, mock_tokenizer, mock_model):
        """Test visualization with empty results."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create empty results
        empty_results = {
            'correct': pd.DataFrame(columns=['text', 'confidence']),
            'incorrect': pd.DataFrame(columns=['text', 'confidence']),
            'low_confidence': pd.DataFrame(columns=['text', 'confidence']),
            'confusion_matrix': np.array([[0, 0], [0, 0]]),
            'error_patterns': {
                'text_length_stats': {'mean': 0, 'std': 0},
                'common_words': [],
                'sentiment_ambiguity': 0
            }
        }
        
        # Create visualization directory
        viz_dir = os.path.join(self.test_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualize empty results
        analyzer.visualize_results(empty_results, viz_dir)
        
        # Check that the figures were created
        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.savefig.called)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    @patch('src.analysis.error_analyzer.plt')
    def test_visualize_extreme_values(self, mock_plt, mock_tokenizer, mock_model):
        """Test visualization with extreme values in confusion matrix."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create results with extreme values
        extreme_results = {
            'correct': pd.DataFrame({'text': ['correct1'], 'confidence': [0.99]}),
            'incorrect': pd.DataFrame({'text': ['incorrect1'], 'confidence': [0.01]}),
            'low_confidence': pd.DataFrame({'text': ['low_conf1'], 'confidence': [0.5]}),
            'confusion_matrix': np.array([[1000, 0], [0, 1000]]),  # Extreme values
            'error_patterns': {
                'text_length_stats': {'mean': 1000, 'std': 500},  # Extreme values
                'common_words': ['word1', 'word2'],
                'sentiment_ambiguity': 0.9  # Extreme value
            }
        }
        
        # Create visualization directory
        viz_dir = os.path.join(self.test_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualize results with extreme values
        analyzer.visualize_results(extreme_results, viz_dir)
        
        # Check that the figures were created
        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.savefig.called)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_generate_report_missing_data(self, mock_tokenizer, mock_model):
        """Test report generation with missing data."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create results with missing data
        incomplete_results = {
            'correct': pd.DataFrame({'text': ['correct1'], 'confidence': [0.9]}),
            'incorrect': pd.DataFrame({'text': ['incorrect1'], 'confidence': [0.8]}),
            'low_confidence': pd.DataFrame({'text': ['low_conf1'], 'confidence': [0.6]}),
            'confusion_matrix': np.array([[10, 5], [3, 12]]),
            'error_patterns': {
                'text_length_stats': {'mean': 100, 'std': 20},
                'common_words': ['word1', 'word2'],
                # Missing sentiment_ambiguity
            }
        }
        
        # Generate report with missing data
        report = analyzer._generate_report(incomplete_results)
        
        # Check that the report is still generated
        self.assertIsInstance(report, str)
        self.assertIn('Error Analysis Report', report)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_generate_report_extreme_values(self, mock_tokenizer, mock_model):
        """Test report generation with extreme values."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create results with extreme values
        extreme_results = {
            'correct': pd.DataFrame({'text': ['correct1'], 'confidence': [0.99]}),
            'incorrect': pd.DataFrame({'text': ['incorrect1'], 'confidence': [0.01]}),
            'low_confidence': pd.DataFrame({'text': ['low_conf1'], 'confidence': [0.5]}),
            'confusion_matrix': np.array([[1000, 0], [0, 1000]]),  # Extreme values
            'error_patterns': {
                'text_length_stats': {'mean': 1000, 'std': 500},  # Extreme values
                'common_words': ['word1', 'word2'],
                'sentiment_ambiguity': 0.9  # Extreme value
            }
        }
        
        # Generate report with extreme values
        report = analyzer._generate_report(extreme_results)
        
        # Check that the report is generated
        self.assertIsInstance(report, str)
        self.assertIn('Error Analysis Report', report)
    
    @patch('src.analysis.error_analyzer.AutoModelForSequenceClassification.from_pretrained')
    @patch('src.analysis.error_analyzer.AutoTokenizer.from_pretrained')
    def test_generate_report_special_characters(self, mock_tokenizer, mock_model):
        """Test report generation with special characters and different languages."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize analyzer
        analyzer = ErrorAnalyzer(self.model_path)
        
        # Create results with special characters and different languages
        special_results = {
            'correct': pd.DataFrame({'text': ['Special chars: !@#$%^&*()'], 'confidence': [0.9]}),
            'incorrect': pd.DataFrame({'text': ['Emojis: 😊 😢 👍'], 'confidence': [0.8]}),
            'low_confidence': pd.DataFrame({'text': ['Spanish: Excelente producto'], 'confidence': [0.6]}),
            'confusion_matrix': np.array([[10, 5], [3, 12]]),
            'error_patterns': {
                'text_length_stats': {'mean': 100, 'std': 20},
                'common_words': ['!@#$', '😊', 'Excelente'],
                'sentiment_ambiguity': 0.3
            }
        }
        
        # Generate report with special characters and different languages
        report = analyzer._generate_report(special_results)
        
        # Check that the report is generated
        self.assertIsInstance(report, str)
        self.assertIn('Error Analysis Report', report)

if __name__ == '__main__':
    unittest.main() 