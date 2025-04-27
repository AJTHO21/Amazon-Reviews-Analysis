import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the ModelComparison class
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis.model_comparison import ModelComparison, ModelEvaluator

class TestModelComparison(unittest.TestCase):
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
        
        # Create mock model files
        os.makedirs(self.bert_path, exist_ok=True)
        
        # Create a dummy FastText model file
        with open(self.fasttext_path, 'w') as f:
            f.write('dummy fasttext model')
        
        # Create mock models config
        self.models_config = [
            {
                'name': 'BERT',
                'path': self.bert_path
            },
            {
                'name': 'FastText',
                'path': self.fasttext_path
            }
        ]
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
    
    @patch('src.analysis.model_comparison.ModelEvaluator')
    def test_initialization(self, mock_evaluator):
        """Test ModelComparison initialization."""
        # Set up mock
        mock_evaluator.return_value = MagicMock()
        
        # Initialize comparison
        comparison = ModelComparison(self.models_config)
        
        # Check that the evaluators were initialized
        self.assertEqual(len(comparison.evaluators), 2)
        self.assertIn('BERT', comparison.evaluators)
        self.assertIn('FastText', comparison.evaluators)
        
        # Check that the evaluator was called with the correct arguments
        mock_evaluator.assert_any_call(self.bert_path, 'BERT', model_params={})
        mock_evaluator.assert_any_call(self.fasttext_path, 'FastText', model_params={})
    
    @patch('src.analysis.model_comparison.ModelEvaluator')
    def test_load_models(self, mock_evaluator):
        """Test load_models method."""
        # Set up mock
        mock_evaluator.return_value = MagicMock()
        
        # Initialize comparison
        comparison = ModelComparison(self.models_config)
        
        # Load models
        comparison.load_models()
        
        # Check that the evaluators were initialized
        self.assertEqual(len(comparison.evaluators), 2)
        self.assertIn('BERT', comparison.evaluators)
        self.assertIn('FastText', comparison.evaluators)
    
    @patch('src.analysis.model_comparison.ModelEvaluator')
    def test_compare_models(self, mock_evaluator):
        """Test compare_models method."""
        # Set up mock
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        
        # Set up mock evaluate method
        mock_evaluator_instance.evaluate.return_value = {
            'accuracy': 0.9,
            'precision': 0.85,
            'recall': 0.9,
            'f1': 0.87,
            'inference_time': 0.1,
            'memory_usage': 100
        }
        
        # Initialize comparison
        comparison = ModelComparison(self.models_config)
        comparison.load_models()
        
        # Compare models
        comparison.compare_models(self.test_data)
        
        # Check that the results were stored
        self.assertEqual(len(comparison.results), 2)
        self.assertIn('BERT', comparison.results)
        self.assertIn('FastText', comparison.results)
        
        # Check that the evaluate method was called for each model
        self.assertEqual(mock_evaluator_instance.evaluate.call_count, 2)
    
    @patch('src.analysis.model_comparison.ModelEvaluator')
    def test_cross_validate(self, mock_evaluator):
        """Test cross_validate method."""
        # Set up mock
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        
        # Set up mock evaluate method
        mock_evaluator_instance.evaluate.return_value = {
            'accuracy': 0.9,
            'precision': 0.85,
            'recall': 0.9,
            'f1': 0.87,
            'inference_time': 0.1,
            'memory_usage': 100
        }
        
        # Initialize comparison
        comparison = ModelComparison(self.models_config)
        comparison.load_models()
        
        # Compare models first
        comparison.compare_models(self.test_data)
        
        # Perform cross-validation
        comparison.cross_validate(self.test_data, n_splits=2)
        
        # Check that the cross-validation results were stored
        for model_name in comparison.results:
            self.assertIn('cv_accuracy', comparison.results[model_name])
            self.assertIn('cv_f1', comparison.results[model_name])
            self.assertIn('cv_std_accuracy', comparison.results[model_name])
            self.assertIn('cv_std_f1', comparison.results[model_name])
    
    @patch('src.analysis.model_comparison.ModelEvaluator')
    @patch('src.analysis.model_comparison.optuna.create_study')
    def test_hyperparameter_tuning(self, mock_create_study, mock_evaluator):
        """Test hyperparameter_tuning method."""
        # Set up mock
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        
        # Set up mock evaluate method
        mock_evaluator_instance.evaluate.return_value = {
            'accuracy': 0.9,
            'precision': 0.85,
            'recall': 0.9,
            'f1': 0.87,
            'inference_time': 0.1,
            'memory_usage': 100
        }
        
        # Set up mock study
        mock_study = MagicMock()
        mock_study.best_params = {'param1': 0.5, 'param2': 10}
        mock_study.best_value = 0.9
        mock_create_study.return_value = mock_study
        
        # Initialize comparison
        comparison = ModelComparison(self.models_config)
        comparison.load_models()
        
        # Define parameter space
        param_space = {
            'param1': (0.1, 1.0),
            'param2': (1, 20)
        }
        
        # Perform hyperparameter tuning
        best_params, best_score = comparison.hyperparameter_tuning('BERT', self.test_data, param_space, n_trials=2)
        
        # Check that the best parameters were returned
        self.assertEqual(best_params, {'param1': 0.5, 'param2': 10})
        self.assertEqual(best_score, 0.9)
        
        # Check that the study was created and optimized
        mock_create_study.assert_called_once()
        mock_study.optimize.assert_called_once()
    
    @patch('src.analysis.model_comparison.ModelEvaluator')
    @patch('src.analysis.model_comparison.shap.Explainer')
    def test_interpret_model(self, mock_explainer, mock_evaluator):
        """Test interpret_model method."""
        # Set up mock
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        
        # Set up mock model and tokenizer
        mock_evaluator_instance.model = MagicMock()
        mock_evaluator_instance.tokenizer = MagicMock()
        
        # Set up mock explainer
        mock_explainer_instance = MagicMock()
        mock_explainer.return_value = mock_explainer_instance
        
        # Initialize comparison
        comparison = ModelComparison(self.models_config)
        comparison.load_models()
        
        # Interpret model
        shap_values = comparison.interpret_model('BERT', self.test_data, n_samples=1)
        
        # Check that the explainer was created
        mock_explainer.assert_called_once_with(mock_evaluator_instance.model, mock_evaluator_instance.tokenizer)
        
        # Check that the explainer was called with the data
        mock_explainer_instance.assert_called_once()
    
    @patch('src.analysis.model_comparison.ModelEvaluator')
    @patch('src.analysis.model_comparison.plt')
    def test_visualize_comparison(self, mock_plt, mock_evaluator):
        """Test visualize_comparison method."""
        # Set up mock
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        
        # Set up mock evaluate method
        mock_evaluator_instance.evaluate.return_value = {
            'accuracy': 0.9,
            'precision': 0.85,
            'recall': 0.9,
            'f1': 0.87,
            'inference_time': 0.1,
            'memory_usage': 100
        }
        
        # Initialize comparison
        comparison = ModelComparison(self.models_config)
        comparison.load_models()
        
        # Compare models
        comparison.compare_models(self.test_data)
        
        # Create visualization directory
        viz_dir = os.path.join(self.test_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualize comparison
        comparison.visualize_comparison(save_dir=viz_dir)
        
        # Check that the figures were created
        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.savefig.called)
    
    @patch('src.analysis.model_comparison.ModelEvaluator')
    def test_generate_report(self, mock_evaluator):
        """Test generate_report method."""
        # Set up mock
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        
        # Set up mock evaluate method
        mock_evaluator_instance.evaluate.return_value = {
            'accuracy': 0.9,
            'precision': 0.85,
            'recall': 0.9,
            'f1': 0.87,
            'inference_time': 0.1,
            'memory_usage': 100
        }
        
        # Initialize comparison
        comparison = ModelComparison(self.models_config)
        comparison.load_models()
        
        # Compare models
        comparison.compare_models(self.test_data)
        
        # Create report directory
        report_dir = os.path.join(self.test_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report
        report = comparison.generate_report(save_dir=report_dir)
        
        # Check that the report was generated
        self.assertIsInstance(report, str)
        self.assertIn('Model Comparison Report', report)
        self.assertIn('BERT', report)
        self.assertIn('FastText', report)
        
        # Check that the report was saved
        report_file = os.path.join(report_dir, 'model_comparison_report.txt')
        self.assertTrue(os.path.exists(report_file))

if __name__ == '__main__':
    unittest.main() 