import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    def __init__(self, results_dir: str = "../results"):
        """
        Initialize the visualizer.
        
        Args:
            results_dir (str): Path to results directory
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str = "Confusion Matrix"):
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            title (str): Plot title
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            square=True
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plt.savefig(self.results_dir / "confusion_matrix.png")
        plt.close()
    
    def plot_classification_report(self, report: Dict[str, Any]):
        """
        Plot classification report metrics.
        
        Args:
            report (Dict[str, Any]): Classification report
        """
        # Convert report to DataFrame
        df = pd.DataFrame(report).transpose()
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Precision",
                "Recall",
                "F1-Score",
                "Support"
            )
        )
        
        # Add traces
        metrics = ['precision', 'recall', 'f1-score', 'support']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, pos in zip(metrics, positions):
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df[metric],
                    name=metric.capitalize()
                ),
                row=pos[0],
                col=pos[1]
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Classification Report Metrics",
            showlegend=False
        )
        
        # Save plot
        fig.write_html(str(self.results_dir / "classification_report.html"))
    
    def plot_training_history(self, history: Dict[str, List[float]]):
        """
        Plot training history.
        
        Args:
            history (Dict[str, List[float]]): Training history
        """
        plt.figure(figsize=(12, 6))
        
        for metric, values in history.items():
            plt.plot(values, label=metric)
        
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(self.results_dir / "training_history.png")
        plt.close()
    
    def create_dashboard(self, results: Dict[str, Any]):
        """
        Create an interactive dashboard with all results.
        
        Args:
            results (Dict[str, Any]): Analysis results
        """
        # Create dashboard layout
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Confusion Matrix",
                "Classification Metrics",
                "Training History",
                "Model Comparison"
            )
        )
        
        # Add confusion matrix
        fig.add_trace(
            go.Heatmap(
                z=results['confusion_matrix'],
                colorscale='Blues',
                showscale=True
            ),
            row=1,
            col=1
        )
        
        # Add classification metrics
        metrics_df = pd.DataFrame(results['classification_report']).transpose()
        for metric in ['precision', 'recall', 'f1-score']:
            fig.add_trace(
                go.Bar(
                    x=metrics_df.index,
                    y=metrics_df[metric],
                    name=metric.capitalize()
                ),
                row=1,
                col=2
            )
        
        # Add training history if available
        if 'training_history' in results:
            for metric, values in results['training_history'].items():
                fig.add_trace(
                    go.Scatter(
                        y=values,
                        name=metric,
                        mode='lines'
                    ),
                    row=2,
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            title_text="Sentiment Analysis Results Dashboard",
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html(str(self.results_dir / "dashboard.html"))

if __name__ == "__main__":
    # Example usage
    visualizer = ResultsVisualizer()
    
    # Example results
    example_results = {
        'confusion_matrix': np.array([[100, 20], [15, 95]]),
        'classification_report': {
            '0': {'precision': 0.87, 'recall': 0.83, 'f1-score': 0.85, 'support': 120},
            '1': {'precision': 0.83, 'recall': 0.86, 'f1-score': 0.84, 'support': 110}
        },
        'training_history': {
            'loss': [0.5, 0.3, 0.2, 0.1],
            'accuracy': [0.7, 0.8, 0.85, 0.9]
        }
    }
    
    # Create visualizations
    visualizer.plot_confusion_matrix(example_results['confusion_matrix'])
    visualizer.plot_classification_report(example_results['classification_report'])
    visualizer.plot_training_history(example_results['training_history'])
    visualizer.create_dashboard(example_results) 