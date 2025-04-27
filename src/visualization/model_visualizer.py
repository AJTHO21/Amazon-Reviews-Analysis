import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelVisualizer:
    def __init__(self, results_dir):
        """
        Initialize the model visualizer.
        
        Args:
            results_dir (str): Directory containing model comparison results
        """
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / 'model_comparison_results.json'
        
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        # Create output directory for visualizations
        self.viz_dir = self.results_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
    
    def create_radar_chart(self):
        """
        Create a radar chart comparing model performance across metrics.
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(self.results.keys())
        
        # Prepare data
        data = []
        for model_name in model_names:
            values = [self.results[model_name][metric] for metric in metrics]
            data.append(values)
        
        # Create radar chart
        fig = go.Figure()
        
        for i, model_name in enumerate(model_names):
            fig.add_trace(go.Scatterpolar(
                r=data[i],
                theta=metrics,
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Model Performance Radar Chart",
            showlegend=True
        )
        
        fig.write_html(self.viz_dir / 'radar_chart.html')
        fig.write_image(self.viz_dir / 'radar_chart.png')
        logger.info("Created radar chart")
    
    def create_performance_comparison(self):
        """
        Create a comprehensive performance comparison visualization.
        """
        model_names = list(self.results.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Accuracy Comparison", "F1 Score Comparison",
                "Inference Time Comparison", "Memory Usage Comparison"
            )
        )
        
        # Accuracy comparison
        accuracies = [self.results[model_name]['accuracy'] for model_name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name="Accuracy"),
            row=1, col=1
        )
        
        # F1 score comparison
        f1_scores = [self.results[model_name]['f1'] for model_name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=f1_scores, name="F1 Score"),
            row=1, col=2
        )
        
        # Inference time comparison
        inference_times = [self.results[model_name]['inference_time'] for model_name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=inference_times, name="Inference Time (s)"),
            row=2, col=1
        )
        
        # Memory usage comparison
        memory_usage = [self.results[model_name]['memory_usage'] for model_name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=memory_usage, name="Memory Usage (MB)"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Comprehensive Model Performance Comparison",
            showlegend=False
        )
        
        fig.write_html(self.viz_dir / 'performance_comparison.html')
        fig.write_image(self.viz_dir / 'performance_comparison.png')
        logger.info("Created performance comparison visualization")
    
    def create_cross_validation_plot(self):
        """
        Create a visualization of cross-validation results if available.
        """
        if not any('cv_accuracy' in results for results in self.results.values()):
            logger.warning("Cross-validation results not available")
            return
        
        model_names = list(self.results.keys())
        accuracies = []
        std_accuracies = []
        
        for model_name in model_names:
            if 'cv_accuracy' in self.results[model_name]:
                accuracies.append(self.results[model_name]['cv_accuracy'])
                std_accuracies.append(self.results[model_name]['cv_std_accuracy'])
            else:
                accuracies.append(None)
                std_accuracies.append(None)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=accuracies,
            error_y=dict(
                type='data',
                array=std_accuracies,
                visible=True
            ),
            name="Cross-validation Accuracy"
        ))
        
        fig.update_layout(
            title="Cross-validation Results",
            xaxis_title="Models",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1]
        )
        
        fig.write_html(self.viz_dir / 'cross_validation.html')
        fig.write_image(self.viz_dir / 'cross_validation.png')
        logger.info("Created cross-validation visualization")
    
    def create_model_comparison_table(self):
        """
        Create an HTML table with model comparison results.
        """
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'inference_time', 'memory_usage']
        
        # Prepare data
        data = []
        for model_name in model_names:
            row = [model_name]
            for metric in metrics:
                if metric in self.results[model_name]:
                    value = self.results[model_name][metric]
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("N/A")
            data.append(row)
        
        # Create HTML table
        html = "<html><head><style>"
        html += "table {border-collapse: collapse; width: 100%;}"
        html += "th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}"
        html += "th {background-color: #f2f2f2;}"
        html += "tr:nth-child(even) {background-color: #f9f9f9;}"
        html += "tr:hover {background-color: #f5f5f5;}"
        html += "</style></head><body>"
        html += "<h2>Model Comparison Results</h2>"
        html += "<table>"
        
        # Header
        html += "<tr><th>Model</th>"
        for metric in metrics:
            html += f"<th>{metric.replace('_', ' ').title()}</th>"
        html += "</tr>"
        
        # Data rows
        for row in data:
            html += "<tr>"
            for cell in row:
                html += f"<td>{cell}</td>"
            html += "</tr>"
        
        html += "</table></body></html>"
        
        with open(self.viz_dir / 'model_comparison_table.html', 'w') as f:
            f.write(html)
        
        logger.info("Created model comparison table")
    
    def create_all_visualizations(self):
        """
        Create all visualizations.
        """
        logger.info("Creating all visualizations...")
        self.create_radar_chart()
        self.create_performance_comparison()
        self.create_cross_validation_plot()
        self.create_model_comparison_table()
        logger.info("All visualizations created successfully")

def main():
    # Example usage
    results_dir = Path("results/model_comparison")
    visualizer = ModelVisualizer(results_dir)
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main() 