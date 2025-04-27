import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttentionVisualizer:
    def __init__(self, model_path):
        """
        Initialize the attention visualizer.
        
        Args:
            model_path (str): Path to the saved BERT model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            output_attentions=True  # Enable attention output
        ).to(self.device)
        self.model.eval()
    
    def get_attention_weights(self, text, max_length=256):
        """
        Get attention weights for a given text.
        
        Args:
            text (str): Input text
            max_length (int): Maximum sequence length
            
        Returns:
            tuple: (attention weights, tokens)
        """
        # Tokenize input
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
        
        # Move inputs to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Get attention weights from all layers
        attention_weights = outputs.attentions  # Tuple of attention weights
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return attention_weights, tokens
    
    def visualize_attention(self, text, layer_idx=-1, head_idx=None, save_path=None):
        """
        Visualize attention weights for a given text.
        
        Args:
            text (str): Input text
            layer_idx (int): Index of the layer to visualize (-1 for last layer)
            head_idx (int, optional): Index of the attention head to visualize
            save_path (str, optional): Path to save the visualization
        """
        attention_weights, tokens = self.get_attention_weights(text)
        
        # Get attention weights for the specified layer
        layer_weights = attention_weights[layer_idx].squeeze(0)  # Remove batch dimension
        
        if head_idx is not None:
            # Visualize specific attention head
            attention_matrix = layer_weights[head_idx].cpu().numpy()
        else:
            # Average attention across all heads
            attention_matrix = layer_weights.mean(dim=0).cpu().numpy()
        
        # Create the visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True
        )
        plt.title(f'Attention Weights {"(Head " + str(head_idx) + ")" if head_idx is not None else "(Averaged)"}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Attention visualization saved to {save_path}")
        
        plt.close()
    
    def visualize_all_heads(self, text, layer_idx=-1, save_dir=None):
        """
        Visualize attention weights for all heads in a layer.
        
        Args:
            text (str): Input text
            layer_idx (int): Index of the layer to visualize
            save_dir (str, optional): Directory to save the visualizations
        """
        attention_weights, tokens = self.get_attention_weights(text)
        num_heads = attention_weights[layer_idx].size(1)
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for head_idx in range(num_heads):
            save_path = save_dir / f'attention_head_{head_idx}.png' if save_dir else None
            self.visualize_attention(text, layer_idx, head_idx, save_path)
            logger.info(f"Visualized attention head {head_idx}")

def main():
    # Example usage
    model_path = Path("models/bert_model")
    visualizer = AttentionVisualizer(str(model_path))
    
    # Example text
    text = "This product exceeded my expectations. The quality is outstanding!"
    
    # Visualize attention for the last layer, averaged across heads
    visualizer.visualize_attention(
        text,
        save_path="results/attention_visualization.png"
    )
    
    # Visualize all attention heads in the last layer
    visualizer.visualize_all_heads(
        text,
        save_dir="results/attention_heads"
    )

if __name__ == "__main__":
    main() 