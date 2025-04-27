import argparse
import logging
from pathlib import Path
from data.data_processor import AmazonReviewsProcessor
from models.sentiment_analyzer import SentimentAnalyzer
from visualization.visualizer import ResultsVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Amazon Reviews Sentiment Analysis')
    parser.add_argument('--model', type=str, default='bert',
                      choices=['bert', 'fasttext'],
                      help='Model to use for sentiment analysis')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory containing the data files')
    parser.add_argument('--results-dir', type=str, default='results',
                      help='Directory to save results')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize components
    processor = AmazonReviewsProcessor(data_dir=args.data_dir)
    analyzer = SentimentAnalyzer(model_type=args.model)
    visualizer = ResultsVisualizer(results_dir=args.results_dir)
    
    try:
        # Process data
        logger.info("Processing data...")
        train_df = processor.load_fasttext_data("train.ft.txt")
        train_df = processor.preprocess_data(train_df)
        processor.save_processed_data(train_df, "processed_train.csv")
        
        test_df = processor.load_fasttext_data("test.ft.txt")
        test_df = processor.preprocess_data(test_df)
        processor.save_processed_data(test_df, "processed_test.csv")
        
        # Train model
        logger.info(f"Training {args.model} model...")
        if args.model == 'fasttext':
            analyzer.train_fasttext(
                train_file="train.ft.txt",
                model_path=f"models/{args.model}_model.bin"
            )
        else:
            analyzer.train_bert(
                train_data=train_df,
                val_data=test_df
            )
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = analyzer.evaluate(test_df)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        visualizer.plot_confusion_matrix(results['confusion_matrix'])
        visualizer.plot_classification_report(results['classification_report'])
        visualizer.create_dashboard(results)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 