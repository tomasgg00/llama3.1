import os
import sys
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import argparse
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

# Add the current directory to the path so imports work correctly
sys.path.append('.')

from src.utils import setup_logging, compute_metrics, visualize_confusion_matrix, save_json
from src.model import load_model_for_inference
from src.inference import batch_inference

def detailed_classification_analysis(true_labels, predictions, probabilities, output_dir):
    """
    Perform comprehensive classification analysis.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        probabilities: Class probabilities
        output_dir: Directory to save analysis results
    
    Returns:
        Dictionary with detailed classification metrics
    """
    # Create comprehensive classification report
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=['Misinformation', 'Factual'], 
        output_dict=True
    )
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Confusion Matrix with more details
    cm = confusion_matrix(true_labels, predictions)
    
    # Visualize Confusion Matrix with percentages
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Misinformation', 'Factual'],
        yticklabels=['Misinformation', 'Factual']
    )
    plt.title('Detailed Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_confusion_matrix.png'))
    plt.close()
    
    # Probability Distribution Analysis
    plt.figure(figsize=(12, 6))
    
    # Probability distributions for correct and incorrect predictions
    correct_probs = probabilities[true_labels == predictions]
    incorrect_probs = probabilities[true_labels != predictions]
    
    plt.subplot(1, 2, 1)
    sns.histplot(correct_probs[:, 1], color='green', label='Correct Predictions', kde=True)
    plt.title('Probability Distribution\nCorrect Predictions')
    plt.xlabel('Probability of Factual')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(incorrect_probs[:, 1], color='red', label='Incorrect Predictions', kde=True)
    plt.title('Probability Distribution\nIncorrect Predictions')
    plt.xlabel('Probability of Factual')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distributions.png'))
    plt.close()
    
    return report

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Advanced Model Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to evaluate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--use-enhanced-prompt", action="store_true", help="Use enhanced prompting")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare output directory
    output_dir = f"results/evaluation_{args.dataset}_{os.path.basename(args.model)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset_path = f"processed_data/{args.dataset}_full.json"
    logger.info(f"Loading dataset from: {dataset_path}")
    test_df = pd.read_json(dataset_path)
    logger.info(f"Loaded {len(test_df)} examples")

    # Dataset Overview
    logger.info("\n===== DATASET OVERVIEW =====")
    label_counts = test_df['label'].value_counts()
    logger.info("Class Distribution:")
    logger.info(label_counts)
    logger.info(f"Class Proportions:\n{label_counts / len(test_df)}")

    # Text Length Analysis
    test_df['text_length'] = test_df['content'].str.len()
    logger.info("\nText Length Statistics:")
    logger.info(test_df.groupby('label')['text_length'].describe())

    # Load model
    logger.info(f"Loading model from: {args.model}")
    model, tokenizer = load_model_for_inference(args.model)

    if model is None or tokenizer is None:
        logger.error("Failed to load model")
        return

    # Prepare inputs
    texts = test_df["content"].tolist()
    true_labels = test_df["label"].tolist()

    # Run inference
    logger.info(f"Running evaluation on {len(test_df)} examples")
    results = batch_inference(
        model,
        tokenizer,
        texts,
        batch_size=args.batch_size,
        use_enhanced_prompt=args.use_enhanced_prompt
    )

    # Extract predictions and probabilities
    predictions = [result["prediction_label"] for result in results]
    probabilities = np.array([
        [result["class_probabilities"]["FALSE (Misinformation)"], 
         result["class_probabilities"]["TRUE (Factual)"]]
        for result in results
    ])

    # Compute metrics
    metrics = compute_metrics(probabilities, true_labels)

    # Detailed Classification Analysis
    detailed_report = detailed_classification_analysis(
        true_labels, 
        predictions, 
        probabilities, 
        output_dir
    )

    # Save comprehensive results
    comprehensive_results = {
        "dataset_info": {
            "total_samples": len(test_df),
            "class_distribution": label_counts.to_dict(),
            "class_proportions": (label_counts / len(test_df)).to_dict()
        },
        "metrics": metrics,
        "detailed_classification_report": detailed_report
    }

    # Save results
    save_json(comprehensive_results, os.path.join(output_dir, "comprehensive_evaluation.json"))

    # Predictions DataFrame for error analysis
    predictions_df = pd.DataFrame({
        "content": texts,
        "true_label": true_labels,
        "predicted_label": predictions,
        "confidence": [result["confidence"] for result in results],
        "prediction": [result["prediction"] for result in results]
    })

    # Error Analysis
    errors_df = predictions_df[predictions_df["true_label"] != predictions_df["predicted_label"]]
    errors_df.to_csv(os.path.join(output_dir, "error_cases.csv"), index=False)

    # Logging Results
    logger.info("\n===== EVALUATION RESULTS =====")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset} ({len(test_df)} examples)")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"AUC: {metrics.get('auc', 0):.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Total Error Cases: {len(errors_df)}")

if __name__ == "__main__":
    main()