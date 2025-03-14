"""
inference.py: Inference module for the misinformation detection pipeline.

This module handles:
- Model inference with single and batch inputs
- Ensemble prediction
- Evaluation and benchmarking
- Error analysis
"""
import os
import time
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.utils import (
    compute_metrics, save_json, load_json,
    visualize_confusion_matrix, visualize_roc_curve, visualize_precision_recall_curve
)
from src.config import RESULTS_DIR, IMPORTANT_FEATURES
from src.preprocessing import create_prompt

def ensemble_inference(model, tokenizer, text, features=None, num_runs=5, device=None):
    """
    Run ensemble inference with multiple forward passes for more robust prediction.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        text: Text to classify
        features: Optional dictionary of numerical features for enhanced prompting
        num_runs: Number of forward passes to aggregate (default=5)
        device: Device to run inference on
    
    Returns:
        Dictionary with prediction results and confidence metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Configure device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Track results from multiple runs
    results = []
    
    for i in range(num_runs):
        # Create prompt (enhanced or basic)
        if features is not None:
            prompt = create_prompt(text, features, use_enhanced_prompt=True)
        else:
            prompt = create_prompt(text)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        results.append(probabilities[0].cpu().numpy())
    
    # Aggregate results
    results_array = np.array(results)
    
    # Calculate mean and standard deviation of probabilities
    mean_probs = np.mean(results_array, axis=0)
    std_probs = np.std(results_array, axis=0)
    
    # Get prediction (class with highest mean probability)
    prediction = np.argmax(mean_probs)
    
    # Calculate confidence metrics
    confidence = mean_probs[prediction]
    confidence_std = std_probs[prediction]
    
    # Calculate margin (difference between highest and second highest probability)
    sorted_probs = np.sort(mean_probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    
    # Determine confidence level
    if confidence > 0.9 and margin > 0.5:
        confidence_level = "Very High"
    elif confidence > 0.8 and margin > 0.3:
        confidence_level = "High"
    elif confidence > 0.7 and margin > 0.2:
        confidence_level = "Medium"
    elif confidence > 0.6:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"
    
    # Create detailed result
    result = {
        "text": text,
        "prediction": "TRUE (Factual)" if prediction == 1 else "FALSE (Misinformation)",
        "prediction_label": int(prediction),
        "confidence": float(confidence),
        "confidence_std": float(confidence_std),
        "confidence_margin": float(margin),
        "confidence_level": confidence_level,
        "requires_review": confidence < 0.7 or margin < 0.2,
        "class_probabilities": {
            "FALSE (Misinformation)": float(mean_probs[0]),
            "TRUE (Factual)": float(mean_probs[1])
        },
        "ensemble_runs": num_runs
    }
    
    return result

def batch_inference(model, tokenizer, texts, features_list=None, batch_size=8, device=None, use_enhanced_prompt=False):
    """
    Process a batch of texts efficiently for inference.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        texts: List of texts to classify
        features_list: Optional list of feature dictionaries for enhanced prompting
        batch_size: Batch size for processing
        device: Device to run inference on
        use_enhanced_prompt: Whether to use enhanced prompts
    
    Returns:
        List of prediction results
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Configure device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        # Get batch
        batch_texts = texts[i:i+batch_size]
        
        # Get features for batch if available
        batch_features = None
        if features_list is not None and use_enhanced_prompt:
            batch_features = features_list[i:i+batch_size]
        
        # Create prompts
        batch_prompts = []
        for j, text in enumerate(batch_texts):
            if batch_features is not None:
                features = batch_features[j]
                prompt = create_prompt(text, features, use_enhanced_prompt=True)
            else:
                prompt = create_prompt(text, use_enhanced_prompt=False)
            batch_prompts.append(prompt)
        
        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Process each result in batch
        for j, text in enumerate(batch_texts):
            probs = probabilities[j].cpu().numpy()
            prediction = np.argmax(probs)
            confidence = probs[prediction]
            
            # Calculate margin
            sorted_probs = np.sort(probs)[::-1]
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            
            # Determine confidence level
            if confidence > 0.9 and margin > 0.5:
                confidence_level = "Very High"
            elif confidence > 0.8 and margin > 0.3:
                confidence_level = "High"
            elif confidence > 0.7 and margin > 0.2:
                confidence_level = "Medium"
            elif confidence > 0.6:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
            
            result = {
                "text": text,
                "prediction": "TRUE (Factual)" if prediction == 1 else "FALSE (Misinformation)",
                "prediction_label": int(prediction),
                "confidence": float(confidence),
                "confidence_margin": float(margin),
                "confidence_level": confidence_level,
                "requires_review": confidence < 0.7 or margin < 0.2,
                "class_probabilities": {
                    "FALSE (Misinformation)": float(probs[0]),
                    "TRUE (Factual)": float(probs[1])
                }
            }
            
            # Add features if available
            if batch_features is not None:
                result["features"] = batch_features[j]
            
            results.append(result)
    
    return results

def evaluate_model(model, tokenizer, test_df, output_dir=None, batch_size=8, use_enhanced_prompt=False):
    """
    Evaluate model performance on a test dataset.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        test_df: Test dataset
        output_dir: Output directory for results
        batch_size: Batch size for processing
        use_enhanced_prompt: Whether to use enhanced prompts
    
    Returns:
        Dictionary with evaluation results
    """
    # Set default output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f"evaluation_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Extract texts, labels, and features
    texts = test_df["content"].tolist()
    true_labels = test_df["label"].tolist()
    
    # Extract features for enhanced prompts if needed
    features_list = None
    if use_enhanced_prompt:
        features_list = []
        for _, row in test_df.iterrows():
            features = {
                feature: row[feature] 
                for feature in IMPORTANT_FEATURES 
                if feature in test_df.columns
            }
            features_list.append(features)
    
    # Run inference
    logging.info(f"Running evaluation on {len(texts)} examples")
    
    start_time = time.time()
    results = batch_inference(
        model, 
        tokenizer, 
        texts, 
        features_list=features_list,
        batch_size=batch_size,
        use_enhanced_prompt=use_enhanced_prompt
    )
    inference_time = time.time() - start_time
    
    # Extract predictions and scores
    predictions = [result["prediction_label"] for result in results]
    confidences = [result["confidence"] for result in results]
    probabilities = np.array([
        [result["class_probabilities"]["FALSE (Misinformation)"], 
         result["class_probabilities"]["TRUE (Factual)"]]
        for result in results
    ])
    
    # Calculate metrics
    metrics = compute_metrics(probabilities, true_labels)
    
    # Add inference time metrics
    inference_metrics = {
        "total_inference_time": inference_time,
        "average_inference_time": inference_time / len(texts),
        "examples_per_second": len(texts) / inference_time,
        "batch_size": batch_size,
        "use_enhanced_prompt": use_enhanced_prompt
    }
    
    # Add confidence analysis
    confidence_metrics = {
        "average_confidence": np.mean(confidences),
        "confidence_std": np.std(confidences),
        "confidence_quantiles": {
            "p10": np.percentile(confidences, 10),
            "p25": np.percentile(confidences, 25),
            "p50": np.percentile(confidences, 50),
            "p75": np.percentile(confidences, 75),
            "p90": np.percentile(confidences, 90)
        },
        "confidence_distribution": {
            "very_high": sum(1 for c in confidences if c >= 0.9),
            "high": sum(1 for c in confidences if 0.8 <= c < 0.9),
            "medium": sum(1 for c in confidences if 0.7 <= c < 0.8),
            "low": sum(1 for c in confidences if 0.6 <= c < 0.7),
            "very_low": sum(1 for c in confidences if c < 0.6)
        }
    }
    
    # Create detailed results
    evaluation_results = {
        "metrics": metrics,
        "inference_metrics": inference_metrics,
        "confidence_metrics": confidence_metrics,
        "dataset_size": len(texts),
        "timestamp": datetime.now().isoformat(),
        "examples": results[:10]  # Include a few example predictions
    }
    
    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    save_json(evaluation_results, results_file)
    
    # Create visualizations
    try:
        # Confusion matrix
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        visualize_confusion_matrix(
            true_labels, 
            predictions,
            output_path=cm_path, 
            title="Test Set Confusion Matrix"
        )
        
        # ROC curve
        roc_path = os.path.join(output_dir, "roc_curve.png")
        visualize_roc_curve(
            true_labels, 
            probabilities[:, 1],
            output_path=roc_path, 
            title="ROC Curve"
        )
        
        # Precision-Recall curve
        pr_path = os.path.join(output_dir, "precision_recall_curve.png")
        visualize_precision_recall_curve(
            true_labels, 
            probabilities[:, 1],
            output_path=pr_path, 
            title="Precision-Recall Curve"
        )
        
        # Confidence distribution
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        
        # Split by correct/incorrect
        correct_mask = np.array(predictions) == np.array(true_labels)
        correct_confidences = np.array(confidences)[correct_mask]
        incorrect_confidences = np.array(confidences)[~correct_mask]
        
        # Plot histograms
        sns.histplot(correct_confidences, color='green', alpha=0.5, label='Correct Predictions', kde=True)
        sns.histplot(incorrect_confidences, color='red', alpha=0.5, label='Incorrect Predictions', kde=True)
        
        plt.axvline(x=0.7, color='grey', linestyle='--', label='Confidence Threshold (0.7)')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by Prediction Correctness')
        plt.legend()
        plt.tight_layout()
        
        conf_path = os.path.join(output_dir, "confidence_distribution.png")
        plt.savefig(conf_path)
        plt.close()
        
        # Add visualization paths to results
        evaluation_results["visualizations"] = {
            "confusion_matrix": cm_path,
            "roc_curve": roc_path,
            "precision_recall_curve": pr_path,
            "confidence_distribution": conf_path
        }
        
        # Update the results file with visualization paths
        save_json(evaluation_results, results_file)
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")
    
    logging.info(f"Evaluation results saved to {results_file}")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    
    return evaluation_results

def benchmark_comparison(model, tokenizer, test_df, output_dir=None):
    """
    Run benchmark comparison between basic and enhanced prompts.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        test_df: Test dataset
        output_dir: Output directory for results
    
    Returns:
        Dictionary with benchmark results
    """
    # Set default output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Running benchmark comparison between basic and enhanced prompts")
    
    # Run evaluations with different settings
    basic_results = evaluate_model(
        model,
        tokenizer,
        test_df,
        output_dir=os.path.join(output_dir, "basic_prompt"),
        use_enhanced_prompt=False
    )
    
    # Check if we have the required features for enhanced prompts
    has_features = all(feature in test_df.columns for feature in IMPORTANT_FEATURES)
    
    if has_features:
        enhanced_results = evaluate_model(
            model,
            tokenizer,
            test_df,
            output_dir=os.path.join(output_dir, "enhanced_prompt"),
            use_enhanced_prompt=True
        )
    else:
        logging.warning("Dataset missing required features for enhanced prompts")
        enhanced_results = None
    
    # Compare results
    if enhanced_results is not None:
        comparison = {
            "metrics": {
                "accuracy": {
                    "basic": basic_results["metrics"]["accuracy"],
                    "enhanced": enhanced_results["metrics"]["accuracy"],
                    "difference": enhanced_results["metrics"]["accuracy"] - basic_results["metrics"]["accuracy"],
                    "percent_improvement": (enhanced_results["metrics"]["accuracy"] - basic_results["metrics"]["accuracy"]) / basic_results["metrics"]["accuracy"] * 100 if basic_results["metrics"]["accuracy"] > 0 else 0
                },
                "f1": {
                    "basic": basic_results["metrics"]["f1"],
                    "enhanced": enhanced_results["metrics"]["f1"],
                    "difference": enhanced_results["metrics"]["f1"] - basic_results["metrics"]["f1"],
                    "percent_improvement": (enhanced_results["metrics"]["f1"] - basic_results["metrics"]["f1"]) / basic_results["metrics"]["f1"] * 100 if basic_results["metrics"]["f1"] > 0 else 0
                },
                "precision": {
                    "basic": basic_results["metrics"]["precision"],
                    "enhanced": enhanced_results["metrics"]["precision"],
                    "difference": enhanced_results["metrics"]["precision"] - basic_results["metrics"]["precision"],
                    "percent_improvement": (enhanced_results["metrics"]["precision"] - basic_results["metrics"]["precision"]) / basic_results["metrics"]["precision"] * 100 if basic_results["metrics"]["precision"] > 0 else 0
                },
                "recall": {
                    "basic": basic_results["metrics"]["recall"],
                    "enhanced": enhanced_results["metrics"]["recall"],
                    "difference": enhanced_results["metrics"]["recall"] - basic_results["metrics"]["recall"],
                    "percent_improvement": (enhanced_results["metrics"]["recall"] - basic_results["metrics"]["recall"]) / basic_results["metrics"]["recall"] * 100 if basic_results["metrics"]["recall"] > 0 else 0
                },
                "auc": {
                    "basic": basic_results["metrics"]["auc"],
                    "enhanced": enhanced_results["metrics"]["auc"],
                    "difference": enhanced_results["metrics"]["auc"] - basic_results["metrics"]["auc"],
                    "percent_improvement": (enhanced_results["metrics"]["auc"] - basic_results["metrics"]["auc"]) / basic_results["metrics"]["auc"] * 100 if basic_results["metrics"]["auc"] > 0 else 0
                }
            },
            "inference_time": {
                "basic": basic_results["inference_metrics"]["average_inference_time"],
                "enhanced": enhanced_results["inference_metrics"]["average_inference_time"],
                "difference": enhanced_results["inference_metrics"]["average_inference_time"] - basic_results["inference_metrics"]["average_inference_time"],
                "percent_change": (enhanced_results["inference_metrics"]["average_inference_time"] - basic_results["inference_metrics"]["average_inference_time"]) / basic_results["inference_metrics"]["average_inference_time"] * 100 if basic_results["inference_metrics"]["average_inference_time"] > 0 else 0
            },
            "confidence": {
                "basic": basic_results["confidence_metrics"]["average_confidence"],
                "enhanced": enhanced_results["confidence_metrics"]["average_confidence"],
                "difference": enhanced_results["confidence_metrics"]["average_confidence"] - basic_results["confidence_metrics"]["average_confidence"],
                "percent_change": (enhanced_results["confidence_metrics"]["average_confidence"] - basic_results["confidence_metrics"]["average_confidence"]) / basic_results["confidence_metrics"]["average_confidence"] * 100 if basic_results["confidence_metrics"]["average_confidence"] > 0 else 0
            }
        }
        
        # Save comparison results
        comparison_file = os.path.join(output_dir, "prompt_comparison.json")
        save_json(comparison, comparison_file)
        
        # Create comparison visualizations
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create bar chart comparison of main metrics
            plt.figure(figsize=(12, 8))
            metrics = ["accuracy", "precision", "recall", "f1", "auc"]
            x = np.arange(len(metrics))
            width = 0.35
            
            basic_values = [basic_results["metrics"][m] for m in metrics]
            enhanced_values = [enhanced_results["metrics"][m] for m in metrics]
            
            plt.bar(x - width/2, basic_values, width, label='Basic Prompt')
            plt.bar(x + width/2, enhanced_values, width, label='Enhanced Prompt')
            
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Performance Comparison: Basic vs Enhanced Prompts')
            plt.xticks(x, metrics)
            plt.legend()
            
            # Add improvement percentages
            for i, metric in enumerate(metrics):
                improvement = comparison["metrics"][metric]["percent_improvement"]
                color = 'green' if improvement > 0 else 'red'
                plt.annotate(f"{improvement:.1f}%", 
                            xy=(i, max(basic_values[i], enhanced_values[i]) + 0.02),
                            ha='center', 
                            color=color)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
            plt.close()
            
            logging.info(f"Benchmark comparison saved to {comparison_file}")
            
        except Exception as e:
            logging.error(f"Error creating comparison visualizations: {e}")
        
        return {
            "basic_results": basic_results,
            "enhanced_results": enhanced_results,
            "comparison": comparison
        }
    else:
        return {
            "basic_results": basic_results
        }

def detect_misinformation(text, model, tokenizer, features=None, use_ensemble=True, num_runs=5):
    """
    Detect misinformation in a text input.
    
    Args:
        text: Text to analyze
        model: Model for inference
        tokenizer: Tokenizer for encoding
        features: Optional features for enhanced prompting
        use_ensemble: Whether to use ensemble prediction
        num_runs: Number of runs for ensemble prediction
    
    Returns:
        Prediction result
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if use_ensemble:
        # Use ensemble prediction
        result = ensemble_inference(
            model,
            tokenizer,
            text,
            features=features,
            num_runs=num_runs
        )
    else:
        # Use single prediction
        from src.preprocessing import create_prompt
        
        # Create prompt
        if features is not None:
            prompt = create_prompt(text, features, use_enhanced_prompt=True)
        else:
            prompt = create_prompt(text)
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probabilities = probabilities[0].cpu().numpy()
        
        # Get prediction
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        
        # Create result
        result = {
            "text": text,
            "prediction": "TRUE (Factual)" if prediction == 1 else "FALSE (Misinformation)",
            "prediction_label": prediction,
            "confidence": confidence,
            "class_probabilities": {
                "FALSE (Misinformation)": float(probabilities[0]),
                "TRUE (Factual)": float(probabilities[1])
            }
        }
    
    return result

def extract_features_from_text(text):
    """
    Extract features from a text for enhanced prompting.
    
    Args:
        text: Text to extract features from
    
    Returns:
        Dictionary with extracted features
    """
    from src.preprocessing import extract_all_features
    
    # Extract features
    features = extract_all_features(text)
    
    # Filter to important features
    important_features = {
        feature: features[feature]
        for feature in IMPORTANT_FEATURES
        if feature in features
    }
    
    return important_features

def load_and_prepare_model(model_path, device=None):
    """
    Load and prepare a model for inference.
    
    Args:
        model_path: Path to the model or adapter
        device: Device to load the model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    from src.model import load_model_for_inference
    
    # Load model and tokenizer
    model, tokenizer = load_model_for_inference(model_path, device)
    
    if model is None or tokenizer is None:
        logging.error(f"Failed to load model from {model_path}")
        return None, None
    
    # Ensure model is in evaluation mode
    model.eval()
    
    return model, tokenizer