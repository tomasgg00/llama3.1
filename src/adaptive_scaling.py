"""
adaptive_scaling.py: Adaptive batch and sample size optimization for the misinformation detection pipeline.

This module provides functions to:
- Dynamically determine optimal batch sizes based on hardware and model complexity
- Find the optimal sample size for large datasets using learning curves
- Monitor and adapt batch/sample sizes during training
"""
import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

def determine_optimal_batch_size(model=None, max_sequence_length=512, available_memory=None, start_batch_size=None):
    """
    Dynamically determine optimal batch size through binary search based on:
    - Available GPU memory
    - Model size and complexity
    - Maximum sequence length
    
    Args:
        model: Model to be used (optional)
        max_sequence_length: Maximum token length for training
        available_memory: Available GPU memory in GB (optional)
        start_batch_size: Initial batch size to try (optional)
    
    Returns:
        Optimal batch size
    """
    # Check available GPU memory if not provided
    if available_memory is None:
        try:
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
                free_memory = torch.cuda.memory_reserved(0) / (1024**3)
                available_memory = available_memory - free_memory
            else:
                available_memory = 8  # Assume modest CPU memory
                logging.info("No GPU detected, using CPU with assumed 8GB memory")
        except Exception as e:
            available_memory = 8
            logging.warning(f"Error detecting GPU memory: {e}. Assuming 8GB.")
    
    # Determine starting batch size based on model and sequence length
    if start_batch_size is None:
        if model is not None:
            model_size_str = str(model)
            if "70B" in model_size_str:
                start_batch_size = 1
            elif "30B" in model_size_str or "20B" in model_size_str:
                start_batch_size = 2
            elif "13B" in model_size_str:
                start_batch_size = 4
            elif "8B" in model_size_str:
                start_batch_size = 8
            else:
                start_batch_size = 16
        else:
            # Default based on available memory
            if available_memory > 24:
                start_batch_size = 16
            elif available_memory > 16:
                start_batch_size = 8
            elif available_memory > 8:
                start_batch_size = 4
            else:
                start_batch_size = 2
    
    logging.info(f"Starting batch size search with batch_size={start_batch_size}, available_memory={available_memory:.2f}GB")
    
    # If no GPU or no model provided, use heuristic approach
    if not torch.cuda.is_available() or model is None:
        # Apply scaling factor based on sequence length
        seq_length_factor = max(0.5, min(1.0, 512 / max_sequence_length))
        memory_factor = available_memory / 16  # Scale relative to 16GB baseline
        
        adjusted_batch_size = int(start_batch_size * seq_length_factor * memory_factor)
        adjusted_batch_size = max(1, adjusted_batch_size)  # Ensure minimum batch size of 1
        
        logging.info(f"Heuristic batch size determination: {adjusted_batch_size}")
        return adjusted_batch_size
    
    # With GPU and model available, use binary search to find maximum feasible batch size
    max_batch_size = start_batch_size * 2
    min_batch_size = 1
    current_batch_size = start_batch_size
    found_oom = False
    
    # Generate some dummy input to test memory
    dummy_length = min(max_sequence_length, 512)  # Cap sequence length for testing
    
    for attempt in range(10):  # Limit attempts to prevent infinite loop
        try:
            logging.info(f"Testing batch size: {current_batch_size}")
            
            # Clear GPU cache before test
            torch.cuda.empty_cache()
            
            # Generate dummy inputs for testing
            dummy_input_ids = torch.randint(0, 1000, (current_batch_size, dummy_length)).to("cuda")
            dummy_attention_mask = torch.ones(current_batch_size, dummy_length).to("cuda")
            
            # Try a forward pass
            with torch.no_grad():
                # For sequence classification models
                if hasattr(model, "forward") and "labels" in model.forward.__code__.co_varnames:
                    dummy_labels = torch.zeros(current_batch_size, dtype=torch.long).to("cuda")
                    _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, labels=dummy_labels)
                else:
                    _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
            
            # If we get here, batch size works - try a larger one
            min_batch_size = current_batch_size
            if found_oom:
                # If we've already found OOM, we're in binary search mode
                new_batch_size = (current_batch_size + max_batch_size) // 2
            else:
                # If no OOM yet, grow exponentially
                new_batch_size = current_batch_size * 2
            
            if new_batch_size == current_batch_size:
                # We've converged
                break
                
            current_batch_size = new_batch_size
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # We've hit a memory error, reduce batch size
            logging.info(f"Batch size {current_batch_size} too large: {str(e)[:100]}...")
            found_oom = True
            max_batch_size = current_batch_size
            current_batch_size = (min_batch_size + current_batch_size) // 2
            
            # If we can't even fit a batch size of 1, we're in trouble
            if current_batch_size < 1:
                logging.warning("Cannot fit even batch_size=1 in memory, using batch_size=1 with gradient accumulation")
                return 1
    
    # Apply a safety factor to avoid edge cases
    optimal_batch_size = max(1, int(min_batch_size * 0.9))
    
    logging.info(f"Determined optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def estimate_optimal_sample_size(df, model_fn, eval_metric='f1', sample_sizes=None, n_jobs=-1, cv=3, verbose=1):
    """
    Estimate the optimal sample size using learning curves.
    
    Args:
        df: DataFrame with training data
        model_fn: Function that returns a model compatible with scikit-learn
        eval_metric: Metric to evaluate ('accuracy' or 'f1')
        sample_sizes: List of sample sizes to try
        n_jobs: Number of jobs for parallel processing
        cv: Number of cross-validation folds
        verbose: Verbosity level
    
    Returns:
        Tuple of (optimal_sample_size, learning_curve_data)
    """
    logging.info(f"Estimating optimal sample size using learning curves with {eval_metric} metric")
    
    # If sample sizes not provided, create a exponential range
    if sample_sizes is None:
        # Create sample sizes from 1% to 100% of the dataset
        total_samples = len(df)
        sample_sizes = np.unique([
            int(total_samples * p) for p in 
            [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ])
        # Ensure minimum 100 samples
        sample_sizes = [s for s in sample_sizes if s >= 100]
    
    logging.info(f"Testing sample sizes: {sample_sizes}")
    
    # Extract features and labels
    y = df["label"].values
    
    # For text classification, we'll use a simplified feature representation for estimation
    # This is just to estimate sample size, not for actual model training
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create a simplified feature set for quick analysis
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
    X = vectorizer.fit_transform(df["content"].values)
    
    # Get the scoring function
    if eval_metric == 'f1':
        scoring = 'f1'
    else:
        scoring = 'accuracy'
    
    # Calculate learning curve
    model = model_fn()
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, 
            train_sizes=sample_sizes,
            cv=cv, 
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
    except Exception as e:
        logging.error(f"Error calculating learning curve: {e}")
        # Fall back to simple sampling
        return int(len(df) * 0.7), None
    
    # Calculate mean and std of test scores
    mean_test_scores = np.mean(test_scores, axis=1)
    std_test_scores = np.std(test_scores, axis=1)
    
    # Calculate incremental gains
    incremental_gains = np.zeros_like(mean_test_scores)
    incremental_gains[1:] = np.diff(mean_test_scores)
    
    # Find the point of diminishing returns (where incremental gain falls below threshold)
    threshold = 0.005  # 0.5% improvement threshold
    diminishing_returns_idx = np.where(incremental_gains < threshold)[0]
    if len(diminishing_returns_idx) > 0:
        optimal_idx = diminishing_returns_idx[0]
    else:
        # If no clear diminishing returns, use the largest value
        optimal_idx = len(sample_sizes) - 1
    
    optimal_sample_size = sample_sizes[optimal_idx]
    
    # Prepare result data
    curve_data = {
        'train_sizes': train_sizes.tolist(),
        'mean_train_scores': np.mean(train_scores, axis=1).tolist(),
        'mean_test_scores': mean_test_scores.tolist(),
        'std_test_scores': std_test_scores.tolist(),
        'incremental_gains': incremental_gains.tolist(),
        'optimal_sample_size': int(optimal_sample_size),
        'optimal_score': float(mean_test_scores[optimal_idx])
    }
    
    logging.info(f"Optimal sample size determined: {optimal_sample_size} with {eval_metric}={mean_test_scores[optimal_idx]:.4f}")
    
    # Plot learning curve if matplotlib is available
    try:
        plt.figure(figsize=(10, 6))
        plt.errorbar(train_sizes, mean_test_scores, yerr=std_test_scores, capsize=3, label=f'Test {eval_metric}')
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label=f'Train {eval_metric}')
        
        # Mark optimal point
        plt.axvline(x=optimal_sample_size, color='r', linestyle='--', label=f'Optimal: {optimal_sample_size}')
        
        plt.xlabel('Training Examples')
        plt.ylabel(f'{eval_metric.upper()} Score')
        plt.title(f'Learning Curve: {eval_metric.upper()} vs Sample Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot to file
        plot_path = os.path.join(os.getcwd(), "learning_curve.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Learning curve plot saved to {plot_path}")
        
    except Exception as e:
        logging.warning(f"Could not create learning curve plot: {e}")
    
    return optimal_sample_size, curve_data


def create_stratified_sample(df, sample_size, stratify_column="label", random_state=42):
    """
    Create a stratified sample of the dataset.
    
    Args:
        df: DataFrame to sample from
        sample_size: Number of samples to take
        stratify_column: Column to stratify by
        random_state: Random state for reproducibility
    
    Returns:
        DataFrame with stratified sample
    """
    # If sample size is larger than dataset, return the full dataset
    if sample_size >= len(df):
        logging.info(f"Requested sample size {sample_size} >= dataset size {len(df)}, using full dataset")
        return df
    
    logging.info(f"Creating stratified sample of size {sample_size} from dataset of size {len(df)}")
    
    # Check if stratify column exists
    if stratify_column not in df.columns:
        logging.warning(f"Stratify column '{stratify_column}' not found, using random sampling")
        return df.sample(sample_size, random_state=random_state)
    
    # Get stratification values
    stratify_values = df[stratify_column]
    
    # Create train/test split with stratification
    sample_df, _ = train_test_split(
        df, 
        train_size=sample_size,
        stratify=stratify_values,
        random_state=random_state
    )
    
    logging.info(f"Created stratified sample with shape: {sample_df.shape}")
    
    # Verify class distribution
    original_dist = df[stratify_column].value_counts(normalize=True)
    sample_dist = sample_df[stratify_column].value_counts(normalize=True)
    
    for label in original_dist.index:
        if label in sample_dist.index:
            diff = abs(original_dist[label] - sample_dist[label])
            if diff > 0.05:  # More than 5% difference
                logging.warning(f"Class imbalance in sample: label={label}, original={original_dist[label]:.2f}, sample={sample_dist[label]:.2f}")
    
    return sample_df


def monitor_batch_performance(batch_sizes, train_metrics, eval_metrics=None):
    """
    Monitor performance across different batch sizes to provide recommendations.
    
    Args:
        batch_sizes: List of batch sizes used
        train_metrics: Dictionary mapping batch sizes to training metrics
        eval_metrics: Dictionary mapping batch sizes to evaluation metrics
    
    Returns:
        Dictionary with batch size recommendations
    """
    if len(batch_sizes) < 2:
        return {"optimal_batch_size": batch_sizes[0], "recommendation": "More batch sizes needed for analysis"}
    
    # Calculate training throughput (examples/second)
    throughputs = {}
    for batch_size in batch_sizes:
        if batch_size in train_metrics and "examples_per_second" in train_metrics[batch_size]:
            throughputs[batch_size] = train_metrics[batch_size]["examples_per_second"]
    
    # Find batch size with best throughput
    if throughputs:
        best_throughput_batch_size = max(throughputs.items(), key=lambda x: x[1])[0]
    else:
        best_throughput_batch_size = max(batch_sizes)
    
    # Check evaluation metrics if available
    if eval_metrics is not None:
        # Calculate evaluation performance
        eval_scores = {}
        for batch_size in batch_sizes:
            if batch_size in eval_metrics and "f1" in eval_metrics[batch_size]:
                eval_scores[batch_size] = eval_metrics[batch_size]["f1"]
        
        # Find batch size with best evaluation score
        if eval_scores:
            best_eval_batch_size = max(eval_scores.items(), key=lambda x: x[1])[0]
        else:
            best_eval_batch_size = best_throughput_batch_size
        
        # Check if there's a significant difference between throughput and eval recommendations
        if best_eval_batch_size != best_throughput_batch_size:
            # Calculate the throughput sacrifice for better evaluation
            throughput_sacrifice = (throughputs.get(best_throughput_batch_size, 0) - 
                                  throughputs.get(best_eval_batch_size, 0)) / throughputs.get(best_throughput_batch_size, 1)
            
            # Calculate the evaluation gain
            eval_gain = (eval_scores.get(best_eval_batch_size, 0) - 
                        eval_scores.get(best_throughput_batch_size, 0)) / eval_scores.get(best_throughput_batch_size, 1)
            
            # If evaluation gain outweighs throughput sacrifice, prefer evaluation
            if eval_gain > throughput_sacrifice:
                optimal_batch_size = best_eval_batch_size
                recommendation = (f"Batch size {optimal_batch_size} provides best evaluation performance with "
                                f"acceptable throughput sacrifice of {throughput_sacrifice:.1%}")
            else:
                optimal_batch_size = best_throughput_batch_size
                recommendation = (f"Batch size {optimal_batch_size} provides best throughput with "
                                f"minimal evaluation sacrifice of {eval_gain:.1%}")
        else:
            optimal_batch_size = best_throughput_batch_size
            recommendation = f"Batch size {optimal_batch_size} provides both best throughput and evaluation performance"
    else:
        optimal_batch_size = best_throughput_batch_size
        recommendation = f"Batch size {optimal_batch_size} provides best throughput (no evaluation metrics available)"
    
    return {
        "optimal_batch_size": optimal_batch_size,
        "recommendation": recommendation,
        "throughputs": throughputs,
        "eval_metrics": eval_metrics
    }


def adaptive_batch_size_training(model, train_dataset, eval_dataset, training_fn, batch_size_range=None):
    """
    Perform adaptive batch size training by testing different batch sizes and selecting optimal.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_fn: Function to train model (should accept batch_size parameter)
        batch_size_range: Range of batch sizes to try
    
    Returns:
        Tuple of (trained_model, optimal_batch_size, training_results)
    """
    if batch_size_range is None:
        # Start with power of 2 values
        batch_size_range = [1, 2, 4, 8, 16, 32]
        
        # Filter based on dataset size (don't use large batch sizes for small datasets)
        max_reasonable_batch = max(1, min(16, len(train_dataset) // 100))
        batch_size_range = [b for b in batch_size_range if b <= max_reasonable_batch]
    
    logging.info(f"Adaptive batch size training with batch sizes: {batch_size_range}")
    
    # Try different batch sizes
    results = {}
    train_metrics = {}
    eval_metrics = {}
    
    # Test different batch sizes with short training runs
    for batch_size in batch_size_range:
        try:
            logging.info(f"Testing batch size: {batch_size}")
            result = training_fn(model, train_dataset, eval_dataset, batch_size=batch_size, epochs=1)
            
            results[batch_size] = result
            if "train_metrics" in result:
                train_metrics[batch_size] = result["train_metrics"]
            if "eval_metrics" in result:
                eval_metrics[batch_size] = result["eval_metrics"]
                
            logging.info(f"Batch size {batch_size} results: {result.get('summary', 'No summary available')}")
        except Exception as e:
            logging.error(f"Error with batch size {batch_size}: {e}")
            # If we get OOM with this batch size, it's too large
            if "CUDA out of memory" in str(e):
                logging.warning(f"Batch size {batch_size} causes CUDA OOM, will not try larger batch sizes")
                break
    
    # Find optimal batch size
    recommendation = monitor_batch_performance(
        list(results.keys()),
        train_metrics,
        eval_metrics
    )
    
    optimal_batch_size = recommendation["optimal_batch_size"]
    logging.info(f"Selected optimal batch size: {optimal_batch_size}")
    logging.info(f"Recommendation: {recommendation['recommendation']}")
    
    # Train final model with optimal batch size
    logging.info(f"Training final model with batch size {optimal_batch_size}")
    final_result = training_fn(model, train_dataset, eval_dataset, batch_size=optimal_batch_size)
    
    return final_result["model"], optimal_batch_size, {
        "batch_size_results": results,
        "optimal_batch_size": optimal_batch_size,
        "recommendation": recommendation,
        "final_result": final_result
    }


def adaptive_sample_size_training(df, model_fn, training_fn, optimal_sample_size=None, min_sample_size=1000,
                               eval_metric='f1', stratify_column="label", max_sample_size=None):
    """
    Perform adaptive sample size training by finding optimal sample size and training with it.
    
    Args:
        df: DataFrame with training data
        model_fn: Function that returns a model
        training_fn: Function to train model (should accept train_df parameter)
        optimal_sample_size: Pre-determined optimal sample size (if None, will be estimated)
        min_sample_size: Minimum sample size to consider
        eval_metric: Metric to optimize for
        stratify_column: Column to stratify by
        max_sample_size: Maximum sample size to consider (default: full dataset)
        
    Returns:
        Tuple of (trained_model, optimal_sample_size, training_results)
    """
    # Set default max sample size to dataset size
    if max_sample_size is None:
        max_sample_size = len(df)
    
    # Find optimal sample size if not provided
    if optimal_sample_size is None:
        from sklearn.linear_model import LogisticRegression
        
        # Use logistic regression as a fast proxy model for sample size estimation
        def lr_model_fn():
            return LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
        
        logging.info("Estimating optimal sample size...")
        optimal_sample_size, curve_data = estimate_optimal_sample_size(
            df, lr_model_fn, eval_metric=eval_metric
        )
        
        # Apply constraints
        optimal_sample_size = max(min_sample_size, min(optimal_sample_size, max_sample_size))
        logging.info(f"Constrained optimal sample size: {optimal_sample_size}")
    else:
        curve_data = None
        logging.info(f"Using provided optimal sample size: {optimal_sample_size}")
    
    # Create stratified sample
    sample_df = create_stratified_sample(df, optimal_sample_size, stratify_column)
    
    # Train model with optimal sample size
    logging.info(f"Training model with sample size {len(sample_df)}")
    model = model_fn()
    result = training_fn(model, sample_df)
    
    return result["model"], optimal_sample_size, {
        "optimal_sample_size": optimal_sample_size,
        "learning_curve_data": curve_data,
        "training_result": result
    }