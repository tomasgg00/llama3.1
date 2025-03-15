"""
integration.py
Integration of adaptive scaling features with the existing pipeline.

This module provides:
1. Integration with app.py to add intelligent batch/sample size determination
2. Extension of preprocessing.py with smart sampling
3. Extension of training.py with adaptive batch sizing
4. Command-line argument handlers for the new features
5. Overrides for existing functions to incorporate adaptive scaling
"""
import os
import sys
import logging
import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from src.utils import setup_logging, clean_up_memory
from src.config import TRAINING_CONFIG, PROCESSED_DATA_DIR, RESULTS_DIR

# Add to preprocessing.py
def smart_sampling(df, max_samples=None, sampling_strategy='stratified', stratify_column='label', 
                   estimate_optimal=False, min_samples=1000, random_state=42):
    """
    Perform intelligent sampling of a large dataset.
    
    Args:
        df: DataFrame to sample
        max_samples: Maximum number of samples (or None to estimate optimal)
        sampling_strategy: 'stratified', 'random', or 'learning_curve'
        stratify_column: Column to stratify by
        estimate_optimal: Whether to estimate optimal sample size
        min_samples: Minimum sample size
        random_state: Random state for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    # If max_samples is None or greater than dataset size, return full dataset
    if max_samples is None or max_samples >= len(df):
        logging.info(f"Using full dataset of size {len(df)}")
        return df
    
    # If requested, estimate optimal sample size
    if estimate_optimal:
        from sklearn.linear_model import LogisticRegression
        
        def lr_model_fn():
            return LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
        
        logging.info("Estimating optimal sample size...")
        optimal_sample_size, _ = estimate_optimal_sample_size(
            df, lr_model_fn, eval_metric='f1'
        )
        
        # Apply constraints
        optimal_sample_size = max(min_samples, min(optimal_sample_size, max_samples, len(df)))
        logging.info(f"Using estimated optimal sample size: {optimal_sample_size}")
        max_samples = optimal_sample_size
    
    # Perform sampling based on strategy
    if sampling_strategy == 'stratified':
        return create_stratified_sample(df, max_samples, stratify_column, random_state)
    elif sampling_strategy == 'random':
        return df.sample(max_samples, random_state=random_state)
    else:
        logging.warning(f"Unknown sampling strategy: {sampling_strategy}, using stratified")
        return create_stratified_sample(df, max_samples, stratify_column, random_state)

# Add to load_and_preprocess in preprocessing.py
def load_and_preprocess_with_smart_sampling(dataset_name, balance=True, progress_tracker=None, 
                                         sample_dataset=False, max_samples=100000, 
                                         sampling_strategy='stratified', estimate_optimal=True):
    """
    Modified load_and_preprocess function with intelligent sampling.
    
    Replaces the existing load_and_preprocess function in preprocessing.py
    """
    # This would be a wrapper around your existing load_and_preprocess function
    # that adds the smart sampling capabilities
    
    # First, load the dataset using your existing code
    # ...
    
    # Then, apply smart sampling if requested
    if sample_dataset:
        df = smart_sampling(
            df, 
            max_samples=max_samples,
            sampling_strategy=sampling_strategy,
            estimate_optimal=estimate_optimal
        )
    
    # Continue with your existing preprocessing
    # ...
    
    return df

# Add to train_model in training.py
def train_model_with_adaptive_batch_size(model, tokenizer, train_df, eval_df=None, output_dir=None,
                                     training_args=None, class_weights=None, use_enhanced_prompt=None,
                                     adaptive_batch_size=True, batch_size_range=None):
    """
    Modified train_model function with adaptive batch sizing.
    
    This is a wrapper around your existing train_model function that adds
    adaptive batch size capabilities.
    """
    # If adaptive batch sizing is enabled, find optimal batch size
    if adaptive_batch_size:
        # Determine device
        device = next(model.parameters()).device
        
        # Estimate optimal batch size
        optimal_batch_size = determine_optimal_batch_size(
            model=model,
            max_sequence_length=512,  # Adjust based on your tokenizer's max length
            available_memory=None  # Auto-detect
        )
        
        logging.info(f"Using optimal batch size: {optimal_batch_size}")
        
        # Create or update training args with optimal batch size
        if training_args is None:
            # Use existing config but update batch size
            config = TRAINING_CONFIG.copy()
            config["batch_size"] = optimal_batch_size
            
            # Create training arguments
            # ... (Your existing code to create training_args)
        else:
            # Update existing training args
            training_args.per_device_train_batch_size = optimal_batch_size
            training_args.per_device_eval_batch_size = optimal_batch_size
    
    # Call the original train_model function with the updated arguments
    from src.training import train_model as original_train_model
    results = original_train_model(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        eval_df=eval_df,
        output_dir=output_dir,
        training_args=training_args,
        class_weights=class_weights,
        use_enhanced_prompt=use_enhanced_prompt
    )
    
    # Add adaptive batch size info to results
    if adaptive_batch_size:
        results["adaptive_batch_size"] = {
            "optimal_batch_size": optimal_batch_size,
            "original_batch_size": TRAINING_CONFIG["batch_size"]
        }
    
    return results

# --- Command-line integration ---

def add_adaptive_scaling_args(parser):
    """
    Add adaptive scaling arguments to an argument parser.
    
    Args:
        parser: ArgumentParser to add arguments to
    
    Returns:
        Updated ArgumentParser
    """
    # Add to existing argument groups
    for action_group in parser._action_groups:
        if action_group.title == 'optional arguments' or action_group.title == 'options':
            # Add adaptive scaling arguments
            action_group.add_argument("--auto-batch-size", action="store_true", help="Automatically determine optimal batch size")
            action_group.add_argument("--estimate-optimal-samples", action="store_true", help="Estimate optimal sample size")
            action_group.add_argument("--min-samples", type=int, default=1000, help="Minimum sample size to use")
            action_group.add_argument("--sample-ratio", type=float, help="Sample ratio of dataset (0.0-1.0)")
            action_group.add_argument("--learning-curve-analysis", action="store_true", help="Perform learning curve analysis to find optimal sample size")
            break
    
    return parser


def process_adaptive_scaling_args(args):
    """
    Process adaptive scaling arguments and return updated args.
    
    Args:
        args: Parsed arguments namespace
        
    Returns:
        Updated arguments namespace
    """
    # Handle sample ratio if provided
    if hasattr(args, 'sample_ratio') and args.sample_ratio is not None:
        if args.sample_ratio > 0 and args.sample_ratio <= 1.0:
            # This will be used to calculate max_samples when dataset size is known
            args.sample = True  # Enable sampling
            logging.info(f"Will sample {args.sample_ratio * 100:.1f}% of dataset")
        else:
            logging.warning(f"Invalid sample ratio {args.sample_ratio}, must be between 0 and 1. Using default.")
    
    # If learning curve analysis is requested, also set estimate_optimal_samples
    if hasattr(args, 'learning_curve_analysis') and args.learning_curve_analysis:
        args.estimate_optimal_samples = True
        logging.info("Learning curve analysis enabled for optimal sample size estimation")
    
    return args


def modify_preprocess_command(original_function):
    """
    Decorator to modify the preprocess_command function to include adaptive sampling.
    
    Args:
        original_function: Original preprocess_command function
        
    Returns:
        Modified function
    """
    def modified_preprocess_command(args):
        # Process adaptive scaling args
        args = process_adaptive_scaling_args(args)
        
        # Calculate max_samples if sample_ratio was provided
        if hasattr(args, 'sample_ratio') and args.sample_ratio is not None and args.sample:
            # We need to load the dataset first to determine its size
            from src.preprocessing import load_dataset
            df = load_dataset(args.dataset)
            if df is not None:
                args.max_samples = int(len(df) * args.sample_ratio)
                logging.info(f"Using max_samples={args.max_samples} based on sample ratio {args.sample_ratio}")
        
        # Call original function
        return original_function(args)
    
    return modified_preprocess_command


def modify_train_command(original_function):
    """
    Decorator to modify the train_command function to include adaptive batch sizing.
    
    Args:
        original_function: Original train_command function
        
    Returns:
        Modified function
    """
    def modified_train_command(args):
        # Process adaptive scaling args
        args = process_adaptive_scaling_args(args)
        
        # If auto-batch-size is enabled, modify args
        if hasattr(args, 'auto_batch_size') and args.auto_batch_size:
            logging.info("Auto batch size determination enabled")
            
            # Import necessary modules
            from src.model import get_tokenizer, load_base_model
            
            # We need to load the model first to determine optimal batch size
            tokenizer = get_tokenizer(args.model)
            base_model = load_base_model(
                model_name=args.model,
                quantize=True,
                device_map="auto"
            )
            
            if base_model is not None:
                try:
                    # Determine optimal batch size
                    from src.adaptive_scaling import determine_optimal_batch_size
                    optimal_batch_size = determine_optimal_batch_size(
                        model=base_model, 
                        max_sequence_length=512  # Default max sequence length
                    )
                    
                    # Update args
                    args.batch_size = optimal_batch_size
                    logging.info(f"Using optimal batch size: {optimal_batch_size}")
                    
                    # Clean up
                    del base_model
                    clean_up_memory()
                except Exception as e:
                    logging.error(f"Error determining optimal batch size: {e}")
                    logging.info("Falling back to default batch size")
        
        # Call original function
        return original_function(args)
    
    return modified_train_command


def modify_pipeline_command(original_function):
    """
    Decorator to modify the pipeline_command function to include all adaptive scaling features.
    
    Args:
        original_function: Original pipeline_command function
        
    Returns:
        Modified function
    """
    def modified_pipeline_command(args):
        # Process adaptive scaling args
        args = process_adaptive_scaling_args(args)
        
        # Apply adaptive batch sizing
        if hasattr(args, 'auto_batch_size') and args.auto_batch_size:
            logging.info("Pipeline will use auto batch size determination")
        
        # Apply optimal sample size estimation
        if hasattr(args, 'estimate_optimal_samples') and args.estimate_optimal_samples:
            logging.info("Pipeline will estimate optimal sample size")
            
            if not args.skip_preprocess:
                # Modify preprocess args to include estimation
                args.sample = True
                
                # Other preprocess args will be handled in preprocess_command
        
        # Call original function
        return original_function(args)
    
    return modified_pipeline_command


def patch_app_main():
    """
    Patch the main app.py module to include adaptive scaling features.
    
    This function should be called at the start of app.py to modify the
    command functions with our adaptive scaling features.
    """
    try:
        import src.app as app
        
        # Patch command functions
        app.preprocess_command = modify_preprocess_command(app.preprocess_command)
        app.train_command = modify_train_command(app.train_command)
        app.pipeline_command = modify_pipeline_command(app.pipeline_command)
        
        # Patch argument parser
        original_setup_parser = app.setup_parser
        
        def patched_setup_parser():
            parser = original_setup_parser()
            return add_adaptive_scaling_args(parser)
        
        app.setup_parser = patched_setup_parser
        
        logging.info("Applied adaptive scaling patches to app.py")
        
        return app
    except ImportError:
        # If src.app doesn't exist, try to patch the current module
        import sys
        current_module = sys.modules['__main__']
        
        # Patch command functions if they exist
        if hasattr(current_module, 'preprocess_command'):
            current_module.preprocess_command = modify_preprocess_command(current_module.preprocess_command)
        
        if hasattr(current_module, 'train_command'):
            current_module.train_command = modify_train_command(current_module.train_command)
        
        if hasattr(current_module, 'pipeline_command'):
            current_module.pipeline_command = modify_pipeline_command(current_module.pipeline_command)
        
        # Patch argument parser if it exists
        if hasattr(current_module, 'setup_parser'):
            original_setup_parser = current_module.setup_parser
            
            def patched_setup_parser():
                parser = original_setup_parser()
                return add_adaptive_scaling_args(parser)
            
            current_module.setup_parser = patched_setup_parser
        
        logging.info("Applied adaptive scaling patches to main module")
        
        return current_module