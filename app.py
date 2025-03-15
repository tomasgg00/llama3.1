"""
app.py: Main application entry point for the misinformation detection pipeline.

This module provides a command-line interface to the pipeline, allowing users to:
- Preprocess datasets
- Train models
- Optimize hyperparameters
- Run inference
- Evaluate model performance

Usage:
    python app.py --help
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import pandas as pd
from datetime import datetime
import shutil

# Setup path for local imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Add this import to enable adaptive scaling features
from integration_code import patch_app_main

# Call this to patch the app with adaptive scaling features
patch_app_main()

from src.utils import setup_logging, ProgressTracker, get_available_device
from src.config import (
    MODEL_CONFIG, TRAINING_CONFIG, PROCESSED_DATA_DIR, 
    MODELS_DIR, RESULTS_DIR, LOGS_DIR
)

def setup_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Misinformation Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess a dataset")
    preprocess_parser.add_argument("--dataset", required=True, help="Dataset name (without extension)")
    preprocess_parser.add_argument("--no-balance", action="store_true", help="Skip dataset balancing")
    preprocess_parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    preprocess_parser.add_argument("--sample", action="store_true", help="Sample the dataset if it's too large")
    preprocess_parser.add_argument("--max-samples", type=int, default=100000, help="Maximum number of samples to use")
    preprocess_parser.add_argument("--sampling-strategy", choices=['stratified', 'random'], default='stratified', help="Sampling strategy to use")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--dataset", required=True, help="Dataset name (preprocessed)")
    train_parser.add_argument("--model", default=MODEL_CONFIG["base_model"], help="Base model to use")
    train_parser.add_argument("--output", help="Output directory for the trained model")
    train_parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"], help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"], help="Number of epochs")
    train_parser.add_argument("--learning-rate", type=float, default=TRAINING_CONFIG["learning_rate"], help="Learning rate")
    train_parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    train_parser.add_argument("--no-enhanced-prompt", action="store_true", help="Disable enhanced prompts")
    
    # Hyperparameter optimization command
    hpo_parser = subparsers.add_parser("optimize", help="Run hyperparameter optimization")
    hpo_parser.add_argument("--dataset", required=True, help="Dataset name (preprocessed)")
    hpo_parser.add_argument("--model", default=MODEL_CONFIG["base_model"], help="Base model to use")
    hpo_parser.add_argument("--output", help="Output directory for optimization results")
    hpo_parser.add_argument("--n-trials", type=int, default=10, help="Number of optimization trials")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference on a text")
    inference_parser.add_argument("--text", help="Text to analyze")
    inference_parser.add_argument("--file", help="File containing texts to analyze (one per line)")
    inference_parser.add_argument("--model", required=True, help="Path to trained model")
    inference_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    inference_parser.add_argument("--ensemble", action="store_true", help="Use ensemble prediction")
    inference_parser.add_argument("--enhanced", action="store_true", help="Use enhanced prompts with features")
    inference_parser.add_argument("--output", help="Output file for results")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--dataset", required=True, help="Dataset name (preprocessed)")
    eval_parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to use")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    eval_parser.add_argument("--output", help="Output directory for evaluation results")
    eval_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    eval_parser.add_argument("--compare", action="store_true", help="Compare basic and enhanced prompts")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline")
    pipeline_parser.add_argument("--dataset", required=True, help="Dataset name (without extension)")
    pipeline_parser.add_argument("--model", default=MODEL_CONFIG["base_model"], help="Base model to use")
    pipeline_parser.add_argument("--output", help="Output directory for all results")
    pipeline_parser.add_argument("--n-trials", type=int, default=10, help="Number of optimization trials")
    pipeline_parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    pipeline_parser.add_argument("--skip-optimize", action="store_true", help="Skip optimization step")
    
    return parser

def preprocess_command(args):
    """Run preprocessing command."""
    from src.preprocessing import load_and_preprocess
    
    logger = logging.getLogger()
    logger.info(f"Preprocessing dataset: {args.dataset}")
    
    # Determine balancing setting
    balance = not args.no_balance
    
    # Run preprocessing
    df = load_and_preprocess(
        args.dataset,
        balance=balance,
        progress_tracker=None,
        sample_dataset=args.sample,
        max_samples=args.max_samples,
        sampling_strategy=args.sampling_strategy
    )
    
    if df is not None:
        logger.info(f"Preprocessing completed successfully. Dataset shape: {df.shape}")
        return True
    else:
        logger.error("Preprocessing failed")
        return False

def train_command(args):
    """Run training command."""
    from src.model import (
        get_tokenizer, load_base_model, prepare_for_lora, 
        apply_lora, get_lora_config
    )
    from src.training import train_model
    from src.preprocessing import load_processed_dataset
    
    logger = logging.getLogger()
    logger.info(f"Training model using dataset: {args.dataset}")
    
    # Set output directory using dataset name 
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(MODELS_DIR, args.dataset)
    
    # Delete the directory if it exists to avoid accumulating files
    if os.path.exists(output_dir):
        logger.info(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    train_df = load_processed_dataset(args.dataset, "train")
    val_df = load_processed_dataset(args.dataset, "val")
    
    if train_df is None or val_df is None:
        logger.error("Failed to load datasets")
        return False
    
    # Load tokenizer
    tokenizer = get_tokenizer(args.model)
    if tokenizer is None:
        logger.error("Failed to load tokenizer")
        return False
    
    # Load base model
    base_model = load_base_model(
        model_name=args.model,
        quantize=True,
        device_map="auto"
    )
    
    if base_model is None:
        logger.error("Failed to load base model")
        return False
    
    # Configure LoRA
    lora_config = get_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Prepare model for LoRA and apply LoRA
    prepared_model = prepare_for_lora(base_model)
    model = apply_lora(prepared_model, lora_config)
    
    # Configure training arguments
    training_args = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "use_enhanced_prompt": not args.no_enhanced_prompt
    }
    
    # Train model
    results = train_model(
    model=model,
    tokenizer=tokenizer,
    train_df=train_df,
    eval_df=val_df,
    output_dir=output_dir,
    training_args=None,  # Use defaults
    class_weights=None,  # Calculate automatically
    use_enhanced_prompt=not args.no_enhanced_prompt,
    auto_batch_size=args.auto_batch_size if hasattr(args, 'auto_batch_size') else False
    )
    
    if results:
        logger.info(f"Training completed successfully")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Evaluation results: Accuracy={results['eval_results']['eval_accuracy']:.4f}, F1={results['eval_results']['eval_f1']:.4f}")
        return True
    else:
        logger.error("Training failed")
        return False

def optimize_command(args):
    """Run hyperparameter optimization command."""
    from src.training import run_hyperparameter_optimization, train_with_best_params
    from src.preprocessing import load_processed_dataset
    
    logger = logging.getLogger()
    logger.info(f"Running hyperparameter optimization using dataset: {args.dataset}")
    
    # Set output directory using dataset name
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(MODELS_DIR, f"hpo_{args.dataset}")
    
    # Delete the directory if it exists to avoid accumulating files
    if os.path.exists(output_dir):
        logger.info(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    train_df = load_processed_dataset(args.dataset, "train")
    val_df = load_processed_dataset(args.dataset, "val")
    
    if train_df is None or val_df is None:
        logger.error("Failed to load datasets")
        return False
    
    # Run hyperparameter optimization
    study_results = run_hyperparameter_optimization(
        train_df=train_df,
        eval_df=val_df,
        base_model_name=args.model,
        output_dir=output_dir,
        n_trials=args.n_trials,
        device="auto"
    )
    
    if study_results:
        logger.info(f"Hyperparameter optimization completed successfully")
        logger.info(f"Best parameters: {study_results['best_trial']['params']}")
        
        # Train final model with best parameters and save in dataset folder
        final_model_dir = os.path.join(MODELS_DIR, args.dataset)
        
        # Delete the directory if it exists to avoid accumulating files
        if os.path.exists(final_model_dir):
            logger.info(f"Removing existing directory: {final_model_dir}")
            shutil.rmtree(final_model_dir)
            
        os.makedirs(final_model_dir, exist_ok=True)
        
        logger.info(f"Training final model with best parameters")
        
        results = train_with_best_params(
            train_df=train_df,
            eval_df=val_df,
            base_model_name=args.model,
            best_params=study_results["best_trial"]["params"],
            output_dir=final_model_dir
        )
        
        if results:
            logger.info(f"Final model training completed successfully")
            logger.info(f"Final model saved to: {final_model_dir}")
            return True
        else:
            logger.error("Final model training failed")
            return False
    else:
        logger.error("Hyperparameter optimization failed")
        return False

# Additions for app.py to enable DeepSpeed

def add_deepspeed_args(parser):
    """Add DeepSpeed arguments to the parser."""
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed for training")
    parser.add_argument("--zero-stage", type=int, choices=[0, 1, 2, 3], 
                       help="DeepSpeed ZeRO stage (0, 1, 2, or 3)")
    parser.add_argument("--offload-optimizer", action="store_true", 
                       help="Offload optimizer states to CPU")
    parser.add_argument("--offload-parameters", action="store_true", 
                       help="Offload parameters to CPU (ZeRO-3 only)")
    parser.add_argument("--no-auto-detect", action="store_true", 
                       help="Disable auto-detection of optimal DeepSpeed config")
    return parser

def setup_parser_with_deepspeed():
    """Set up command-line argument parser with DeepSpeed arguments."""
    parser = setup_parser()  # Call original setup_parser
    
    # Add DeepSpeed arguments to specific commands
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for cmd, subparser in action.choices.items():
                # Add to train, optimize, and pipeline commands
                if cmd in ['train', 'optimize', 'pipeline']:
                    add_deepspeed_args(subparser)
    
    return parser

def modify_train_command(args):
    """Modified train_command function that supports DeepSpeed."""
    from src.model import (
        get_tokenizer, load_base_model, prepare_for_lora, 
        apply_lora, get_lora_config
    )
    from src.training import train_model, train_model_with_deepspeed
    from src.preprocessing import load_processed_dataset
    
    logger = logging.getLogger()
    logger.info(f"Training model using dataset: {args.dataset}")
    
    # Set output directory using dataset name 
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(MODELS_DIR, args.dataset)
    
    # Delete the directory if it exists to avoid accumulating files
    if os.path.exists(output_dir):
        logger.info(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    train_df = load_processed_dataset(args.dataset, "train")
    val_df = load_processed_dataset(args.dataset, "val")
    
    if train_df is None or val_df is None:
        logger.error("Failed to load datasets")
        return False
    
    # Load tokenizer
    tokenizer = get_tokenizer(args.model)
    if tokenizer is None:
        logger.error("Failed to load tokenizer")
        return False
    
    # Load base model
    base_model = load_base_model(
        model_name=args.model,
        quantize=True,
        device_map="auto"
    )
    
    if base_model is None:
        logger.error("Failed to load base model")
        return False
    
    # Configure LoRA
    lora_config = get_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Prepare model for LoRA and apply LoRA
    prepared_model = prepare_for_lora(base_model)
    model = apply_lora(prepared_model, lora_config)
    
    # Check if DeepSpeed is requested
    if hasattr(args, 'deepspeed') and args.deepspeed:
        # Configure DeepSpeed
        deepspeed_config = {
            "zero_stage": args.zero_stage if args.zero_stage is not None else 2,
            "offload_optimizer": args.offload_optimizer,
            "offload_parameters": args.offload_parameters,
            "auto_detect": not args.no_auto_detect,
            "fp16": True,
            "gradient_accumulation_steps": 4
        }
        
        # Train with DeepSpeed
        logger.info("Using DeepSpeed for training")
        results = train_model_with_deepspeed(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            eval_df=val_df,
            output_dir=output_dir,
            class_weights=None,  # Calculate automatically
            use_enhanced_prompt=not args.no_enhanced_prompt,
            deepspeed_config=deepspeed_config
        )
    else:
        # Use regular training
        results = train_model(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            eval_df=val_df,
            output_dir=output_dir,
            training_args=None,  # Use defaults
            class_weights=None,  # Calculate automatically
            use_enhanced_prompt=not args.no_enhanced_prompt
        )
    
    if results:
        logger.info(f"Training completed successfully")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Evaluation results: Accuracy={results['eval_results']['eval_accuracy']:.4f}, F1={results['eval_results']['eval_f1']:.4f}")
        return True
    else:
        logger.error("Training failed")
        return False

def inference_command(args):
    """Run inference command."""
    from src.inference import detect_misinformation, extract_features_from_text, load_and_prepare_model
    import json
    
    logger = logging.getLogger()
    
    # Load model
    logger.info(f"Loading model from: {args.model}")
    model, tokenizer = load_and_prepare_model(args.model)
    
    if model is None or tokenizer is None:
        logger.error("Failed to load model")
        return False
    
    # Check if we have text input
    if not args.text and not args.file:
        logger.error("No input provided. Please specify either --text or --file")
        return False
    
    results = []
    
    if args.text:
        # Process single text
        logger.info("Processing single text input")
        text = args.text
        
        # Extract features if enhanced prompting is requested
        features = None
        if args.enhanced:
            features = extract_features_from_text(text)
        
        # Run inference
        result = detect_misinformation(
            text=text,
            model=model,
            tokenizer=tokenizer,
            features=features,
            use_ensemble=args.ensemble
        )
        
        # Print result
        logger.info(f"Result: {result['prediction']} (Confidence: {result['confidence']:.4f})")
        results.append(result)
    
    if args.file:
        # Process file
        logger.info(f"Processing texts from file: {args.file}")
        
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return False
        
        # Read lines from file
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(texts)} texts")
        
        # Process in batches
        from src.inference import batch_inference
        
        # Extract features if enhanced prompting is requested
        features_list = None
        if args.enhanced:
            features_list = [extract_features_from_text(text) for text in texts]
        
        # Run batch inference
        batch_results = batch_inference(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            features_list=features_list,
            batch_size=args.batch_size,
            use_enhanced_prompt=args.enhanced
        )
        
        results.extend(batch_results)
        
        # Print summary
        true_count = sum(1 for r in results if r["prediction"] == "TRUE (Factual)")
        false_count = sum(1 for r in results if r["prediction"] == "FALSE (Misinformation)")
        
        logger.info(f"Processed {len(results)} texts")
        logger.info(f"Results: {true_count} TRUE (Factual), {false_count} FALSE (Misinformation)")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    
    return True

def evaluate_command(args):
    """Run evaluation command."""
    from src.inference import evaluate_model, benchmark_comparison, load_and_prepare_model
    from src.preprocessing import load_processed_dataset
    
    logger = logging.getLogger()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f"evaluation_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    test_df = load_processed_dataset(args.dataset, args.split)
    
    if test_df is None:
        logger.error(f"Failed to load dataset: {args.dataset}, split: {args.split}")
        return False
    
    # Load model
    logger.info(f"Loading model from: {args.model}")
    model, tokenizer = load_and_prepare_model(args.model)
    
    if model is None or tokenizer is None:
        logger.error("Failed to load model")
        return False
    
    # Run evaluation
    if args.compare:
        # Run benchmark comparison
        logger.info("Running benchmark comparison between basic and enhanced prompts")
        
        results = benchmark_comparison(
            model=model,
            tokenizer=tokenizer,
            test_df=test_df,
            output_dir=output_dir
        )
        
        if results:
            logger.info("Benchmark comparison completed successfully")
            
            if "comparison" in results:
                comparison = results["comparison"]
                logger.info("Comparison results:")
                logger.info(f"Accuracy: Basic={comparison['metrics']['accuracy']['basic']:.4f}, Enhanced={comparison['metrics']['accuracy']['enhanced']:.4f}, Improvement={comparison['metrics']['accuracy']['percent_improvement']:.2f}%")
                logger.info(f"F1 Score: Basic={comparison['metrics']['f1']['basic']:.4f}, Enhanced={comparison['metrics']['f1']['enhanced']:.4f}, Improvement={comparison['metrics']['f1']['percent_improvement']:.2f}%")
            
            return True
        else:
            logger.error("Benchmark comparison failed")
            return False
    else:
        # Run standard evaluation
        logger.info(f"Evaluating model on {len(test_df)} examples")
        
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_df=test_df,
            output_dir=output_dir,
            batch_size=args.batch_size
        )
        
        if results:
            logger.info("Evaluation completed successfully")
            logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            logger.info(f"F1 Score: {results['metrics']['f1']:.4f}")
            logger.info(f"AUC: {results['metrics']['auc']:.4f}")
            logger.info(f"Results saved to: {output_dir}")
            return True
        else:
            logger.error("Evaluation failed")
            return False

def pipeline_command(args):
    """Run the full pipeline."""
    logger = logging.getLogger()
    logger.info("Running full pipeline")
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(MODELS_DIR, f"pipeline_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Track progress
    pipeline_steps = []
    if not args.skip_preprocess:
        pipeline_steps.append("preprocess")
    if not args.skip_optimize:
        pipeline_steps.append("optimize")
    pipeline_steps.extend(["train", "evaluate"])
    
    progress = ProgressTracker(total_steps=len(pipeline_steps))
    
    # Step 1: Preprocess
    if not args.skip_preprocess:
        progress.start_step("Preprocessing dataset")
        
        # Create preprocess args
        preprocess_args = argparse.Namespace(
            dataset=args.dataset,
            no_balance=False,
            batch_size=1000,
            sample=args.sample,
            max_samples=args.max_samples,
            sampling_strategy=args.sampling_strategy
        )
        
        success = preprocess_command(preprocess_args)
        if not success:
            logger.error("Pipeline failed at preprocessing step")
            return False
        
        progress.complete_step()
    
    # Step 2: Hyperparameter optimization
    best_params = None
    if not args.skip_optimize:
        progress.start_step("Hyperparameter optimization")
        
        # Create optimize args
        optimize_args = argparse.Namespace(
            dataset=args.dataset,
            model=args.model,
            output=os.path.join(output_dir, "optimization"),
            n_trials=args.n_trials
        )
        
        success = optimize_command(optimize_args)
        if not success:
            logger.error("Pipeline failed at optimization step")
            return False
        
        # Load best parameters
        import json
        best_params_file = os.path.join(output_dir, "optimization", "hpo_results.json")
        try:
            with open(best_params_file, 'r') as f:
                hpo_results = json.load(f)
                best_params = hpo_results["best_trial"]["params"]
                logger.info(f"Loaded best parameters: {best_params}")
        except Exception as e:
            logger.error(f"Failed to load best parameters: {e}")
            best_params = None
        
        progress.complete_step()
    
    # Step 3: Train with best parameters
    progress.start_step("Training model")
    
    # Final model will be in MODELS_DIR/dataset_name
    model_output_dir = os.path.join(MODELS_DIR, args.dataset)
    
    if best_params is not None:
        # Use best parameters from optimization
        train_args = argparse.Namespace(
            dataset=args.dataset,
            model=args.model,
            output=model_output_dir,
            batch_size=best_params.get("batch_size", 4),
            epochs=10,  # Use more epochs for final training
            learning_rate=best_params.get("learning_rate", 2e-5),
            lora_r=best_params.get("lora_r", 32),
            lora_alpha=best_params.get("lora_alpha", 64),
            no_enhanced_prompt=not best_params.get("use_enhanced_prompt", True)
        )
    else:
        # Use default parameters
        train_args = argparse.Namespace(
            dataset=args.dataset,
            model=args.model,
            output=model_output_dir,
            batch_size=TRAINING_CONFIG["batch_size"],
            epochs=TRAINING_CONFIG["epochs"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            lora_r=32,
            lora_alpha=64,
            no_enhanced_prompt=False
        )
    
    success = train_command(train_args)
    if not success:
        logger.error("Pipeline failed at training step")
        return False
    
    progress.complete_step()
    
    # Step 4: Evaluate
    progress.start_step("Evaluating model")
    
    # Create evaluate args
    evaluate_args = argparse.Namespace(
        dataset=args.dataset,
        split="test",
        model=model_output_dir,  # Now this is simple - just use the dataset folder
        output=os.path.join(output_dir, "evaluation"),
        batch_size=8,
        compare=True  # Always run comparison for full pipeline
    )
    
    success = evaluate_command(evaluate_args)
    if not success:
        logger.error("Pipeline failed at evaluation step")
        return False
    
    progress.complete_step()
    
    # Pipeline completed successfully
    logger.info("Pipeline completed successfully!")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info(f"Model saved to: {model_output_dir}")
    
    return True

def main():
    """Main entry point."""
    # Set up logging
    logger = setup_logging()
    
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Report available device
    device = get_available_device()
    logger.info(f"Using device: {device}")
    
    # Check command
    try:
        if args.command == "preprocess":
            success = preprocess_command(args)
        elif args.command == "train":
            success = train_command(args)
        elif args.command == "optimize":
            success = optimize_command(args)
        elif args.command == "inference":
            success = inference_command(args)
        elif args.command == "evaluate":
            success = evaluate_command(args)
        elif args.command == "pipeline":
            success = pipeline_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        return 0 if success else 1
    except Exception as e:
        logger.exception(f"Error running command {args.command}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())