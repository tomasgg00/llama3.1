"""
training.py: Training module for the misinformation detection pipeline.

This module handles:
- Model training setup
- Custom trainer classes
- Hyperparameter optimization with Optuna
- Early stopping and model checkpointing
- Training metrics and evaluation
"""
import os
import json
import time
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from functools import partial
import gc

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
import optuna

from src.utils import (
    compute_metrics, clean_up_memory, save_json, 
    visualize_confusion_matrix, format_time
)
from src.config import (
    TRAINING_CONFIG, MODELS_DIR, RESULTS_DIR, LOGS_DIR,
    OPTUNA_CONFIG, IMPORTANT_FEATURES
)
from src.model import get_lora_config

def determine_optimal_training_params(model, train_dataset, eval_dataset=None):
    """
    Determine optimal training parameters (batch size, etc.) for a given model and dataset.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
    
    Returns:
        Dictionary with optimal parameters
    """
    try:
        # Import adaptive scaling features
        from src.adaptive_scaling import determine_optimal_batch_size
        
        # Determine optimal batch size
        optimal_batch_size = determine_optimal_batch_size(
            model=model,
            max_sequence_length=512,  # Adjust based on your tokenizer's max length
            available_memory=None  # Auto-detect
        )
        
        return {
            "batch_size": optimal_batch_size,
            "auto_determined": True
        }
    except Exception as e:
        logging.error(f"Error determining optimal training parameters: {e}")
        return {
            "batch_size": TRAINING_CONFIG["batch_size"],
            "auto_determined": False
        }

class MisinformationTrainer(Trainer):
    """Custom trainer with weighted loss function for misinformation detection."""
    
    def __init__(self, class_weights=None, **kwargs):
        """
        Initialize the trainer.
        
        Args:
            class_weights: Optional weights for loss calculation
            **kwargs: Additional arguments for Trainer
        """
        super().__init__(**kwargs)
        
        self.class_weights = class_weights
        self.start_time = time.time()
        self.step_times = []
        
        # Log training start
        logging.info(f"Initializing trainer with class weights: {class_weights}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with optional class weighting.
        
        Args:
            model: Model to compute loss for
            inputs: Inputs to the model
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in the batch (for weighted loss)
        
        Returns:
            Loss value, or tuple of (loss, outputs) if return_outputs is True
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            # Apply weighted loss
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        # Record step time for efficiency metrics
        step_time = time.time() - self.start_time
        self.step_times.append(step_time)
        self.start_time = time.time()
        
        return (loss, outputs) if return_outputs else loss
    
    def get_efficiency_metrics(self):
        """
        Calculate efficiency metrics based on timing data.
        
        Returns:
            Dictionary with efficiency metrics
        """
        if not self.step_times:
            return {}
            
        avg_step_time = np.mean(self.step_times)
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        
        return {
            "avg_step_time": avg_step_time,
            "steps_per_second": steps_per_second,
            "total_steps": len(self.step_times),
            "total_training_time": sum(self.step_times)
        }

class SaveBestModelCallback(EarlyStoppingCallback):
    """Callback to save the best model during training."""
    
    def __init__(self, output_dir, **kwargs):
        """
        Initialize the callback.
        
        Args:
            output_dir: Directory to save the best model to
            **kwargs: Additional arguments for EarlyStoppingCallback
        """
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.best_metric = None
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Called after evaluation.
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            metrics: Dict of metrics
            **kwargs: Additional arguments
        """
        # Call parent implementation first
        super().on_evaluate(args, state, control, metrics, **kwargs)
        
        # Extract the metric we're monitoring
        metric_to_check = f"eval_{args.metric_for_best_model}"
        if metric_to_check not in metrics:
            return
            
        metric_value = metrics[metric_to_check]
        
        # Check if this is the best model
        if self.best_metric is None or (args.greater_is_better and metric_value > self.best_metric) or (not args.greater_is_better and metric_value < self.best_metric):
            self.best_metric = metric_value
            
            # Save the best model directly to the output directory (not in a subdirectory)
            best_dir = self.output_dir
            
            if 'model' in kwargs:
                model = kwargs['model']
                model.save_pretrained(best_dir)
                logging.info(f"🔥 New best model saved with {args.metric_for_best_model} = {metric_value}")
                
                # Save training args and metrics
                metrics_file = os.path.join(best_dir, "metrics.json")
                save_json(metrics, metrics_file)
                
                args_file = os.path.join(best_dir, "training_args.json")
                save_json(args.to_dict(), args_file)

def prepare_dataset_for_training(df, tokenizer, use_enhanced_prompt=False):
    """
    Prepare dataset for training.
    
    Args:
        df: DataFrame with dataset
        tokenizer: Tokenizer for encoding
        use_enhanced_prompt: Whether to use enhanced prompts
    
    Returns:
        Dictionary with train and eval datasets
    """
    from src.preprocessing import preprocess_for_training
    
    # Check if dataset contains required columns
    if "content" not in df.columns or "label" not in df.columns:
        logging.error("Dataset missing required columns: content, label")
        return None
    
    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Keep only essential columns
    essential_columns = ['content', 'label']
    if use_enhanced_prompt:
        essential_columns.extend([col for col in IMPORTANT_FEATURES if col in dataset.column_names])
    
    keep_columns = [col for col in dataset.column_names if col in essential_columns]
    dataset = dataset.select_columns(keep_columns)
    
    # Preprocess with tokenizer
    preprocess_fn = partial(
        preprocess_for_training, 
        tokenizer=tokenizer, 
        use_enhanced_prompt=use_enhanced_prompt,
        dataset_df=df if use_enhanced_prompt else None
    )
    
    # Process the dataset
    tokenized_dataset = dataset.map(
        preprocess_fn, 
        batched=True,
        remove_columns=['content'] + [col for col in dataset.column_names if col in IMPORTANT_FEATURES]
    )
    
    return tokenized_dataset

def train_model(
    model,
    tokenizer,
    train_df,
    eval_df=None,
    output_dir=None,
    training_args=None,
    class_weights=None,
    use_enhanced_prompt=None,
    auto_batch_size=False  # Add this parameter
):
    """
    Train a model on the given dataset.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer for encoding
        train_df: Training dataset
        eval_df: Evaluation dataset
        output_dir: Directory to save model to
        training_args: Custom training arguments
        class_weights: Optional weights for loss calculation
        use_enhanced_prompt: Whether to use enhanced prompts
        auto_batch_size: Whether to automatically determine optimal batch size
    
    Returns:
        Dictionary with training results
    """
    # Set default output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(MODELS_DIR, f"model_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define compute metrics wrapper
    def compute_metrics_wrapper(eval_pred):
        """Wrapper for compute_metrics to match Trainer's expected interface"""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        return compute_metrics(predictions, labels)
    
    # Determine whether to use enhanced prompts
    if use_enhanced_prompt is None:
        use_enhanced_prompt = TRAINING_CONFIG["use_enhanced_prompt"]
    
    # Prepare datasets
    logging.info("Preparing datasets for training")
    
    train_dataset = prepare_dataset_for_training(
        train_df, tokenizer, use_enhanced_prompt=use_enhanced_prompt
    )
    
    if eval_df is not None:
        eval_dataset = prepare_dataset_for_training(
            eval_df, tokenizer, use_enhanced_prompt=use_enhanced_prompt
        )
    else:
        # Use a portion of training data for evaluation if no eval set provided
        dataset_dict = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
    
    # Log dataset information
    logging.info(f"Training dataset: {len(train_dataset)} examples")
    logging.info(f"Evaluation dataset: {len(eval_dataset)} examples")
    
    # Extract label distribution (for logging and class weights)
    train_0_indices = [i for i, example in enumerate(train_dataset) if example['labels'] == 0]
    train_1_indices = [i for i, example in enumerate(train_dataset) if example['labels'] == 1]
    
    train_0_count = len(train_0_indices)
    train_1_count = len(train_1_indices)
    
    logging.info(f"Train labels: {train_0_count} FALSE (misinfo), {train_1_count} TRUE (factual)")
    
    # Calculate class weights if not provided
    if class_weights is None and (train_0_count > 0 and train_1_count > 0):
        weights = torch.tensor([
            1.0 * (train_0_count + train_1_count) / (2 * train_0_count),
            1.0 * (train_0_count + train_1_count) / (2 * train_1_count)
        ])
        logging.info(f"Calculated class weights: {weights}")
        class_weights = weights
    
    # Determine optimal batch size if requested
    optimal_batch_size = None
    if auto_batch_size:
        logging.info("Determining optimal batch size...")
        optimal_params = determine_optimal_training_params(model, train_dataset)
        optimal_batch_size = optimal_params["batch_size"]
        logging.info(f"Using optimal batch size: {optimal_batch_size}")
    
    # Set up training arguments
    if training_args is None:
        # Use default configuration
        config = TRAINING_CONFIG.copy()
        
        # Update batch size if auto_batch_size is enabled
        if auto_batch_size and optimal_batch_size is not None:
            config["batch_size"] = optimal_batch_size
        
        # Calculate training steps and warmup steps
        num_train_samples = len(train_dataset)
        total_train_steps = (
            num_train_samples // 
            (config["batch_size"] * config["gradient_accumulation_steps"])
        ) * config["epochs"]
        
        warmup_steps = int(config["warmup_ratio"] * total_train_steps)
        
        logging.info(f"Total training steps: {total_train_steps}, warmup steps: {warmup_steps}")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=config["eval_steps"],
            save_strategy="steps",
            save_steps=config["save_steps"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["epochs"],
            fp16=True,
            gradient_checkpointing=True,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            logging_dir=os.path.join(LOGS_DIR, "training_logs"),
            logging_steps=25,
            report_to="none",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            remove_unused_columns=True,
        )
    elif auto_batch_size and optimal_batch_size is not None:
        # Update existing training args with optimal batch size
        training_args.per_device_train_batch_size = optimal_batch_size
        training_args.per_device_eval_batch_size = optimal_batch_size
    
    # Create custom data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Add callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.005
        ),
        SaveBestModelCallback(
            output_dir=output_dir,
            early_stopping_patience=3,
            early_stopping_threshold=0.005
        )
    ]
    
    # Create trainer
    trainer = MisinformationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper,
        callbacks=callbacks,
        data_collator=data_collator,
        class_weights=class_weights
    )
    
    # Start training
    logging.info("Starting training")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Final evaluation
    logging.info("Performing final evaluation")
    eval_results = trainer.evaluate()
    
    # Save model
    logging.info("Saving trained model")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Get efficiency metrics
    efficiency_metrics = trainer.get_efficiency_metrics()
    
    # Combine metrics
    results = {
        "training_time": training_time,
        "training_time_formatted": format_time(training_time),
        "samples_per_second": len(train_dataset) / training_time,
        "eval_results": eval_results,
        "efficiency_metrics": efficiency_metrics,
        "hyperparameters": {
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "epochs": training_args.num_train_epochs,
            "use_enhanced_prompt": use_enhanced_prompt
        }
    }
    
    # Add adaptive batch size info if used
    if auto_batch_size and optimal_batch_size is not None:
        results["adaptive_batch_size"] = {
            "optimal_batch_size": optimal_batch_size,
            "original_batch_size": TRAINING_CONFIG["batch_size"]
        }
    
    # Save results
    results_file = os.path.join(output_dir, "training_results.json")
    save_json(results, results_file)
    
    # Generate metrics visualization
    try:
        # Run predictions on evaluation dataset
        predictions = trainer.predict(eval_dataset)
        y_true = predictions.label_ids
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Create confusion matrix visualization
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        visualize_confusion_matrix(
            y_true, 
            y_pred,
            output_path=cm_path, 
            title="Confusion Matrix for Misinformation Detection"
        )
        
        # Add visualization paths to results
        results["visualizations"] = {
            "confusion_matrix": cm_path
        }
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
    
    logging.info(f"Training completed. Results saved to {results_file}")
    return results

def objective(
    trial,
    train_df,
    eval_df,
    base_model_name,
    output_dir,
    device="auto"
):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        train_df: Training dataset
        eval_df: Evaluation dataset
        base_model_name: Base model name
        output_dir: Output directory
        device: Device to use
    
    Returns:
        Optimization score (F1 score)
    """
    try:
        # Create trial directory
        trial_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_trial_{trial.number}"
        trial_dir = os.path.join(output_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        
        # Define compute metrics wrapper
        def compute_metrics_wrapper(eval_pred):
            """Wrapper for compute_metrics to match Trainer's expected interface"""
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
            return compute_metrics(predictions, labels)
        
        # Sample hyperparameters
        search_space = OPTUNA_CONFIG["search_space"]
        
        learning_rate = trial.suggest_float(
            "learning_rate", 
            search_space["learning_rate"][0], 
            search_space["learning_rate"][1], 
            log=True
        )
        
        batch_size = trial.suggest_categorical(
            "batch_size", 
            search_space["batch_size"]
        )
        
        lora_r = trial.suggest_categorical(
            "lora_r", 
            search_space["lora_r"]
        )
        
        lora_alpha = trial.suggest_categorical(
            "lora_alpha", 
            search_space["lora_alpha"]
        )
        
        lora_dropout = trial.suggest_float(
            "lora_dropout", 
            search_space["lora_dropout"][0], 
            search_space["lora_dropout"][1]
        )
        
        use_enhanced_prompt = trial.suggest_categorical(
            "use_enhanced_prompt", 
            search_space["use_enhanced_prompt"]
        )
        
        gradient_accumulation_steps = trial.suggest_categorical(
            "gradient_accumulation_steps", 
            search_space["gradient_accumulation_steps"]
        )
        
        # Log hyperparameters
        logging.info(f"Trial {trial.number}: Hyperparameters: {trial.params}")
        
        # Import here to avoid circular imports
        from src.model import (
            get_tokenizer, load_base_model, prepare_for_lora, apply_lora
        )
        
        # Load tokenizer
        tokenizer = get_tokenizer(base_model_name)
        if tokenizer is None:
            raise ValueError("Failed to load tokenizer")
        
        # Load base model
        base_model = load_base_model(
            model_name=base_model_name,
            num_labels=2,
            device_map=device,
            quantize=True
        )
        
        if base_model is None:
            raise ValueError("Failed to load base model")
        
        # Create LoRA config
        lora_config = get_lora_config(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # Prepare model for LoRA and apply LoRA
        prepared_model = prepare_for_lora(base_model)
        model = apply_lora(prepared_model, lora_config)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=trial_dir,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=5,  # Use early stopping to control actual training length
            fp16=True,
            gradient_checkpointing=True,
            lr_scheduler_type="linear",
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(trial_dir, "logs"),
            logging_steps=25,
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            remove_unused_columns=True,
        )
        
        # Train the model
        results = train_model(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            eval_df=eval_df,
            output_dir=trial_dir,
            training_args=training_args,
            use_enhanced_prompt=use_enhanced_prompt,
            auto_batch_size=False  # Don't use auto batch size in trials
        )
        
        # Extract F1 score
        f1_score = results["eval_results"].get("eval_f1", 0)
        
        # Clean up to avoid memory issues
        del model, base_model, prepared_model
        clean_up_memory()
        
        # Save trial results
        trial_results = {
            "trial_number": trial.number,
            "params": trial.params,
            "f1_score": f1_score,
            "results": results
        }
        
        results_file = os.path.join(trial_dir, "trial_results.json")
        save_json(trial_results, results_file)
        
        return f1_score
    
    except Exception as e:
        logging.error(f"Error in trial {trial.number}: {e}")
        # Clean up on error
        clean_up_memory()
        return -1

def run_hyperparameter_optimization(
    train_df,
    eval_df,
    base_model_name,
    output_dir=None,
    n_trials=None,
    device="auto"
):
    """
    Run hyperparameter optimization with Optuna.
    
    Args:
        train_df: Training dataset
        eval_df: Evaluation dataset
        base_model_name: Base model name
        output_dir: Output directory
        n_trials: Number of trials
        device: Device to use
    
    Returns:
        Dictionary with optimization results
    """
    # Set default values
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(MODELS_DIR, f"hpo_{timestamp}")
    
    if n_trials is None:
        n_trials = OPTUNA_CONFIG["n_trials"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Starting hyperparameter optimization with {n_trials} trials")
    logging.info(f"Output directory: {output_dir}")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run optimization
    study.optimize(
        partial(
            objective, 
            train_df=train_df,
            eval_df=eval_df,
            base_model_name=base_model_name,
            output_dir=output_dir,
            device=device
        ),
        n_trials=n_trials,
        timeout=None,
        catch=(Exception,)
    )
    
    # Get best trial
    best_trial = study.best_trial
    logging.info(f"Best trial: {best_trial.number}")
    logging.info(f"Best value: {best_trial.value}")
    logging.info(f"Best hyperparameters: {best_trial.params}")
    
    # Save study results
    study_results = {
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params
        },
        "all_trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state)
            }
            for trial in study.trials
        ]
    }
    
    results_file = os.path.join(output_dir, "hpo_results.json")
    save_json(study_results, results_file)
    
    # Generate visualization of hyperparameter importance
    try:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "optimization_history.png"))
        
        # Plot parameter importances
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "param_importances.png"))
        
        # Plot parallel coordinate plot
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parallel_coordinate.png"))
        
    except Exception as e:
        logging.warning(f"Could not generate visualizations: {e}")
    
    return study_results

def train_with_best_params(
    train_df,
    eval_df,
    base_model_name,
    best_params,
    output_dir=None,
    auto_batch_size=False  # Add this parameter
):
    """
    Train the final model with the best hyperparameters.
    
    Args:
        train_df: Training dataset
        eval_df: Evaluation dataset
        base_model_name: Base model name
        best_params: Best hyperparameters
        output_dir: Output directory
        auto_batch_size: Whether to automatically determine optimal batch size
    
    Returns:
        Dictionary with training results
    """
    # Set default output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(MODELS_DIR, f"final_model_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Training final model with best hyperparameters: {best_params}")
    
    # Import here to avoid circular imports
    from src.model import (
        get_tokenizer, load_base_model, 
        prepare_for_lora, apply_lora, get_lora_config
    )
    
    # Extract best params
    learning_rate = best_params.get("learning_rate", 2e-5)
    batch_size = best_params.get("batch_size", 4)
    lora_r = best_params.get("lora_r", 32)
    lora_alpha = best_params.get("lora_alpha", 64)
    lora_dropout = best_params.get("lora_dropout", 0.1)
    use_enhanced_prompt = best_params.get("use_enhanced_prompt", True)
    gradient_accumulation_steps = best_params.get("gradient_accumulation_steps", 4)
    
    # Load tokenizer
    tokenizer = get_tokenizer(base_model_name)
    if tokenizer is None:
        raise ValueError("Failed to load tokenizer")
    
    # Load base model
    base_model = load_base_model(
        model_name=base_model_name,
        num_labels=2,
        device_map="auto",
        quantize=True
    )
    
    if base_model is None:
        raise ValueError("Failed to load base model")
    
    # Prepare model for LoRA and apply LoRA
    lora_config = get_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    prepared_model = prepare_for_lora(base_model, use_gradient_checkpointing=True)
    model = apply_lora(prepared_model, lora_config)
    
    # Configure training arguments (with longer training and more patience)
    epochs = 10  # Use early stopping for actual length
    
    # Calculate training steps and warmup steps
    num_train_samples = len(train_df)
    total_train_steps = (
        num_train_samples // 
        (batch_size * gradient_accumulation_steps)
    ) * epochs
    
    warmup_steps = int(0.1 * total_train_steps)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        fp16=True,
        gradient_checkpointing=True,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(LOGS_DIR, "training_logs"),
        logging_steps=25,
        report_to="none",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    
    # Train with patience of 5 instead of 3
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.002
        ),
        SaveBestModelCallback(
            output_dir=output_dir,
            early_stopping_patience=5,
            early_stopping_threshold=0.002
        )
    ]
    
    # Train the model
    results = train_model(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        eval_df=eval_df,
        output_dir=output_dir,
        training_args=training_args,
        use_enhanced_prompt=use_enhanced_prompt,
        auto_batch_size=auto_batch_size  # Pass through the auto_batch_size parameter
    )
    
    # Save model configuration and best hyperparameters
    config_file = os.path.join(output_dir, "model_config.json")
    config = {
        "base_model": base_model_name,
        "hyperparameters": best_params,
        "lora_config": {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": lora_config.target_modules
        },
        "training_args": training_args.to_dict(),
        "results": results
    }
    save_json(config, config_file)
    
    logging.info(f"Final model trained and saved to {output_dir}")
    return results

def analyze_errors(model, tokenizer, test_df, output_dir):
    """
    Analyze errors made by the model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_df: Test dataset
        output_dir: Output directory for results
    
    Returns:
        Dictionary with error analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Analyzing errors on {len(test_df)} test examples")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Import preprocessing function
    from src.preprocessing import create_prompt, preprocess_for_training
    
    # Prepare dataset
    test_dataset = prepare_dataset_for_training(
        test_df, tokenizer, use_enhanced_prompt=False
    )
    
    # Run predictions
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define compute metrics wrapper
    def compute_metrics_wrapper(eval_pred):
        """Wrapper for compute_metrics to match Trainer's expected interface"""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        return compute_metrics(predictions, labels)
    
    trainer = MisinformationTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=8,
            report_to="none"
        ),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper,
        data_collator=data_collator
    )
    
    predictions = trainer.predict(test_dataset)
    
    # Extract predictions and labels
    y_true = predictions.label_ids
    y_probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    y_pred = np.argmax(y_probs, axis=1)
    
    # Overall metrics
    metrics = compute_metrics(predictions.predictions, y_true)
    
    # Analyze errors
    errors = []
    for i, (true, pred, probs) in enumerate(zip(y_true, y_pred, y_probs)):
        if true != pred:
            # Get the original example
            idx = test_df.index[i] if i < len(test_df) else None
            content = test_df.iloc[i]["content"] if idx is not None else "Unknown"
            
            # Get features if available
            features = {
                feature: test_df.iloc[i][feature]
                for feature in IMPORTANT_FEATURES
                if feature in test_df.columns
            } if idx is not None else {}
            
            error_type = "False Positive" if pred == 1 and true == 0 else "False Negative"
            
            errors.append({
                "index": int(idx) if idx is not None else None,
                "content": content,
                "true_label": int(true),
                "predicted_label": int(pred),
                "true_label_text": "TRUE (Factual)" if true == 1 else "FALSE (Misinformation)",
                "predicted_label_text": "TRUE (Factual)" if pred == 1 else "FALSE (Misinformation)",
                "confidence": float(probs[pred]),
                "error_type": error_type,
                "features": features
            })
    
    # Group errors by type
    false_positives = [e for e in errors if e["error_type"] == "False Positive"]
    false_negatives = [e for e in errors if e["error_type"] == "False Negative"]
    
    # Analyze error patterns
    error_analysis = {
        "total_examples": len(test_df),
        "correct_predictions": len(test_df) - len(errors),
        "accuracy": metrics["accuracy"],
        "total_errors": len(errors),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "error_rate": len(errors) / len(test_df),
        "false_positive_rate": metrics["false_positive_rate"],
        "false_negative_rate": metrics["false_negative_rate"],
        "error_examples": errors[:20],  # Limit to 20 examples
        "metrics": metrics
    }
    
    # Save error analysis
    analysis_file = os.path.join(output_dir, "error_analysis.json")
    save_json(error_analysis, analysis_file)
    
    # Create confusion matrix visualization
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    visualize_confusion_matrix(
        y_true, 
        y_pred,
        output_path=cm_path, 
        title="Test Set Confusion Matrix"
    )
    
    # Create feature correlation analysis for errors
    if len(errors) > 0 and all(len(e["features"]) > 0 for e in errors):
        try:
            # Extract features from errors
            error_features = pd.DataFrame([e["features"] for e in errors])
            
            # Calculate correlation with error types
            error_features["error_type"] = [1 if e["error_type"] == "False Positive" else 0 for e in errors]
            
            # Calculate correlations
            correlations = error_features.corr()["error_type"].drop("error_type").sort_values(ascending=False)
            
            # Save correlations
            correlations_file = os.path.join(output_dir, "feature_correlations.json")
            save_json(correlations.to_dict(), correlations_file)
            
            # Plot feature correlations
            plt.figure(figsize=(12, 8))
            correlations.plot(kind="bar")
            plt.title("Feature Correlation with Error Type")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_correlations.png"))
            plt.close()
        except Exception as e:
            logging.warning(f"Could not analyze feature correlations: {e}")
    
    logging.info(f"Error analysis saved to {analysis_file}")
    return error_analysis


# Add to the bottom of your training.py file or as a new function

def train_model_with_deepspeed(
    model,
    tokenizer,
    train_df,
    eval_df=None,
    output_dir=None,
    training_args=None,
    class_weights=None,
    use_enhanced_prompt=None,
    deepspeed_config=None
):
    """
    Train a model using DeepSpeed for distributed and optimized training.
    This is an extension of the train_model function with DeepSpeed integration.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer for encoding
        train_df: Training dataset
        eval_df: Evaluation dataset
        output_dir: Directory to save model to
        training_args: Custom training arguments
        class_weights: Optional weights for loss calculation
        use_enhanced_prompt: Whether to use enhanced prompts
        deepspeed_config: DeepSpeed configuration
        
    Returns:
        Dictionary with training results
    """
    import deepspeed
    from src.config import DEEPSPEED_CONFIG, TRAINING_CONFIG
    
    # Set default output directory
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(MODELS_DIR, f"model_deepspeed_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use config from parameter or fall back to defaults
    ds_config = deepspeed_config or DEEPSPEED_CONFIG
    
    # Auto-detect optimal settings if enabled
    if ds_config.get("auto_detect", True):
        try:
            ds_config = _auto_detect_deepspeed_config(model)
        except Exception as e:
            logging.warning(f"Error in auto-detecting DeepSpeed config: {e}")
            # Continue with defaults
    
    # Create DeepSpeed config file
    ds_config_file = os.path.join(output_dir, "ds_config.json")
    deepspeed_config_dict = _create_deepspeed_config_dict(
        zero_stage=ds_config.get("zero_stage", 2),
        offload_optimizer=ds_config.get("offload_optimizer", False),
        offload_parameters=ds_config.get("offload_parameters", False),
        gradient_accumulation_steps=ds_config.get("gradient_accumulation_steps", 4),
        fp16=ds_config.get("fp16", True)
    )
    
    # Save config to file
    with open(ds_config_file, 'w') as f:
        import json
        json.dump(deepspeed_config_dict, f, indent=2)
    
    # Set up DeepSpeed training arguments
    if training_args is None:
        # Use default configuration with DeepSpeed
        config = TRAINING_CONFIG.copy()
        
        # Update with DeepSpeed-appropriate batch size
        from src.config import determine_optimal_batch_size_with_deepspeed
        config["batch_size"] = determine_optimal_batch_size_with_deepspeed(model)
        
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=config["eval_steps"],
            save_strategy="steps",
            save_steps=config["save_steps"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=ds_config.get("gradient_accumulation_steps", 4),
            num_train_epochs=config["epochs"],
            fp16=ds_config.get("fp16", True),
            gradient_checkpointing=True,
            warmup_ratio=config["warmup_ratio"],
            lr_scheduler_type="linear",
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            logging_dir=os.path.join(LOGS_DIR, "training_logs"),
            logging_steps=25,
            report_to="none",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,  # Recommended for DeepSpeed
            remove_unused_columns=True,
            deepspeed=ds_config_file  # Add DeepSpeed config
        )
    else:
        # Update existing training args with DeepSpeed config
        training_args.deepspeed = ds_config_file
        
    # Determine whether to use enhanced prompts
    if use_enhanced_prompt is None:
        use_enhanced_prompt = TRAINING_CONFIG["use_enhanced_prompt"]
    
    # Prepare datasets
    logging.info("Preparing datasets for training with DeepSpeed")
    
    train_dataset = prepare_dataset_for_training(
        train_df, tokenizer, use_enhanced_prompt=use_enhanced_prompt
    )
    
    if eval_df is not None:
        eval_dataset = prepare_dataset_for_training(
            eval_df, tokenizer, use_enhanced_prompt=use_enhanced_prompt
        )
    else:
        # Use a portion of training data for evaluation if no eval set provided
        dataset_dict = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
    
    # Define compute metrics wrapper
    def compute_metrics_wrapper(eval_pred):
        """Wrapper for compute_metrics to match Trainer's expected interface"""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        return compute_metrics(predictions, labels)
    
    # Add callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=5,  # Increased patience for DeepSpeed
            early_stopping_threshold=0.002
        ),
        SaveBestModelCallback(
            output_dir=output_dir,
            early_stopping_patience=5,
            early_stopping_threshold=0.002
        )
    ]
    
    # Create trainer
    trainer = MisinformationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper,
        callbacks=callbacks,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        class_weights=class_weights
    )
    
    # Start training
    logging.info("Starting DeepSpeed training")
    import time
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Final evaluation
    logging.info("Performing final evaluation")
    eval_results = trainer.evaluate()
    
    # Save model
    logging.info("Saving trained model")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Get efficiency metrics
    efficiency_metrics = trainer.get_efficiency_metrics()
    
    # Combine metrics
    results = {
        "training_time": training_time,
        "training_time_formatted": format_time(training_time),
        "samples_per_second": len(train_dataset) / training_time,
        "eval_results": eval_results,
        "efficiency_metrics": efficiency_metrics,
        "hyperparameters": {
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "epochs": training_args.num_train_epochs,
            "use_enhanced_prompt": use_enhanced_prompt
        },
        "deepspeed_config": {
            "zero_stage": ds_config.get("zero_stage", 2),
            "offload_optimizer": ds_config.get("offload_optimizer", False),
            "offload_parameters": ds_config.get("offload_parameters", False)
        }
    }
    
    # Save results
    results_file = os.path.join(output_dir, "training_results.json")
    save_json(results, results_file)
    
    # Generate metrics visualization
    try:
        # Run predictions on evaluation dataset
        predictions = trainer.predict(eval_dataset)
        y_true = predictions.label_ids
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Create confusion matrix visualization
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        visualize_confusion_matrix(
            y_true, 
            y_pred,
            output_path=cm_path, 
            title="Confusion Matrix for Misinformation Detection"
        )
        
        # Add visualization paths to results
        results["visualizations"] = {
            "confusion_matrix": cm_path
        }
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
    
    logging.info(f"DeepSpeed training completed. Results saved to {results_file}")
    return results

def _auto_detect_deepspeed_config(model):
    """
    Auto-detect the best DeepSpeed configuration based on model size and hardware.
    
    Args:
        model: The model to analyze
        
    Returns:
        DeepSpeed configuration dictionary
    """
    import torch
    
    # Determine model size
    model_size_category = "medium"  # Default
    model_name = getattr(model, "name_or_path", None)
    if model_name is None and hasattr(model, "config"):
        model_name = getattr(model.config, "name_or_path", "unknown")
    
    if "70B" in model_name:
        model_size_category = "xlarge"
    elif "13B" in model_name or "20B" in model_name or "30B" in model_name:
        model_size_category = "large"
    elif "7B" in model_name or "8B" in model_name:
        model_size_category = "medium"
    elif "3B" in model_name or "1.5B" in model_name:
        model_size_category = "small"
    
    # Get GPU info
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    available_memory = None
    
    if torch.cuda.is_available():
        try:
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        except:
            available_memory = 16  # Assume 16GB
    
    # Configure based on model size and hardware
    config = {
        "zero_stage": 2,
        "offload_optimizer": False,
        "offload_parameters": False,
        "gradient_accumulation_steps": 4,
        "fp16": True
    }
    
    # Single GPU optimizations
    if available_gpus <= 1:
        if model_size_category == "xlarge":
            # 70B models need the most optimization
            config.update({
                "zero_stage": 3,
                "offload_optimizer": True,
                "offload_parameters": True,
                "gradient_accumulation_steps": 16
            })
        elif model_size_category == "large":
            # 13B models need significant optimization
            config.update({
                "zero_stage": 3,
                "offload_optimizer": True,
                "gradient_accumulation_steps": 8
            })
    # Multi-GPU optimizations
    else:
        if model_size_category == "xlarge":
            # 70B models with multiple GPUs
            config.update({
                "zero_stage": 3,
                "gradient_accumulation_steps": max(1, 8 // available_gpus)
            })
        elif model_size_category == "large":
            # 13B models with multiple GPUs
            config.update({
                "zero_stage": 2,
                "gradient_accumulation_steps": max(1, 4 // available_gpus)
            })
    
    return config

def _create_deepspeed_config_dict(
    zero_stage=2,
    offload_optimizer=False,
    offload_parameters=False,
    gradient_accumulation_steps=1,
    fp16=True
):
    """
    Create a DeepSpeed configuration dictionary.
    
    Args:
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
        offload_optimizer: Whether to offload optimizer states to CPU
        offload_parameters: Whether to offload parameters to CPU (ZeRO-3 only)
        gradient_accumulation_steps: Number of gradient accumulation steps
        fp16: Whether to use mixed precision training
        
    Returns:
        DeepSpeed configuration dictionary
    """
    # Base DeepSpeed configuration
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto"
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }
    
    # Add FP16 mixed precision settings
    if fp16:
        config["fp16"] = {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    
    # Configure ZeRO optimization
    zero_config = {"stage": zero_stage}
    
    # Stage 3 specific configurations
    if zero_stage == 3:
        zero_config.update({
            "offload_optimizer": {
                "device": "cpu" if offload_optimizer else "none"
            },
            "offload_param": {
                "device": "cpu" if offload_parameters else "none"
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        })
    
    # Stage 1 & 2 offloading configs
    elif offload_optimizer and (zero_stage == 1 or zero_stage == 2):
        zero_config.update({
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
        })
    
    config["zero_optimization"] = zero_config
    
    return config