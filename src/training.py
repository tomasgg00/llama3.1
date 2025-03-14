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
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with optional class weighting.
        
        Args:
            model: Model to compute loss for
            inputs: Inputs to the model
            return_outputs: Whether to return model outputs
        
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
            
            # Save the best model
            best_dir = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            
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
    use_enhanced_prompt=None
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
    
    Returns:
        Dictionary with training results
    """
    # Set default output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(MODELS_DIR, f"model_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Set up training arguments
    if training_args is None:
        # Use default configuration
        config = TRAINING_CONFIG.copy()
        
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
        compute_metrics=compute_metrics,
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
    # Create trial directory
    trial_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_trial_{trial.number}"
    trial_dir = os.path.join(output_dir, trial_id)
    os.makedirs(trial_dir, exist_ok=True)
    
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
    
    try:
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
            use_enhanced_prompt=use_enhanced_prompt
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
    output_dir=None
):
    """
    Train the final model with the best hyperparameters.
    
    Args:
        train_df: Training dataset
        eval_df: Evaluation dataset
        base_model_name: Base model name
        best_params: Best hyperparameters
        output_dir: Output directory
    
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
        logging_dir=os.path.join(output_dir, "logs"),
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
        use_enhanced_prompt=use_enhanced_prompt
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
    
    trainer = MisinformationTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=8,
            report_to="none"
        ),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
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