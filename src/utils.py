"""utils.py: Utility functions for the Misinformation Detection Pipeline"""
"""
Utility functions for the misinformation detection pipeline.
"""
import os
import json
import time
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

# Configure logging
def setup_logging(log_file=None):
    """Set up logging configuration with file and console handlers."""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Generate default log filename if none provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"pipeline_{timestamp}.log"
    
    log_path = log_dir / log_file
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_path}")
    return logging.getLogger()

# Progress tracking
class ProgressTracker:
    """Track progress of multi-step processes with time estimation."""
    def __init__(self, total_steps, logger=None):
        self.total_steps = total_steps
        self.completed_steps = 0
        self.start_time = time.time()
        self.step_times = {}
        self.logger = logger or logging.getLogger()
        
    def start_step(self, step_name):
        """Start timing a new step."""
        self.current_step = step_name
        self.step_start_time = time.time()
        self.logger.info(f"[{self.completed_steps+1}/{self.total_steps}] Starting: {step_name}")
        
    def complete_step(self):
        """Complete the current step and update progress."""
        elapsed = time.time() - self.step_start_time
        self.step_times[self.current_step] = elapsed
        self.completed_steps += 1
        
        # Calculate overall progress
        overall_progress = (self.completed_steps / self.total_steps) * 100
        
        self.logger.info(f"[{self.completed_steps}/{self.total_steps}] Completed: {self.current_step} in {elapsed:.2f}s")
        self.logger.info(f"Overall progress: {overall_progress:.1f}% complete")
        
        # Estimate remaining time if we have at least one step completed
        if self.completed_steps > 0:
            avg_step_time = sum(self.step_times.values()) / len(self.step_times)
            remaining_steps = self.total_steps - self.completed_steps
            est_remaining_time = avg_step_time * remaining_steps
            
            # Format estimated remaining time
            time_str = format_time(est_remaining_time)
            self.logger.info(f"Estimated remaining time: {time_str}")
    
    def get_summary(self):
        """Get summary of all completed steps and timing."""
        total_time = time.time() - self.start_time
        return {
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "total_time": total_time,
            "total_time_formatted": format_time(total_time),
            "step_times": self.step_times,
            "average_step_time": sum(self.step_times.values()) / max(1, len(self.step_times))
        }

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"

def save_json(data, filepath, indent=2):
    """Save data to a JSON file with proper handling."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Handle non-serializable types
    def json_serializer(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)  # Default for anything else
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=json_serializer)

def load_json(filepath):
    """Load data from a JSON file with proper handling."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def clean_up_memory():
    """Clean up memory to avoid CUDA out of memory errors."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def visualize_confusion_matrix(y_true, y_pred, labels=None, output_path=None, title="Confusion Matrix"):
    """Create and save a confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels or ['FALSE (Misinfo)', 'TRUE (Factual)'],
        yticklabels=labels or ['FALSE (Misinfo)', 'TRUE (Factual)']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        return plt

def visualize_roc_curve(y_true, y_score, output_path=None, title="ROC Curve"):
    """Create and save a ROC curve visualization."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        return plt

def visualize_precision_recall_curve(y_true, y_score, output_path=None, title="Precision-Recall Curve"):
    """Create and save a precision-recall curve visualization."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        return plt

def compute_metrics(predictions, labels):
    """Compute comprehensive evaluation metrics for classification."""
    preds = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else predictions
    
    # Calculate base metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=1)
    
    # Calculate class-specific metrics
    precision_class, recall_class, f1_class, support_class = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=1
    )
    
    # Calculate confusion matrix for analysis
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional diagnostic metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Calculate AUC if possible (requires probability scores)
    auc_score = 0
    if len(predictions.shape) > 1:
        try:
            # Check if we have probability scores
            proba = predictions[:, 1] if predictions.shape[1] > 1 else predictions
            auc_score = roc_auc_score(labels, proba)
        except:
            pass
    
    return {
        # Overall metrics
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        
        # Class-specific metrics
        "precision_misinfo": precision_class[0],  # For class 0 (FALSE/misinformation)
        "recall_misinfo": recall_class[0],
        "f1_misinfo": f1_class[0],
        "support_misinfo": support_class[0],
        
        "precision_factual": precision_class[1],  # For class 1 (TRUE/factual)
        "recall_factual": recall_class[1],
        "f1_factual": f1_class[1],
        "support_factual": support_class[1],
        
        # Additional metrics
        "specificity": specificity,
        "npv": npv,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "auc": auc_score
    }

def get_available_device():
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"Using GPU: {device_name} with {memory_gb:.2f} GB memory")
    else:
        device = torch.device("cpu")
        logging.info("No GPU available, using CPU")
    
    return device

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that works with tqdm progress bars."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            from tqdm import tqdm
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)