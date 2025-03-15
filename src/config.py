""""config.py: Configuration for the Misinformation Detection Pipeline"""
"""
Configuration settings for the misinformation detection pipeline.
"""
import os
from pathlib import Path
import torch

def determine_optimal_batch_size(model=None, available_memory=None):
    """
    Dynamically determine optimal batch size based on model and available memory.
    
    Args:
        model: Model to be used (optional)
        available_memory: Available GPU memory in GB (optional)
    
    Returns:
        Optimal batch size
    """
    # Default base batch size
    base_batch_size = 4
    
    # Check available GPU memory if not provided
    if available_memory is None:
        try:
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            else:
                available_memory = 8  # Assume modest CPU memory
        except:
            available_memory = 8
    
    # Adjust based on model size if model is provided
    if model is not None:
        model_size_str = str(model)
        if "70B" in model_size_str:
            base_batch_size = 2  # Larger models need smaller batches
    
    # Dynamic adjustment based on available memory
    if available_memory > 24:  # GB
        return base_batch_size * 2
    elif available_memory > 16:
        return base_batch_size
    else:
        return max(base_batch_size // 2, 1)

# Base paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "processed_data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
SYNTHETIC_DATA_DIR = os.path.join(ROOT_DIR, "synthetic_data")

# Create directories if they don't exist
for dir_path in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, SYNTHETIC_DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)



# Model configuration
MODEL_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",  # Base model for fine-tuning
    "alt_models": {
        "small": "meta-llama/Llama-3.1-8B-Instruct",
        "medium": "meta-llama/Llama-3.1-8B-Instruct",
        "large": "meta-llama/Llama-3.1-70B-Instruct"
    }
}

# Feature sets for engineering
BASIC_FEATURES = [
    "word_count", "char_count", "sentence_count", "avg_word_length",
    "type_token_ratio", "readability_score"
]

NLP_FEATURES = [
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio"
]

SENTIMENT_FEATURES = [
    "sentiment_polarity", "sentiment_positive", "sentiment_negative", "sentiment_neutral"
]

OTHER_FEATURES = [
    "num_named_entities"
]

# Combined feature sets
ALL_FEATURES = BASIC_FEATURES + NLP_FEATURES + SENTIMENT_FEATURES + OTHER_FEATURES

# Important features subset (based on feature importance analysis)
IMPORTANT_FEATURES = [
    "sentiment_polarity", "char_count", "avg_word_length", "verb_ratio", 
    "word_count", "type_token_ratio", "readability_score"
]

# LoRA configuration
LORA_CONFIG = {
    "r": 32,  # Default rank
    "lora_alpha": 64,  # Default scaling factor
    "lora_dropout": 0.1,  # Default dropout rate
    "bias": "none",
    "task_type": "SEQ_CLS",
    # Default target modules for Llama models
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": determine_optimal_batch_size(),  # Move this line after function definition
    "learning_rate": 2e-5,
    "epochs": 5,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "evaluation_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "use_enhanced_prompt": True
}

# Hyperparameter optimization search space
OPTUNA_CONFIG = {
    "n_trials": 10,  # Default number of trials
    "search_space": {
        "learning_rate": (1e-5, 5e-5),
        "batch_size": [2, 4, 8],
        "lora_r": [8, 16, 32, 64],
        "lora_alpha": [16, 32, 64, 128],
        "lora_dropout": (0.05, 0.2),
        "use_enhanced_prompt": [True, False],
        "gradient_accumulation_steps": [2, 4, 8]
    }
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "batch_size": 1000,  # Batch size for processing
    "max_text_length": 5000,  # Maximum text length to process
    "max_token_length": 512,  # Maximum token length for model input
    "use_enhanced_prompts": True,  # Whether to use enhanced prompts with features
    "truncation": True,  # Whether to truncate text to max_token_length
    "padding": "max_length"  # Padding strategy
}

# DeepSpeed configuration
DEEPSPEED_CONFIG = {
    "enabled": False,  # Whether to use DeepSpeed by default
    "zero_stage": 2,   # ZeRO optimization stage (0, 1, 2, or 3)
    "offload_optimizer": False,  # Whether to offload optimizer states to CPU
    "offload_parameters": False,  # Whether to offload parameters to CPU (ZeRO-3 only)
    "gradient_accumulation_steps": 4,  # Number of gradient accumulation steps
    "fp16": True,      # Whether to use mixed precision training
    "auto_detect": True  # Whether to auto-detect optimal configuration based on model and hardware
}

# Function to determine optimal DeepSpeed settings based on model and hardware
def determine_optimal_batch_size_with_deepspeed(model=None, available_memory=None):
    """
    Determine optimal batch size when using DeepSpeed.
    Modified version that accounts for DeepSpeed optimizations.
    
    Args:
        model: Model to be used (optional)
        available_memory: Available GPU memory in GB (optional)
    
    Returns:
        Optimal batch size with DeepSpeed optimizations
    """
    # Get base batch size from standard function
    base_batch_size = determine_optimal_batch_size(model, available_memory)
    
    # With DeepSpeed ZeRO-2 we can typically use 1.5-2x larger batches
    # With ZeRO-3 and offloading, even more
    if DEEPSPEED_CONFIG["zero_stage"] == 3 and DEEPSPEED_CONFIG["offload_parameters"]:
        # ZeRO-3 with parameter offloading gives the most memory savings
        return min(base_batch_size * 3, 16)  # Cap at 16 to avoid other issues
    elif DEEPSPEED_CONFIG["zero_stage"] == 3:
        # ZeRO-3 without parameter offloading
        return min(base_batch_size * 2.5, 16)
    elif DEEPSPEED_CONFIG["zero_stage"] == 2:
        # ZeRO-2 gives good memory savings
        return min(base_batch_size * 2, 16)
    else:
        # ZeRO-1 gives modest memory savings
        return min(base_batch_size * 1.5, 16)