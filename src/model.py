"""
model.py: Model definition module for the misinformation detection pipeline.


This module handles:
- Model loading and configuration
- LoRA setup for parameter-efficient fine-tuning
- 4-bit quantization configuration
- Model class definitions
"""
import os
import torch
import logging
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

from src.config import MODEL_CONFIG, LORA_CONFIG, MODELS_DIR

def get_tokenizer(model_name=None, use_fast=True):
    """
    Load the tokenizer for the specified model.
    
    Args:
        model_name: Name or path of the model
        use_fast: Whether to use the fast tokenizer implementation
    
    Returns:
        Loaded tokenizer
    """
    model_name = model_name or MODEL_CONFIG["base_model"]
    
    try:
        logging.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            trust_remote_code=True
        )
        
        # Set padding token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logging.info(f"Set pad_token to eos_token ({tokenizer.pad_token})")
        
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        return None

def get_quantization_config(bits=4, compute_dtype="float16"):
    """
    Create quantization configuration for efficient inference.
    
    Args:
        bits: Number of bits for quantization (4 or 8)
        compute_dtype: Data type for computation
    
    Returns:
        BitsAndBytesConfig object
    """
    compute_dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    
    compute_dtype_torch = compute_dtype_map.get(compute_dtype, torch.float16)
    
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype_torch,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=compute_dtype_torch
        )
    elif bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=compute_dtype_torch
        )
    else:
        logging.warning(f"Unsupported quantization bits: {bits}. Using 4-bit.")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype_torch,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=compute_dtype_torch
        )

def get_lora_config(
    r=None,
    lora_alpha=None,
    lora_dropout=None,
    target_modules=None,
    bias="none",
    task_type="SEQ_CLS"
):
    """
    Create LoRA configuration for parameter-efficient fine-tuning.
    
    Args:
        r: Rank of LoRA matrices
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout rate for LoRA
        target_modules: List of modules to apply LoRA to
        bias: Bias type ('none', 'all', 'lora_only')
        task_type: Task type for LoRA
    
    Returns:
        LoraConfig object
    """
    config = LORA_CONFIG.copy()
    
    # Override defaults with provided values
    if r is not None:
        config["r"] = r
    if lora_alpha is not None:
        config["lora_alpha"] = lora_alpha
    if lora_dropout is not None:
        config["lora_dropout"] = lora_dropout
    if target_modules is not None:
        config["target_modules"] = target_modules
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias=bias,
        task_type=task_type
    )
    
    return lora_config

def load_base_model(
    model_name=None,
    num_labels=2,
    device_map="auto",
    quantize=True,
    torch_dtype=torch.float16
):
    """
    Load the base model for fine-tuning.
    
    Args:
        model_name: Name or path of the model
        num_labels: Number of classification labels
        device_map: Device mapping strategy
        quantize: Whether to apply quantization
        torch_dtype: Data type for model parameters
    
    Returns:
        Loaded model
    """
    model_name = model_name or MODEL_CONFIG["base_model"]
    
    try:
        # Configure quantization if requested
        quantization_config = get_quantization_config() if quantize else None
        
        # Determine device map based on available hardware
        if device_map == "auto" and not torch.cuda.is_available():
            device_map = None
            logging.warning("CUDA not available, using CPU for model loading")
        
        logging.info(f"Loading base model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # Get tokenizer to set pad_token_id
        tokenizer = get_tokenizer(model_name)
        if tokenizer is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        logging.info(f"Successfully loaded model with {model.num_parameters():,} parameters")
        return model
    except Exception as e:
        logging.error(f"Error loading base model: {e}")
        return None

def prepare_for_lora(model, use_gradient_checkpointing=True):
    """
    Prepare model for LoRA fine-tuning.
    
    Args:
        model: Base model to prepare
        use_gradient_checkpointing: Whether to use gradient checkpointing
    
    Returns:
        Model prepared for LoRA fine-tuning
    """
    try:
        if model is None:
            logging.error("Cannot prepare None model for LoRA")
            return None
        
        logging.info("Preparing model for LoRA fine-tuning")
        prepared_model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        logging.info("Model successfully prepared for LoRA fine-tuning")
        return prepared_model
    except Exception as e:
        logging.error(f"Error preparing model for LoRA: {e}")
        return model  # Return original model if preparation fails

def apply_lora(model, lora_config=None):
    """
    Apply LoRA adapter to the model.
    
    Args:
        model: Model to apply LoRA to
        lora_config: LoRA configuration
    
    Returns:
        Model with LoRA applied
    """
    try:
        if model is None:
            logging.error("Cannot apply LoRA to None model")
            return None
        
        # Use default config if none provided
        if lora_config is None:
            lora_config = get_lora_config()
        
        logging.info(f"Applying LoRA with rank={lora_config.r}, alpha={lora_config.lora_alpha}")
        lora_model = get_peft_model(model, lora_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        logging.info(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}% of total)")
        return lora_model
    except Exception as e:
        logging.error(f"Error applying LoRA: {e}")
        return model  # Return original model if LoRA application fails

def save_model(model, tokenizer, output_dir, model_name=None):
    """
    Save model and tokenizer.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
        model_name: Optional subfolder name
    
    Returns:
        Path to saved model
    """
    # Create output directory
    if model_name:
        save_dir = Path(output_dir) / model_name
    else:
        timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(output_dir) / f"model_{timestamp}"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save model and tokenizer
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        logging.info(f"Model and tokenizer saved to {save_dir}")
        return save_dir
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        return None

def load_model_for_inference(model_path, device=None, adapter_name=None):
    """
    Load a model for inference, including LoRA adapter if applicable.
    
    Args:
        model_path: Path to the model or adapter
        device: Device to load the model on
        adapter_name: Optional adapter name for LoRA models
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if model path exists
    if not os.path.exists(model_path):
        # Try looking in models directory
        alt_path = os.path.join(MODELS_DIR, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            logging.error(f"Model path not found: {model_path}")
            return None, None
    
    # Determine if this is a full model or an adapter
    is_adapter = (
        os.path.exists(os.path.join(model_path, "adapter_config.json")) or
        os.path.exists(os.path.join(model_path, "adapter_model.bin"))
    )
    
    # Load tokenizer first
    tokenizer = None
    base_model_name = MODEL_CONFIG["base_model"]
    
    # Check if tokenizer exists in the model path
    if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logging.info(f"Loaded tokenizer from {model_path}")
        except Exception as e:
            logging.error(f"Error loading tokenizer from model path: {e}")
    
    # Fall back to base model tokenizer if needed
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            logging.info(f"Loaded tokenizer from base model: {base_model_name}")
        except Exception as e:
            logging.error(f"Error loading tokenizer from base model: {e}")
            return None, None
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model based on whether it's an adapter or full model
    try:
        if is_adapter:
            logging.info(f"Loading base model for adapter: {base_model_name}")
            
            # Load the base model with 4-bit quantization
            quantization_config = get_quantization_config()
            
            # Configure device
            if device is None:
                device_map = "auto" if torch.cuda.is_available() else None
            else:
                device_map = device
            
            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=2,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.float16
            )
            
            # Ensure pad token is set
            base_model.config.pad_token_id = tokenizer.pad_token_id
            
            # Load the adapter
            logging.info(f"Loading adapter from {model_path}")
            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                adapter_name=adapter_name
            )
        else:
            # Load full model
            logging.info(f"Loading full model from {model_path}")
            
            # Configure device
            if device is None:
                device_map = "auto" if torch.cuda.is_available() else None
            else:
                device_map = device
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                device_map=device_map
            )
        
        # Set model to evaluation mode
        model.eval()
        logging.info(f"Successfully loaded model for inference")
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model for inference: {e}")
        return None, None

class MisinformationDetectionModel:
    """Model class for misinformation detection."""
    
    def __init__(
        self, 
        model_name=None, 
        lora_config=None, 
        quantize=True,
        device=None
    ):
        """
        Initialize the model.
        
        Args:
            model_name: Name or path of the model
            lora_config: LoRA configuration
            quantize: Whether to apply quantization
            device: Device to load the model on
        """
        self.model_name = model_name or MODEL_CONFIG["base_model"]
        self.lora_config = lora_config or get_lora_config()
        self.quantize = quantize
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.is_lora = False
    
    def load(self):
        """Load the model and tokenizer."""
        # Load tokenizer
        self.tokenizer = get_tokenizer(self.model_name)
        if self.tokenizer is None:
            logging.error("Failed to load tokenizer")
            return False
        
        # Load base model
        base_model = load_base_model(
            model_name=self.model_name,
            quantize=self.quantize,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if base_model is None:
            logging.error("Failed to load base model")
            return False
        
        # Prepare model for LoRA and apply LoRA
        prepared_model = prepare_for_lora(base_model)
        self.model = apply_lora(prepared_model, self.lora_config)
        self.is_lora = True
        
        logging.info("Model and tokenizer loaded successfully")
        return True
    
    def load_for_inference(self, model_path):
        """
        Load a saved model for inference.
        
        Args:
            model_path: Path to the saved model
        
        Returns:
            True if successful, False otherwise
        """
        model, tokenizer = load_model_for_inference(model_path, self.device)
        
        if model is None or tokenizer is None:
            logging.error("Failed to load model for inference")
            return False
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Determine if this is a LoRA model
        self.is_lora = hasattr(model, "base_model") and hasattr(model, "peft_config")
        
        logging.info(f"Loaded model for inference from {model_path}")
        return True
    
    def predict(self, text, features=None, use_enhanced_prompt=False):
        """
        Make a prediction for a single text.
        
        Args:
            text: Text to classify
            features: Optional features for enhanced prompting
            use_enhanced_prompt: Whether to use enhanced prompts
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.tokenizer is None:
            logging.error("Model or tokenizer not loaded")
            return None
        
        # Create prompt
        from src.preprocessing import create_prompt
        prompt = create_prompt(text, features, use_enhanced_prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
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
            "confidence": confidence,
            "class_probabilities": {
                "FALSE (Misinformation)": float(probabilities[0]),
                "TRUE (Factual)": float(probabilities[1])
            }
        }
        
        return result
    
    def save(self, output_dir, model_name=None):
        """
        Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save to
            model_name: Optional model name
        
        Returns:
            Path to saved model
        """
        if self.model is None or self.tokenizer is None:
            logging.error("Model or tokenizer not loaded")
            return None
        
        return save_model(self.model, self.tokenizer, output_dir, model_name)