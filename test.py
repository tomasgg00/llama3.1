from src.model import load_base_model
import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

def check_quantization():
    # Load model with quantization
    model = load_base_model(
        model_name="meta-llama/Llama-3.1-8B-Instruct", 
        quantize=True
    )

    print("Quantization Diagnostic:")
    
    # Check if model was loaded with quantization config
    if hasattr(model, 'config'):
        print("\nModel Configuration:")
        print(f"Is 4-bit quantization configured: {hasattr(model.config, 'quantization_config')}")
        
        if hasattr(model.config, 'quantization_config'):
            quantization_config = model.config.quantization_config
            print("\nQuantization Details:")
            print(f"Load in 4-bit: {quantization_config.load_in_4bit}")
            print(f"Quantization Type: {quantization_config.bnb_4bit_quant_type}")
            print(f"Use Double Quantization: {quantization_config.bnb_4bit_use_double_quant}")
    
    # Alternative method to verify quantization
    try:
        from peft import prepare_model_for_kbit_training
        prepared_model = prepare_model_for_kbit_training(model)
        print("\nModel prepared for k-bit training successfully.")
    except Exception as e:
        print(f"\nError preparing for k-bit training: {e}")

check_quantization()