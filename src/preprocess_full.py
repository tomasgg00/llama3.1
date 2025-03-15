import os
import sys
import pandas as pd
from pathlib import Path

# Add the current directory to the path so imports work correctly
sys.path.append('.')

# Now import from the src module
from src.preprocessing import clean_text, extract_all_features
from src.utils import setup_logging

# Setup logging
logger = setup_logging()

# Load dataset
dataset_name = "unhcr"  # Your dataset name
file_path = f"data/{dataset_name}.csv"  # Adjust if using a different format

logger.info(f"Loading dataset from: {file_path}")

# Read data
df = pd.read_csv(file_path)
logger.info(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")

# Ensure necessary columns exist
if 'content' not in df.columns:
    # Try to find a suitable text column
    text_columns = [col for col in df.columns if col in ['text', 'tweet', 'post', 'message']]
    if text_columns:
        df.rename(columns={text_columns[0]: 'content'}, inplace=True)
        logger.info(f"Renamed column '{text_columns[0]}' to 'content'")
    else:
        logger.error("No text column found in dataset")
        raise ValueError("No text column found in dataset")
        
if 'label' not in df.columns:
    # Try to find a suitable label column
    label_columns = [col for col in df.columns if col in ['class', 'verdict', 'misinfo']]
    if label_columns:
        df.rename(columns={label_columns[0]: 'label'}, inplace=True)
        logger.info(f"Renamed column '{label_columns[0]}' to 'label'")
    else:
        logger.error("No label column found in dataset")
        raise ValueError("No label column found in dataset")

# Process texts
logger.info(f"Processing {len(df)} texts")
processed_rows = []
for idx, row in df.iterrows():
    # Clean text
    content = row['content']
    cleaned_text = clean_text(content)
    
    # Extract features
    features = extract_all_features(cleaned_text)
    
    # Add label
    label = row['label']
    features['label'] = 1 if label in [True, 1, "1", "TRUE", "True", "true"] else 0
    
    # Create processed row
    processed_row = {
        "content": cleaned_text,
        "source": "external_validation",
        **features
    }
    processed_rows.append(processed_row)

# Create processed dataframe
processed_df = pd.DataFrame(processed_rows)

# Save processed data
output_dir = Path("processed_data")
output_dir.mkdir(exist_ok=True)

# Save full dataset (without splitting)
output_path = output_dir / f"{dataset_name}_full.json"
processed_df.to_json(output_path, orient="records")

logger.info(f"Processed {len(processed_df)} examples")
logger.info(f"Saved to {output_path}")