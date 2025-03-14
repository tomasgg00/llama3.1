"""
preprocessing.py: Preprocessing module for the misinformation detection pipeline.

Preprocessing module for the misinformation detection pipeline.

This module handles:
- Data loading and standardization
- Text cleaning and normalization
- Feature extraction
- Dataset balancing
- Data serialization
"""
import os
import json
import re
import time
import logging
import pandas as pd
import numpy as np
import torch
import unidecode
import contractions
from tqdm import tqdm
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from collections import Counter
from sklearn.model_selection import train_test_split

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, some NLP features will be limited")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, sentiment analysis will be approximated")

from src.utils import ProgressTracker, save_json, load_json, format_time
from src.config import (
    PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, DATA_DIR,
    BASIC_FEATURES, NLP_FEATURES, SENTIMENT_FEATURES, 
    ALL_FEATURES, IMPORTANT_FEATURES, PREPROCESSING_CONFIG
)

# Initialize global variables
nlp = None
sia = None

def initialize_nlp_models():
    """Initialize NLP models once to avoid repeated loading."""
    global nlp, sia
    
    if SPACY_AVAILABLE and nlp is None:
        try:
            # Use the small model for faster processing
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
            logging.info("Loaded spaCy model for NLP processing")
        except OSError:
            logging.error("Failed to load spaCy model, attempting to download")
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
                logging.info("Downloaded and loaded spaCy model")
            except Exception as e:
                logging.error(f"Could not download spaCy model: {e}")
    
    if NLTK_AVAILABLE and sia is None:
        try:
            sia = SentimentIntensityAnalyzer()
            logging.info("Loaded NLTK sentiment analyzer")
        except Exception as e:
            logging.error(f"Error initializing sentiment analyzer: {e}")

def clean_text(text):
    """Clean and preprocess text efficiently."""
    if not isinstance(text, str):
        return ""
    
    # Apply basic cleaning operations
    text = text.lower()
    
    # Skip unidecode for non-ASCII text if not necessary
    if any(ord(char) > 127 for char in text):
        text = unidecode.unidecode(text)
    
    # Only fix contractions if they exist
    if "'" in text:
        text = contractions.fix(text)
    
    # Combine regex operations to reduce passes
    text = re.sub(
        r"^rt\s+|http\S+|www\S+|#\w+|@\w+|\b\d+\b", 
        lambda m: {
            "http": "[URL]",
            "www": "[URL]",
            "#": "[HASHTAG]",
            "@": "[MENTION]"
        }.get(m.group(0)[:1] if m.group(0) else "", ""), 
        text
    )
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_basic_features(text):
    """Extract basic features without heavy NLP processing."""
    features = {}
    
    # Calculate basic text stats
    features["char_count"] = len(text)
    
    # Split text into words and sentences more efficiently
    words = text.split()
    features["word_count"] = len(words)
    
    # Approximate sentence count
    sentences = re.split(r'[.!?]+', text)
    features["sentence_count"] = max(1, len([s for s in sentences if s.strip()]))
    
    # Calculate basic ratios
    features["avg_word_length"] = round(features["char_count"] / max(1, features["word_count"]), 3)
    features["type_token_ratio"] = round(len(set(words)) / max(1, len(words)), 3)
    
    # Simple readability based on average sentence length
    features["readability_score"] = features["word_count"] / max(1, features["sentence_count"])
    
    return features

def extract_nlp_features(text, doc=None):
    """Extract NLP-based features using spaCy."""
    global nlp
    
    features = {}
    
    if nlp is None:
        initialize_nlp_models()
    
    if doc is None and nlp is not None:
        # Only process with spaCy if the text is not too long
        if len(text) > PREPROCESSING_CONFIG["max_text_length"]:
            # Truncate for very long texts
            doc = nlp(text[:PREPROCESSING_CONFIG["max_text_length"]])
        else:
            doc = nlp(text)
    
    if doc is None:
        # If spaCy still didn't initialize correctly, return empty features
        return features
    
    # POS tag counting
    pos_counts = Counter([token.pos_ for token in doc])
    total_tokens = max(1, len(doc))
    
    features["noun_ratio"] = round(pos_counts.get("NOUN", 0) / total_tokens, 3)
    features["verb_ratio"] = round(pos_counts.get("VERB", 0) / total_tokens, 3)
    features["adj_ratio"] = round(pos_counts.get("ADJ", 0) / total_tokens, 3)
    features["adv_ratio"] = round(pos_counts.get("ADV", 0) / total_tokens, 3)
    
    # Named entities (only if NER is enabled)
    if hasattr(doc, "ents") and len(doc.ents) > 0:
        features["num_named_entities"] = len([ent for ent in doc.ents])
    else:
        features["num_named_entities"] = 0
    
    return features

def extract_sentiment(text):
    """Extract sentiment features using NLTK."""
    global sia
    
    if sia is None:
        initialize_nlp_models()
    
    features = {}
    
    # Extract sentiment
    if sia:
        sentiment_scores = sia.polarity_scores(text)
        features["sentiment_polarity"] = round(sentiment_scores["compound"], 3)
        features["sentiment_positive"] = round(sentiment_scores["pos"], 3)
        features["sentiment_negative"] = round(sentiment_scores["neg"], 3)
        features["sentiment_neutral"] = round(sentiment_scores["neu"], 3)
    else:
        # Fallback simple sentiment approximation
        positive_words = ["good", "great", "excellent", "amazing", "happy", "positive"]
        negative_words = ["bad", "poor", "terrible", "awful", "sad", "negative"]
        
        words = set(text.lower().split())
        pos_count = sum(1 for word in positive_words if word in words)
        neg_count = sum(1 for word in negative_words if word in words)
        
        features["sentiment_positive"] = round(pos_count / max(1, len(words)), 3)
        features["sentiment_negative"] = round(neg_count / max(1, len(words)), 3)
        features["sentiment_neutral"] = round(1 - features["sentiment_positive"] - features["sentiment_negative"], 3)
        features["sentiment_polarity"] = round(features["sentiment_positive"] - features["sentiment_negative"], 3)
    
    return features

def extract_all_features(text):
    """Combine all feature extraction methods efficiently."""
    if not text:
        return {}
    
    # Get basic features
    features = extract_basic_features(text)
    
    # Process with spaCy only once
    global nlp
    if nlp is None:
        initialize_nlp_models()
    
    # Only process with spaCy if it's available
    if SPACY_AVAILABLE and nlp is not None:
        # Only process with spaCy if the text is not too long
        if len(text) > PREPROCESSING_CONFIG["max_text_length"]:
            # Truncate for very long texts
            doc = nlp(text[:PREPROCESSING_CONFIG["max_text_length"]])
        else:
            doc = nlp(text)
        
        # Add NLP features
        nlp_features = extract_nlp_features(text, doc)
        features.update(nlp_features)
    
    # Add sentiment features
    sentiment_features = extract_sentiment(text)
    features.update(sentiment_features)
    
    return features

def process_text(row, enable_full_nlp=True):
    """Process text with optimized feature extraction."""
    # Clean text - using standardized column name 'content'
    content = row.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return None
    
    cleaned_text = clean_text(content)
    if not cleaned_text:
        return None
    
    # Extract features based on configuration
    if enable_full_nlp:
        features = extract_all_features(cleaned_text)
    else:
        # Use only basic features for speed
        features = extract_basic_features(cleaned_text)
    
    # Add label - using standardized column name 'label'
    label = row.get("label", 0)
    features["label"] = 1 if label in [True, 1, "1", "TRUE", "True", "true"] else 0
    
    # Return complete row with standardized column names
    return {
        "source": row.get("source", "unknown"),
        "media": row.get("media", "unknown"),
        "content": cleaned_text,
        **features
    }

def process_batch(batch, enable_full_nlp=True):
    """Process a batch of rows."""
    return [process_text(row, enable_full_nlp) for row in batch]

def load_synthetic_samples(minority_class, class_imbalance):
    """Load synthetic samples for dataset balancing."""
    if not os.path.exists(SYNTHETIC_DATA_DIR):
        os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
        logging.warning(f"Synthetic data directory created: {SYNTHETIC_DATA_DIR}")
        return []
    
    # Check for available synthetic data files
    cache_path = os.path.join(SYNTHETIC_DATA_DIR, f"label_{minority_class}_samples.json")
    
    if not os.path.exists(cache_path):
        # Try alternative naming patterns
        alt_patterns = [
            f"label_{minority_class}_*_samples.json",
            f"synthetic_label_{minority_class}.json",
            f"synthetic_class_{minority_class}.json"
        ]
        
        found_file = None
        for pattern in alt_patterns:
            import glob
            matching_files = glob.glob(os.path.join(SYNTHETIC_DATA_DIR, pattern))
            if matching_files:
                found_file = matching_files[0]
                break
        
        if found_file:
            cache_path = found_file
            logging.info(f"Found alternative synthetic data file: {cache_path}")
        else:
            logging.error(f"No synthetic data files found for class {minority_class}")
            return []
    
    # Load the synthetic samples
    try:
        with open(cache_path, 'r') as f:
            all_synthetic_samples = json.load(f)
        
        # Select only the number needed for balancing
        synthetic_samples = all_synthetic_samples[:class_imbalance]
        logging.info(f"Using {len(synthetic_samples)}/{len(all_synthetic_samples)} samples from cache to balance dataset")
        
        return synthetic_samples
    except Exception as e:
        logging.error(f"Error loading synthetic samples: {e}")
        return []

def balance_dataset(df, progress_tracker=None):
    """Balance dataset using synthetic samples for the minority class."""
    # Check class balance
    class_counts = df["label"].value_counts()
    
    if len(class_counts) < 2:
        logging.warning("Dataset has only one class, cannot balance")
        return df
    
    # Convert float keys to int if needed
    class_counts_dict = {int(k): v for k, v in class_counts.items()}
    class_counts = pd.Series(class_counts_dict)
    
    # Identify minority and majority classes
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    class_imbalance = class_counts[majority_class] - class_counts[minority_class]
    
    logging.info(f"Class distribution: {class_counts.to_dict()}")
    logging.info(f"Minority class: {minority_class}, Majority class: {majority_class}")
    logging.info(f"Class imbalance: {class_imbalance}")
    
    if class_imbalance <= 0:
        logging.info("Dataset already balanced")
        return df
    
    if progress_tracker:
        progress_tracker.start_step(f"Loading {class_imbalance} synthetic samples for class {minority_class}")
    
    # Load synthetic samples
    synthetic_samples = load_synthetic_samples(minority_class, class_imbalance)
    
    if not synthetic_samples:
        logging.warning("No synthetic samples available for balancing. Using alternative method.")
        # Alternative: Oversample the minority class
        minority_df = df[df["label"] == minority_class]
        if len(minority_df) > 0:
            # Oversample with replacement if needed
            replacement = class_imbalance > len(minority_df)
            oversampled = minority_df.sample(n=class_imbalance, replace=replacement, random_state=42)
            df = pd.concat([df, oversampled], ignore_index=True)
            logging.info(f"Balanced dataset by oversampling minority class. New distribution: {df['label'].value_counts().to_dict()}")
        else:
            logging.error("Cannot balance dataset - no minority class examples")
        
        if progress_tracker:
            progress_tracker.complete_step()
        return df
    
    # Process synthetic samples to extract features
    synthetic_rows = []
    for sample in synthetic_samples:
        try:
            if isinstance(sample, dict):
                # Create a new dictionary with the expected keys
                processed_sample = {}
                
                # Map 'content' to standardized 'content'
                if "content" in sample:
                    processed_sample["content"] = sample["content"]
                elif "Content" in sample:
                    processed_sample["content"] = sample["Content"]
                else:
                    # Skip samples without content
                    continue
                
                # Map 'label' to standardized 'label'
                if "label" in sample:
                    processed_sample["label"] = int(sample["label"])
                elif "Verdict" in sample:
                    processed_sample["label"] = int(sample["Verdict"])
                else:
                    processed_sample["label"] = int(minority_class)
                
                # Force label to be the minority class
                processed_sample["label"] = int(minority_class)
                
                # Add source and media fields
                processed_sample["source"] = "synthetic"
                processed_sample["media"] = "synthetic"
                
                # Process text to extract features
                full_processed = process_text(processed_sample)
                if full_processed:
                    synthetic_rows.append(full_processed)
            elif isinstance(sample, str):
                # If samples are just text strings
                row = {
                    "content": sample,
                    "label": int(minority_class),
                    "source": "synthetic",
                    "media": "synthetic"
                }
                processed = process_text(row)
                if processed:
                    synthetic_rows.append(processed)
        except Exception as e:
            logging.warning(f"Error processing synthetic sample: {e}")
            continue
    
    logging.info(f"Processed {len(synthetic_rows)} synthetic samples")
    
    # Combine with original data
    if len(synthetic_rows) > 0:
        synthetic_df = pd.DataFrame(synthetic_rows)
        
        # Verify synthetic samples have the right class
        logging.info(f"Synthetic data class distribution: {synthetic_df['label'].value_counts().to_dict()}")
        
        # Combine datasets
        balanced_df = pd.concat([df, synthetic_df], ignore_index=True)
        
        # Shuffle dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logging.info(f"New class distribution: {balanced_df['label'].value_counts().to_dict()}")
        
        if progress_tracker:
            progress_tracker.complete_step()
            
        return balanced_df
    else:
        logging.warning("No valid synthetic samples were processed. Returning original dataset.")
        if progress_tracker:
            progress_tracker.complete_step()
        return df

def tokenize_dataset(df, tokenizer, max_length=512, batch_size=128):
    """Tokenize dataset content using the specified tokenizer."""
    if tokenizer is None:
        logging.error("Tokenizer not provided")
        return df
    
    contents = df["content"].tolist()
    
    # Process in batches to avoid memory issues
    all_tokens = []
    n_batches = (len(contents) + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Tokenizing dataset"):
        batch = contents[i*batch_size:(i+1)*batch_size]
        
        # Skip empty texts
        batch = [text if isinstance(text, str) and text else "" for text in batch]
        
        # Tokenize batch
        encoded = tokenizer(
            batch,
            padding="max_length" if PREPROCESSING_CONFIG["padding"] == "max_length" else True,
            truncation=PREPROCESSING_CONFIG["truncation"],
            max_length=max_length,
            return_tensors=None
        )
        
        # Store token IDs
        all_tokens.extend(encoded["input_ids"])
    
    # Add tokens to dataframe
    df["tokens"] = all_tokens
    
    return df

def load_and_preprocess(dataset_name, tokenizer=None, balance=True, progress_tracker=None):
    """
    Load and preprocess a dataset.
    
    Args:
        dataset_name: Name of the dataset file (without extension)
        tokenizer: Optional tokenizer for text tokenization
        balance: Whether to balance the dataset
        progress_tracker: Optional progress tracker
    
    Returns:
        Processed dataframe
    """
    # Initialize progress tracking
    if progress_tracker is None:
        num_steps = 4 if tokenizer is None else 5
        progress_tracker = ProgressTracker(total_steps=num_steps)
    
    # Step 1: Load dataset
    progress_tracker.start_step(f"Loading dataset: {dataset_name}")
    
    # Look for dataset in data directory with various extensions
    extensions = ['.csv', '.json', '.tsv', '.xlsx']
    dataset_path = None
    
    for ext in extensions:
        path = os.path.join(DATA_DIR, f"{dataset_name}{ext}")
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        logging.error(f"Dataset not found: {dataset_name}")
        return None
    
    # Load dataset based on file extension
    try:
        ext = os.path.splitext(dataset_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(dataset_path)
        elif ext == '.json':
            df = pd.read_json(dataset_path)
        elif ext == '.tsv':
            df = pd.read_csv(dataset_path, sep='\t')
        elif ext == '.xlsx':
            df = pd.read_excel(dataset_path)
        else:
            logging.error(f"Unsupported file format: {ext}")
            return None
        
        logging.info(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None
    
    # Standardize column names
    column_mapping = {}
    
    # Map content column
    content_candidates = ['content', 'text', 'tweet', 'message', 'post']
    for col in content_candidates:
        if col in df.columns:
            column_mapping[col] = 'content'
            break
    
    # Map label column
    label_candidates = ['label', 'class', 'misinfo', 'misinformation', 'verdict', 'is_misinfo', 'is_misinformation']
    for col in label_candidates:
        if col in df.columns:
            column_mapping[col] = 'label'
            break
    
    # Apply column mapping if needed
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Validate required columns
    if 'content' not in df.columns:
        logging.error("Dataset missing required 'content' column")
        return None
    
    if 'label' not in df.columns:
        logging.warning("Dataset missing 'label' column, assuming all examples are negative (0)")
        df['label'] = 0
    
    # Drop duplicates
    initial_count = len(df)
    df.drop_duplicates(subset=['content'], keep='first', inplace=True)
    dedup_count = len(df)
    
    if initial_count > dedup_count:
        logging.info(f"Removed {initial_count - dedup_count} duplicate entries")
    
    progress_tracker.complete_step()
    
    # Step 2: Initialize NLP models
    progress_tracker.start_step("Initializing NLP models")
    initialize_nlp_models()
    progress_tracker.complete_step()
    
    # Step 3: Process texts
    progress_tracker.start_step(f"Processing {len(df)} texts")
    
    # Process in batches with parallel execution
    batch_size = PREPROCESSING_CONFIG["batch_size"]
    n_batches = (len(df) + batch_size - 1) // batch_size
    
    processed_rows = []
    
    for i in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        
        batch = df.iloc[start_idx:end_idx].to_dict(orient="records")
        
        # Process batch using multiprocessing
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            batch_results = list(executor.map(process_text, batch))
        
        # Filter out None results
        valid_results = [row for row in batch_results if row is not None]
        processed_rows.extend(valid_results)
        
        # Log progress
        logging.info(f"Processed batch {i+1}/{n_batches} ({len(valid_results)} valid rows)")
    
    # Create processed dataframe
    df_processed = pd.DataFrame(processed_rows)
    
    if len(df_processed) == 0:
        logging.error("No valid rows after processing")
        return None
    
    progress_tracker.complete_step()
    
    # Step 4: Balance dataset if requested
    if balance:
        progress_tracker.start_step("Balancing dataset")
        df_balanced = balance_dataset(df_processed, progress_tracker=None)
        progress_tracker.complete_step()
    else:
        df_balanced = df_processed
    
    # Step 5: Tokenize dataset if tokenizer provided
    if tokenizer is not None:
        progress_tracker.start_step("Tokenizing dataset")
        df_tokenized = tokenize_dataset(
            df_balanced, 
            tokenizer, 
            max_length=PREPROCESSING_CONFIG["max_token_length"],
            batch_size=batch_size
        )
        progress_tracker.complete_step()
    else:
        df_tokenized = df_balanced
    
    # Save processed dataset
    output_dir = Path(PROCESSED_DATA_DIR)
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"{dataset_name}_processed.json"
    df_tokenized.to_json(str(output_path), orient="records")
    
    # Save split datasets
    X = df_tokenized.drop('label', axis=1)
    y = df_tokenized['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_json(str(output_dir / f"{dataset_name}_train.json"), orient="records")
    val_df.to_json(str(output_dir / f"{dataset_name}_val.json"), orient="records")
    test_df.to_json(str(output_dir / f"{dataset_name}_test.json"), orient="records")
    
    logging.info(f"Saved processed dataset to {output_path}")
    logging.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Return the processed dataframe
    return df_tokenized

def create_prompt(content, features=None, use_enhanced_prompt=False):
    """
    Create instruction-tuning prompt for misinformation detection.
    
    Args:
        content: The text content to analyze
        features: Optional dictionary of features for enhanced prompting
        use_enhanced_prompt: Whether to include feature context in the prompt
    
    Returns:
        Formatted prompt string
    """
    base_prompt = (
        "[INST] Analyze the following text and determine if it contains "
        "misinformation about refugees or migrants. "
    )
    
    if use_enhanced_prompt and features is not None:
        # Add key features as context
        feature_context = ""
        
        # Add sentiment information if available
        if "sentiment_polarity" in features:
            polarity = features["sentiment_polarity"]
            sentiment_desc = "negative" if polarity < -0.2 else "positive" if polarity > 0.2 else "neutral"
            feature_context += f"This text has {sentiment_desc} sentiment (polarity: {polarity:.2f}). "
        
        # Add length information if available
        if "char_count" in features and "word_count" in features:
            feature_context += (
                f"The text is {features['word_count']} words ({features['char_count']} chars) long. "
            )
        
        # Add complexity information if available
        if "avg_word_length" in features and "readability_score" in features:
            complexity = "complex" if features["readability_score"] > 20 else "simple"
            feature_context += (
                f"It has {complexity} language (avg word length: {features['avg_word_length']:.2f}, "
                f"readability score: {features['readability_score']:.2f}). "
            )
        
        if feature_context:
            base_prompt += "Context: " + feature_context
    
    # Add the content
    base_prompt += f"Text: {content} [/INST]"
    
    return base_prompt

def preprocess_for_training(examples, tokenizer, use_enhanced_prompt=False, dataset_df=None):
    """
    Tokenize input texts with prompt format and integrate numerical features.
    
    Args:
        examples: Dictionary of examples with 'content' and other fields
        tokenizer: Tokenizer for encoding
        use_enhanced_prompt: Whether to use enhanced prompts with features
        dataset_df: DataFrame with feature data for enhanced prompts
    
    Returns:
        Dictionary with model inputs
    """
    if use_enhanced_prompt and dataset_df is not None:
        # Map content to features for enhanced prompts
        content_to_features = {}
        for idx, row in dataset_df.iterrows():
            content_to_features[row['content']] = {
                feature: row[feature] for feature in IMPORTANT_FEATURES if feature in row
            }
        
        # Create prompts with feature context
        prompts = []
        for content in examples["content"]:
            features = content_to_features.get(content, {})
            prompts.append(create_prompt(content, features, use_enhanced_prompt=True))
    else:
        # Create basic instruction-tuning prompts
        prompts = [create_prompt(content) for content in examples["content"]]
    
    # Tokenize with proper padding and truncation
    model_inputs = tokenizer(
        prompts, 
        padding=PREPROCESSING_CONFIG["padding"],
        truncation=PREPROCESSING_CONFIG["truncation"],
        max_length=PREPROCESSING_CONFIG["max_token_length"],
        return_tensors=None
    )

    # Labels
    model_inputs["labels"] = examples["label"]

    return model_inputs

def load_processed_dataset(dataset_name, split=None):
    """
    Load a preprocessed dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Optional split to load ('train', 'val', 'test')
    
    Returns:
        Dataframe with the loaded dataset
    """
    output_dir = Path(PROCESSED_DATA_DIR)
    
    if split is not None:
        file_path = output_dir / f"{dataset_name}_{split}.json"
    else:
        file_path = output_dir / f"{dataset_name}_processed.json"
    
    if not file_path.exists():
        logging.error(f"Processed dataset not found: {file_path}")
        return None
    
    try:
        df = pd.read_json(str(file_path))
        logging.info(f"Loaded processed dataset from {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error loading processed dataset: {e}")
        return None