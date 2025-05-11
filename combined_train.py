#!/usr/bin/env python
# coding=utf-8
"""
Combined training script for fine-tuning language models.
This script combines features from Train.py and phi-train.ipynb to create
a comprehensive fine-tuning solution.
"""

import os
import json
import logging
import random
import sys
import time
import gc
import re
import collections
import psutil
import numpy as np
import torch
import datasets
import transformers
from tqdm.auto import tqdm
from datasets import load_dataset, ClassLabel
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import set_seed

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)
datasets.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Memory management functions
def cleanup_memory():
    """Force garbage collection and clear CUDA cache if available."""
    # Get memory usage before cleanup
    before = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    
    # Perform cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get memory usage after cleanup
    after = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    logger.info(f"Memory cleaned up. Before: {before:.2f} MB, After: {after:.2f} MB, Freed: {before - after:.2f} MB")
    
    # Print system memory info
    mem = psutil.virtual_memory()
    logger.info(f"System memory: {mem.percent}% used, {mem.available / 1024 / 1024:.2f} MB available")

def monitor_resources():
    """Monitor and report system resources."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    logger.info(f"\nSystem Resources:")
    logger.info(f"CPU Usage: {cpu_percent}%")
    logger.info(f"Process Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    logger.info(f"System Memory: {mem.percent}% used, {mem.available / 1024 / 1024:.2f} MB available\n")

# Dataset loading with retry logic
def load_dataset_with_retry(dataset_id, max_retries=3, retry_delay=5):
    """Load a dataset with retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading dataset (attempt {attempt+1}/{max_retries})...")
            data = load_dataset(dataset_id, trust_remote_code=True)
            logger.info(f"Dataset loaded successfully with {len(data['train'])} examples")
            return data
        except Exception as e:
            logger.error(f"Error loading dataset (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Maximum retries reached. Could not load dataset.")
                raise

def verify_dataset_structure(dataset):
    """Verify that the dataset has the expected structure and columns."""
    required_columns = ['repo_name', 'path', 'content']
    if 'train' not in dataset:
        logger.warning("Dataset does not have a 'train' split.")
        return False
    
    missing_columns = [col for col in required_columns if col not in dataset['train'].column_names]
    if missing_columns:
        logger.warning(f"Dataset is missing required columns: {missing_columns}")
        return False
    
    logger.info("Dataset structure verification passed.")
    return True

# File type categorization for Swift files
def extract_file_type(path):
    """
    Extract the file type/category based on the file path and naming conventions in Swift projects.
    
    Args:
        path (str): The file path
        
    Returns:
        int: The category label (0-5)
    """
    path_lower = path.lower()
    filename = path.split('/')[-1].lower()
    
    # Category 0: Models - Data structures and model definitions
    if ('model' in path_lower or 
        'struct' in path_lower or 
        'entity' in path_lower or
        'data' in path_lower and 'class' in path_lower):
        return 0
    
    # Category 1: Views - UI related files
    elif ('view' in path_lower or 
          'ui' in path_lower or 
          'screen' in path_lower or 
          'page' in path_lower or
          'controller' in path_lower and 'view' in path_lower):
        return 1
    
    # Category 2: Controllers - Application logic
    elif ('controller' in path_lower or 
          'manager' in path_lower or 
          'coordinator' in path_lower or
          'service' in path_lower):
        return 2
    
    # Category 3: Utilities - Helper functions and extensions
    elif ('util' in path_lower or 
          'helper' in path_lower or 
          'extension' in path_lower or
          'common' in path_lower):
        return 3
    
    # Category 4: Tests - Test files
    elif ('test' in path_lower or 
          'spec' in path_lower or 
          'mock' in path_lower):
        return 4
    
    # Category 5: Configuration - Package and configuration files
    elif ('package.swift' in path_lower or 
          'config' in path_lower or 
          'settings' in path_lower or
          'info.plist' in path_lower):
        return 5
    
    # Default to category 3 (Utilities) if no clear category is found
    return 3

# Define category names for better readability
category_names = {
    0: "Models",
    1: "Views",
    2: "Controllers",
    3: "Utilities",
    4: "Tests",
    5: "Configuration"
}

def create_instruction_prompt(example):
    """Convert a code example into an instruction-based prompt for language learning."""
    code = example['content']
    label = example['label']
    category = category_names.get(label, f"Unknown-{label}")
    
    # Create different types of prompts to help the model learn the language
    prompt_types = [
        # Explain code functionality
        "Explain what this Swift code does and how it works:\n\n",
        
        # Identify patterns and features
        "Identify and explain the key Swift language features used in this code:\n\n",
        
        # Complete or extend code
        "Complete or extend this Swift code with appropriate functionality:\n\n",
        
        # Fix or improve code
        "Suggest improvements or best practices for this Swift code:\n\n",
        
        # Understand code structure
        f"This is a Swift {category.lower()} file. Explain its structure and purpose:\n\n",
        
        # Code generation tasks
        "Write a Swift function that accomplishes the same task as this code but more efficiently:\n\n",
        
        # Language understanding
        "Explain the Swift syntax and language features demonstrated in this code:\n\n",
        
        # Learning from examples
        "Study this Swift code example and explain what you can learn from it:\n\n"
    ]
    
    # Select a random prompt type
    instruction = random.choice(prompt_types)
    
    code_section = f"```swift\n{code}\n```\n\n"
    
    # Create the full prompt
    prompt = instruction + code_section
    
    # Create a detailed response based on the prompt type and code category
    if "Explain what this Swift code does" in instruction:
        response = f"This Swift code is a {category.lower()} file that "
        if category == "Models":
            response += "defines data structures and model objects. "
        elif category == "Views":
            response += "implements user interface components. "
        elif category == "Controllers":
            response += "manages application logic and coordinates between models and views. "
        elif category == "Utilities":
            response += "provides helper functions and extensions. "
        elif category == "Tests":
            response += "contains test cases to verify functionality. "
        elif category == "Configuration":
            response += "configures application settings and parameters. "
        
        response += "The code uses Swift syntax with "
    else:
        # Generic response starter for other prompt types
        response = f"Based on the Swift {category.lower()} code provided, "
    
    # Add a placeholder for the model to continue
    response += "[model should continue the response based on the code]"
    
    # Combine prompt and response into a full text for training
    full_text = f"{prompt}\n\n{response}"
    
    return {
        "text": full_text,
        "prompt": prompt,
        "response": response,
        "label": label,
        "category": category
    }

def tokenize_instruction(examples, tokenizer, max_length):
    """Tokenize the instruction text with explicit padding and truncation settings."""
    # Process one example at a time to avoid dimension issues
    results = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for text in examples['text']:
        # Tokenize with explicit padding and truncation settings
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None  # Return Python lists, not PyTorch tensors
        )
        
        # Add to results
        results["input_ids"].append(encoded["input_ids"])
        results["attention_mask"].append(encoded["attention_mask"])
        results["labels"].append(encoded["input_ids"].copy())  # Copy input_ids for labels
    
    return results

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Ensure all features have the same keys
        if not all(k in features[0] for k in ["input_ids", "attention_mask", "labels"]):
            raise ValueError("Some features are missing required keys")
        
        # Create a batch with proper padding
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }
        
        return batch

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune a language model")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-128k-instruct",
                        help="Model identifier from huggingface.co/models")
    parser.add_argument("--dataset_id", type=str, default="mvasiliniuc/iva-swift-codeint",
                        help="Dataset identifier from huggingface.co/datasets")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model",
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with smaller dataset")
    parser.add_argument("--debug_sample_size", type=int, default=100,
                        help="Sample size for debug mode")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU - Note: Training will be much slower on CPU")
    
    # Log configuration
    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Max sequence length: {args.max_length}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"LoRA rank: {args.lora_r}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset: {args.dataset_id}")
        data = load_dataset_with_retry(args.dataset_id)
        logger.info("Dataset structure:")
        logger.info(data)
        
        # If in debug mode, take a small sample of the dataset
        if args.debug and 'train' in data:
            logger.info(f"DEBUG MODE: Sampling {args.debug_sample_size} examples from dataset")
            # Take a stratified sample if possible
            data['train'] = data['train'].shuffle(seed=args.seed).select(range(min(args.debug_sample_size, len(data['train']))))
            logger.info(f"Reduced dataset size: {len(data['train'])} examples")
        
        # Verify dataset structure
        dataset_valid = verify_dataset_structure(data)
        if not dataset_valid:
            logger.warning("Dataset structure is not as expected. Proceeding with caution.")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.max_length)
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
            logger.info(f"Tokenizer type: {tokenizer.__class__.__name__}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Apply the function to create labels
        try:
            # Create a new column with the extracted labels
            labeled_data = data['train'].map(lambda example: {
                **example,
                'label': extract_file_type(example['path'])
            })
            
            # Count the distribution of labels
            label_counts = collections.Counter(labeled_data['label'])
            
            logger.info("Label distribution:")
            for label, count in sorted(label_counts.items()):
                category_name = category_names.get(label, f"Unknown-{label}")
                logger.info(f"Label {label} ({category_name}): {count} examples ({count/len(labeled_data)*100:.2f}%)")
            
            # Get unique labels
            unique_labels = sorted(label_counts.keys())
            num_labels = len(unique_labels)
            
            logger.info(f"\nTotal unique labels: {num_labels}")
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise
        
        # Split the data into train, validation, and test sets
        try:
            # Shuffle the data
            shuffled_data = labeled_data.shuffle(seed=args.seed)
            
            # Split into train (80%), validation (10%), and test (10%)
            train_size = int(0.8 * len(shuffled_data))
            val_size = int(0.1 * len(shuffled_data))
            
            train_data = shuffled_data.select(range(train_size))
            val_data = shuffled_data.select(range(train_size, train_size + val_size))
            test_data = shuffled_data.select(range(train_size + val_size, len(shuffled_data)))
            
            logger.info(f"Training set size: {len(train_data)}")
            logger.info(f"Training set label distribution: {collections.Counter(train_data['label'])}")
            logger.info(f"Validation set size: {len(val_data)}")
            logger.info(f"Validation set label distribution: {collections.Counter(val_data['label'])}")
            logger.info(f"Test set size: {len(test_data)}")
            logger.info(f"Test set label distribution: {collections.Counter(test_data['label'])}")
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
        
        # Apply the function to create instruction-based datasets
        try:
            # Create instruction datasets
            train_instructions = train_data.map(create_instruction_prompt)
            val_instructions = val_data.map(create_instruction_prompt)
            test_instructions = test_data.map(create_instruction_prompt)
            
            # Print an example to verify
            logger.info("Example instruction prompt:")
            logger.info("-" * 80)
            logger.info(train_instructions[0]['text'])
            logger.info("-" * 80)
            
            logger.info(f"Created {len(train_instructions)} training instructions")
        except Exception as e:
            logger.error(f"Error creating instruction prompts: {e}")
            raise
        
        # Tokenize the datasets
        try:
            # Apply tokenization to each split
            tokenize_fn = lambda examples: tokenize_instruction(examples, tokenizer, args.max_length)
            
            tokenized_train = train_instructions.map(
                tokenize_fn,
                batched=True,
                remove_columns=['repo_name', 'path', 'content', 'label', 'text', 'prompt', 'response', 'category']
            )
            
            tokenized_val = val_instructions.map(
                tokenize_fn,
                batched=True,
                remove_columns=['repo_name', 'path', 'content', 'label', 'text', 'prompt', 'response', 'category']
            )
            
            # Set the format for PyTorch
            tokenized_train.set_format("torch")
            tokenized_val.set_format("torch")
            
            logger.info(f"Tokenized {len(tokenized_train)} training examples")
        except Exception as e:
            logger.error(f"Error tokenizing data: {e}")
            raise
        
        # Load model with quantization
        try:
            logger.info("Loading model...")
            
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Load the model for causal language modeling
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Prepare the model for training
            model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            # Apply LoRA to the model
            model = get_peft_model(model, lora_config)
            
            # Print trainable parameters
            model.print_trainable_parameters()
            
            logger.info(f"Model loaded for instruction tuning")
            logger.info(f"Model type: {model.__class__.__name__}")
            
            # Check memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"Memory usage after model loading: {memory_info.rss / 1024 / 1024:.2f} MB")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Create data collator
        data_collator = CustomDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal language modeling, not masked language modeling
        )
        
        # Early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            evaluation_strategy="steps",
            eval_steps=0.1,  # Evaluate every 10% of training
            save_strategy="steps",
            save_steps=0.1,  # Save every 10% of training
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=100,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
            push_to_hub=False,
            disable_tqdm=False,
            seed=args.seed,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping_callback]
        )
        
        logger.info("Training setup complete")
        
        # Monitor resources before training
        logger.info("Resources before training:")
        monitor_resources()
        
        # Train the model
        try:
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Monitor resources after training
            logger.info("Resources after training:")
            monitor_resources()
            
            # Print training results
            logger.info(f"Training completed in {train_result.metrics['train_runtime']:.2f} seconds")
            logger.info(f"Training loss: {train_result.metrics['train_loss']:.4f}")
            
            # Save the model
            trainer.save_model(args.output_dir)
            logger.info(f"Model saved to {args.output_dir}")
            
            # Clean up memory
            cleanup_memory()
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Evaluate the model
        try:
            logger.info("Evaluating model...")
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            
        logger.info("Training and evaluation complete!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()