# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training Phi-3-mini-128k-instruct to Learn Swift Programming Language on TPU
#
# This notebook trains Microsoft's Phi-3-mini-128k-instruct model to understand and work with Swift code using a dataset of real Swift files. This version is specifically optimized for TPU training.

# %%
# Install required libraries
!pip install transformers datasets evaluate torch scikit-learn tqdm dropbox requests accelerate peft 'torch_xla[tpu]>=2.0'

# Set environment variables for TPU
import os
os.environ["TPU_NAME"] = "local"  # For Google Cloud TPU environments, set appropriately

# %%
# Import required libraries
import torch
import numpy as np
import random
import time
import collections
import psutil
import os
import gc
import json
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import TPU-specific libraries
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.distributed.parallel_loader import ParallelLoader
import torch_xla.debug.metrics as met

# Define memory cleanup function for TPU
def cleanup_memory():
    """Clean up memory to avoid fragmentation."""
    print("Cleaning up memory...")
    gc.collect()
    if xm.xrt_world_size() > 0:
        # For TPU, synchronize after cleanup
        xm.mark_step()
        print("TPU memory synchronized")
        
# Define resource monitoring function for TPU
def monitor_resources():
    """Monitor system and TPU resources."""
    # Monitor CPU and RAM
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"CPU memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # Monitor TPU if available
    if xm.xrt_world_size() > 0:
        print(f"TPU device: {xm.get_device_type()}")
        print(f"Number of TPU devices: {xm.xrt_world_size()}")
        
        # Print TPU memory usage metrics
        print("\nTPU Memory Usage:")
        tpu_metrics = met.metrics_report()
        print(tpu_metrics)
        
        # Extract and print specific TPU memory metrics if available
        memory_metrics = [m for m in str(tpu_metrics).split('\n') if 'mem' in m.lower()]
        for metric in memory_metrics:
            print(f"  {metric}")


# %%
# Configure TPU device
device = xm.xla_device()
print(f"Using device: {device}")
print(f"Number of TPU devices: {xm.xrt_world_size()}")
print(f"TPU type: {xm.get_device_type()}")

# Print TPU configuration details
if xm.xrt_world_size() > 0:
    print("TPU Configuration:")
    print(f"  TPU cores: {xm.xrt_world_size()}")
    print(f"  Local ordinal: {xm.get_local_ordinal()}")
    print(f"  Global ordinal: {xm.get_ordinal()}")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# %%
# Dataset configuration - using the same dataset as the original notebook
DATASET_ID = "mvasiliniuc/iva-swift-codeint"

# Model configuration - using Phi-3-mini-128k-instruct
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
MAX_LENGTH = 2048  # Reduced from 4096 to save memory
BATCH_SIZE = 1  # Reduced batch size to avoid OOM errors
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to compensate for smaller batch size

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Debug mode for testing with smaller dataset
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 100

print(f"Using model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Batch size: {BATCH_SIZE} per device")
print(f"Effective batch size: {BATCH_SIZE * xm.xrt_world_size() * GRADIENT_ACCUMULATION_STEPS}")
print(f"LoRA rank: {LORA_R}")


# %%
# Function to load dataset with retry logic
def load_dataset_with_retry(dataset_id, max_retries=3, retry_delay=5):
    """Load a dataset with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Loading dataset (attempt {attempt+1}/{max_retries})...")
            data = load_dataset(dataset_id, trust_remote_code=True)
            print(f"Dataset loaded successfully with {len(data['train'])} examples")
            return data
        except Exception as e:
            print(f"Error loading dataset (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Maximum retries reached. Could not load dataset.")
                raise

# Load the dataset with retry logic
try:
    print(f"Loading dataset: {DATASET_ID}")
    data = load_dataset_with_retry(DATASET_ID)
    print("Dataset structure:")
    print(data)
    
    # If in debug mode, take a small sample of the dataset
    if DEBUG_MODE and 'train' in data:
        print(f"DEBUG MODE: Sampling {DEBUG_SAMPLE_SIZE} examples from dataset")
        # Take a stratified sample if possible
        data['train'] = data['train'].shuffle(seed=42).select(range(min(DEBUG_SAMPLE_SIZE, len(data['train']))))
        print(f"Reduced dataset size: {len(data['train'])} examples")
        
except Exception as e:
    print(f"Fatal error loading dataset: {e}")
    raise


# %%
# Verify dataset structure and column names
def verify_dataset_structure(dataset):
    """Verify that the dataset has the expected structure and columns."""
    required_columns = ['repo_name', 'path', 'content']
    if 'train' not in dataset:
        print("WARNING: Dataset does not have a 'train' split.")
        return False
    
    missing_columns = [col for col in required_columns if col not in dataset['train'].column_names]
    if missing_columns:
        print(f"WARNING: Dataset is missing required columns: {missing_columns}")
        return False
    
    print("Dataset structure verification passed.")
    return True

# Verify dataset structure
dataset_valid = verify_dataset_structure(data)
if not dataset_valid:
    print("Dataset structure is not as expected. Proceeding with caution.")

# %%
# Load the Phi-3 tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_LENGTH)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Tokenizer type: {tokenizer.__class__.__name__}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise


# %%
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

# %%
# Apply the function to create labels
try:
    # Create a new column with the extracted labels
    labeled_data = data['train'].map(lambda example: {
        **example,
        'label': extract_file_type(example['path'])
    })
    
    # Count the distribution of labels
    label_counts = collections.Counter(labeled_data['label'])
    
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        category_name = category_names.get(label, f"Unknown-{label}")
        print(f"Label {label} ({category_name}): {count} examples ({count/len(labeled_data)*100:.2f}%)")
    
    # Get unique labels
    unique_labels = sorted(label_counts.keys())
    num_labels = len(unique_labels)
    
    print(f"\nTotal unique labels: {num_labels}")
except Exception as e:
    print(f"Error in data preparation: {e}")
    raise

# %%
# Split the data into train, validation, and test sets
try:
    # Shuffle the data
    shuffled_data = labeled_data.shuffle(seed=42)
    
    # Split into train (80%), validation (10%), and test (10%)
    train_size = int(0.8 * len(shuffled_data))
    val_size = int(0.1 * len(shuffled_data))
    
    train_data = shuffled_data.select(range(train_size))
    val_data = shuffled_data.select(range(train_size, train_size + val_size))
    test_data = shuffled_data.select(range(train_size + val_size, len(shuffled_data)))
    
    print(f"Training set size: {len(train_data)}")
    print(f"Training set label distribution: {collections.Counter(train_data['label'])}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Validation set label distribution: {collections.Counter(val_data['label'])}")
    print(f"Test set size: {len(test_data)}")
    print(f"Test set label distribution: {collections.Counter(test_data['label'])}")
except Exception as e:
    print(f"Error splitting data: {e}")
    raise


# %%
# Create instruction-based prompts for the model
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
        
        # Add some language-specific details based on code content
        if "class" in code:
            response += "class definitions, "
        if "struct" in code:
            response += "struct definitions, "
        if "func" in code:
            response += "function declarations, "
        if "var" in code:
            response += "variable declarations, "
        if "let" in code:
            response += "constant declarations, "
        if "guard" in code or "if let" in code:
            response += "optional unwrapping, "
        if "extension" in code:
            response += "extensions, "
        if "protocol" in code:
            response += "protocol implementations, "
            
        # Remove trailing comma and space if present
        if response.endswith(", "):
            response = response[:-2] + "."
        else:
            response += "various Swift features."
    
    elif "Identify and explain the key Swift language features" in instruction:
        response = "This Swift code demonstrates several key language features:\n\n"
        
        # Add language features based on code content
        features = []
        if "class" in code:
            features.append("1. **Classes**: Swift classes are reference types that support inheritance and reference counting.")
        if "struct" in code:
            features.append("1. **Structs**: Swift structs are value types that are copied when assigned or passed as arguments.")
        if "protocol" in code:
            features.append("1. **Protocols**: Similar to interfaces in other languages, protocols define a blueprint of methods, properties, and requirements.")
        if "extension" in code:
            features.append("1. **Extensions**: Swift allows adding functionality to existing types through extensions.")
        if "guard" in code:
            features.append("1. **Guard statements**: Used for early returns and unwrapping optionals, improving code readability.")
        if "if let" in code or "guard let" in code:
            features.append("1. **Optional binding**: Swift's way of safely unwrapping optional values.")
        if "enum" in code:
            features.append("1. **Enumerations**: Swift enums are first-class types that can have methods and computed properties.")
        if "func" in code:
            features.append("1. **Functions**: Swift functions can have parameters, return values, and support closures.")
        
        # If no specific features were identified, add a generic response
        if not features:
            features.append("1. **Swift syntax**: The code demonstrates standard Swift syntax and conventions.")
            features.append("2. **Type safety**: Swift's strong type system helps prevent errors at compile time.")
            features.append("3. **Readability**: Swift's clean syntax makes code easy to read and maintain.")
        
        # Renumber the features
        for i, feature in enumerate(features):
            feature_parts = feature.split(": ", 1)
            if len(feature_parts) == 2:
                features[i] = f"{i+1}. **{feature_parts[0].split('**')[1]}**: {feature_parts[1]}"
        
        response += "\n".join(features)
    
    elif "Complete or extend this Swift code" in instruction or "Write a Swift function" in instruction:
        # For code generation tasks, provide a thoughtful response about how to approach the task
        response = f"To extend this Swift {category.lower()} code, I would consider the following approach:\n\n"
        
        if category == "Models":
            response += "1. Add additional properties to capture more data attributes\n"
            response += "2. Implement Codable protocol for easy JSON serialization\n"
            response += "3. Add validation methods to ensure data integrity\n"
            response += "4. Include computed properties for derived values\n\n"
            response += "Here's an implementation example:\n\n```swift\n"
            
            if "struct" in code:
                response += "// Extension to add Codable conformance\nextension MyStruct: Codable {\n    // Codable implementation\n}\n\n"
                response += "// Add validation method\nextension MyStruct {\n    func validate() -> Bool {\n        // Validation logic\n        return true\n    }\n}\n"
            else:
                response += "// Example extension or additional functionality\n// that would be appropriate for this model\n"
            
            response += "```"
            
        elif category == "Views":
            response += "1. Add UI customization options\n"
            response += "2. Implement additional user interaction handlers\n"
            response += "3. Add accessibility support\n"
            response += "4. Implement view lifecycle methods\n\n"
            response += "Here's an implementation example:\n\n```swift\n"
            response += "// Example extension or additional functionality\n// that would be appropriate for this view\n"
            response += "```"
            
        else:
            response += "1. Add error handling to make the code more robust\n"
            response += "2. Implement additional helper methods\n"
            response += "3. Add documentation comments to improve code readability\n"
            response += "4. Consider performance optimizations where appropriate\n\n"
            response += "Here's an implementation example:\n\n```swift\n"
            response += "// Example extension or additional functionality\n// that would be appropriate for this code\n"
            response += "```"
    
    else:
        # Generic response for other prompt types
        response = f"This Swift code demonstrates typical patterns used in {category.lower()} files. "
        response += "It follows Swift language conventions and showcases proper syntax for defining "
        
        if category == "Models":
            response += "data structures with properties and methods. Swift models typically use structs for value semantics or classes when reference semantics are needed. The code demonstrates Swift's strong typing system and property declarations."
        elif category == "Views":
            response += "UI components with layout and interaction logic. Swift views often use UIKit or SwiftUI frameworks, with clear separation of UI elements and their behaviors. The code shows how Swift handles user interface components and event responses."
        elif category == "Controllers":
            response += "application logic and coordination between components. Controllers in Swift manage the flow of data between models and views, implementing business logic and handling user interactions. The code demonstrates Swift's approach to application architecture."
        elif category == "Utilities":
            response += "helper functions and extensions to enhance functionality. Swift utilities often leverage the language's powerful extension capabilities to add functionality to existing types. The code shows how Swift can be extended and customized through utility functions."
        elif category == "Tests":
            response += "test cases with setup, execution, and verification steps. Swift tests typically use XCTest framework with arrange-act-assert pattern. The code demonstrates Swift's approach to unit testing and verification."
        elif category == "Configuration":
            response += "application settings and configuration parameters. Swift configuration files often define constants, environment settings, and application parameters. The code shows how Swift handles application configuration and settings management."
    
    # Combine prompt and response for instruction tuning
    full_text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n"
    
    return {
        "text": full_text,
        "prompt": prompt,
        "response": response,
        "label": label,
        "category": category
    }


# %%
# Apply the function to create instruction-based datasets
try:
    # Create instruction datasets
    train_instructions = train_data.map(create_instruction_prompt)
    val_instructions = val_data.map(create_instruction_prompt)
    test_instructions = test_data.map(create_instruction_prompt)
    
    # Print an example to verify
    print("Example instruction prompt:")
    print("-" * 80)
    print(train_instructions[0]['text'])
    print("-" * 80)
    
    print(f"Created {len(train_instructions)} training instructions")
    print(f"Created {len(val_instructions)} validation instructions")
    print(f"Created {len(test_instructions)} test instructions")
except Exception as e:
    print(f"Error creating instruction prompts: {e}")
    raise


# %%
# Tokenize the instruction data with proper handling of padding and truncation
def tokenize_instruction(examples):
    """Tokenize the instruction text with explicit padding and truncation settings."""
    # Process one example at a time to avoid dimension issues
    results = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for text in examples['text']:
        # Tokenize with explicit padding and truncation settings
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None  # Return Python lists, not PyTorch tensors
        )
        
        # Add to results
        results["input_ids"].append(encoded["input_ids"])
        results["attention_mask"].append(encoded["attention_mask"])
        results["labels"].append(encoded["input_ids"].copy())  # Copy input_ids for labels
    
    return results


# %%
try:
    # Apply tokenization to each split
    tokenized_train = train_instructions.map(
        tokenize_instruction,
        batched=True,
        remove_columns=['repo_name', 'path', 'content', 'label', 'text', 'prompt', 'response', 'category']
    )
    
    tokenized_val = val_instructions.map(
        tokenize_instruction,
        batched=True,
        remove_columns=['repo_name', 'path', 'content', 'label', 'text', 'prompt', 'response', 'category']
    )
    
    # Set the format for PyTorch
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    
    print(f"Tokenized {len(tokenized_train)} training examples")
    print(f"Tokenized {len(tokenized_val)} validation examples")
    print("Data tokenization complete")
except Exception as e:
    print(f"Error tokenizing data: {e}")
    raise

# %%
# Set up training arguments with optimized settings for TPU training
try:
    # Create output directory if it doesn't exist
    os.makedirs("./phi3_tpu_swift_model", exist_ok=True)
    
    # Configure training arguments with TPU-specific settings
    training_args = TrainingArguments(
        output_dir="./phi3_tpu_swift_model",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        # TPU-specific settings
        tpu_num_cores=8,  # Specify the number of TPU cores
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        report_to="none",  # Disable reporting to avoid extra overhead
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    )
    
    print(f"Training arguments configured for TPU training")
    print(f"Using gradient checkpointing: {training_args.gradient_checkpointing}")
    print(f"TPU cores: {training_args.tpu_num_cores}")
    
except Exception as e:
    print(f"Error setting up training arguments: {e}")
    raise

# %%
# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

# TPU-specific callback to monitor and log TPU metrics
class TPUMetricsCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            xm.master_print(f"TPU Step {state.global_step}: {logs}")
            
            # Log TPU metrics periodically
            if state.global_step % 50 == 0:
                tpu_metrics = met.metrics_report()
                metrics_str = str(tpu_metrics)
                
                # Extract and print TPU memory metrics
                memory_metrics = [m for m in metrics_str.split('\n') 
                                 if 'mem' in m.lower() or 'memory' in m.lower()]
                
                xm.master_print("\nTPU Memory Metrics:")
                for metric in memory_metrics:
                    xm.master_print(f"  {metric}")

# Load and prepare the model for TPU
try:
    print(f"Loading {MODEL_NAME} for TPU...")
    
    # Clean up memory before model loading
    cleanup_memory()
    
    # For TPU, we'll use a different approach to model loading compared to 4-bit quantization
    # TPUs work better with standard precision models (bf16 or fp32)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # BF16 is efficient on TPUs
        trust_remote_code=True,
        use_cache=False  # Disable KV cache during training for better memory efficiency
    )
    
    # Move the model to TPU device
    model.to(device)
    print(f"Successfully loaded model on TPU")
    
    # Configure LoRA for efficient fine-tuning
    print("Setting up LoRA fine-tuning...")
    # Get label names from category names
    label_names = [category_names[i] for i in range(len(category_names))]
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        label_names=label_names
    )
    
    # Prepare model for LoRA fine-tuning
    model = get_peft_model(model, lora_config)
    
    # Print information about the model
    print(f"Model loaded and configured with LoRA (rank={LORA_R})")
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # Print TPU device allocation
    print(f"Model is on TPU device: {next(model.parameters()).device}")
    
    # Print TPU memory info
    print("TPU Memory Status after model loading:")
    print(met.metrics_report())
    
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    raise

# %%
# Create a custom data collator for TPU
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

# Create data collator for language modeling
data_collator = CustomDataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked language modeling
)

# %%
# Create TPU-specific trainer
tpu_metrics_callback = TPUMetricsCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback, tpu_metrics_callback]
)

# Verify model device
model_device = next(model.parameters()).device
print(f"Model is on TPU device: {model_device}")

print("TPU training setup complete")


# %%
# Function to monitor TPU resources during training
def monitor_tpu_resources():
    print(f"\nTPU Resources:")
    
    # Monitor CPU memory (for host)
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Process Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"System Memory: {mem.percent}% used, {mem.available / 1024 / 1024:.2f} MB available")
    
    # Monitor TPU memory and metrics
    if xm.xrt_world_size() > 0:
        print("\nTPU Metrics:")
        tpu_metrics = met.metrics_report()
        
        # Print memory related metrics
        print("\nTPU Memory Usage:")
        metrics_str = str(tpu_metrics)
        
        # Extract memory metrics
        memory_metrics = [m for m in metrics_str.split('\n') 
                         if any(word in m.lower() for word in ['mem', 'memory', 'ram'])]
        
        for metric in memory_metrics:
            print(f"  {metric}")
        
        # Extract compilation metrics
        compilation_metrics = [m for m in metrics_str.split('\n') 
                              if any(word in m.lower() for word in ['compile', 'xla', 'execution'])]
        
        print("\nTPU Compilation Metrics:")
        for metric in compilation_metrics:
            print(f"  {metric}")
        
        # Check for TPU errors or problems
        error_metrics = [m for m in metrics_str.split('\n') 
                        if any(word in m.lower() for word in ['error', 'timeout', 'fail'])]
        
        if error_metrics:
            print("\n⚠️ TPU Error Metrics:")
            for metric in error_metrics:
                print(f"  {metric}")
    print("")

# %%
# Create a custom training loop with specific TPU management
try:
    print("Starting training with TPU-specific optimizations...")
    
    # Monitor resources before training
    print("Resources before training:")
    monitor_tpu_resources()
    
    # Additional cleanup before training
    cleanup_memory()
    
    # Custom TPU callback to monitor compilation and execution
    class TPUGuardCallback(transformers.TrainerCallback):
        """Custom callback for TPU monitoring and error prevention"""
        def on_step_end(self, args, state, control, **kwargs):
            # Perform a TPU sync after each step to ensure operations complete
            xm.mark_step()
            
            # Periodically monitor TPU resources
            if state.global_step % 50 == 0:
                xm.master_print(f"\nTPU status at step {state.global_step}:")
                metrics = met.metrics_report()
                
                # Check if we're hitting memory issues
                metrics_str = str(metrics)
                if any(warning in metrics_str.lower() for warning in ["oom", "out of memory", "resource exhausted"]):
                    xm.master_print("⚠️ TPU memory pressure detected!")
                    # Force garbage collection
                    gc.collect()
                    # Mark step to ensure sync
                    xm.mark_step()
                
            # Do deeper analysis every 100 steps
            if state.global_step % 100 == 0 and state.global_step > 0:
                monitor_tpu_resources()
                    
        def on_epoch_end(self, args, state, control, **kwargs):
            # Clean up between epochs
            gc.collect()
            xm.mark_step()
            xm.master_print("\nCompleted epoch - TPU sync point")
    
    # Add our TPU monitoring callback
    trainer.add_callback(TPUGuardCallback())
    
    # Start training with a timeout
    max_training_time = 6 * 60 * 60  # 6 hours max
    start_time = time.time()
    
    # Run training with TPU error handling
    try:
        # Train the model
        print("\nStarting TPU training...")
        train_result = trainer.train()
        
        # Ensure all TPU operations are complete
        xm.mark_step()
        
    except Exception as e:
        print(f"\n🛑 TPU training error: {e}")
        
        # Try to diagnose the issue
        print("TPU diagnostic information:")
        print(met.metrics_report())
        
        # Clean up and try to restart if possible
        gc.collect()
        xm.mark_step()
        
        if "memory" in str(e).lower() or "resource" in str(e).lower():
            # If we hit memory issues, try reducing parameters
            print("\nAttempting recovery with reduced parameters...")
            
            # Reduce sequence length if needed
            if MAX_LENGTH > 1024:
                old_max_len = MAX_LENGTH
                MAX_LENGTH = 1024
                print(f"Reducing sequence length from {old_max_len} to {MAX_LENGTH}")
            
            # Try with smaller batch size and more gradient accumulation
            training_args.per_device_train_batch_size = 1
            training_args.gradient_accumulation_steps = 16
            print("Reduced batch size to 1 and increased gradient accumulation to 16")
            
            # Re-create trainer with updated settings
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[early_stopping_callback, TPUGuardCallback()]
            )
            
            # Try again with new settings
            print("\nRetrying training with reduced TPU memory settings...")
            train_result = trainer.train()
            xm.mark_step()  # Final sync point
    
    # Monitor resources after training
    print("\nResources after training:")
    monitor_tpu_resources()
    
    # Print training results
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.2f} minutes")
    print(f"Training loss: {train_result.metrics['train_loss']:.4f}")
    
    # Save the model - special handling for TPU
    print("\nSynchronizing and saving model...")
    xm.mark_step()  # Ensure all TPU operations complete before saving
    
    # Save on TPU process 0 only to avoid conflicts
    if xm.is_master_ordinal():
        trainer.save_model("./phi3_tpu_swift_model")
        
        # Save adapter separately for easier loading
        if hasattr(model, "save_pretrained"):
            try:
                model.save_pretrained("./phi3_tpu_swift_model_adapter")
                print("Saved LoRA adapter to ./phi3_tpu_swift_model_adapter")
            except Exception as save_error:
                print(f"Error saving adapter: {save_error}")
        
        print("Model saved successfully")
    
    # Final TPU synchronization
    xm.rendezvous("training_complete")
    print("TPU training complete")
    
    # Clean up memory
    cleanup_memory()
    
except Exception as e:
    print(f"\n❌ Error during TPU training: {e}")
    
    # Print stack trace for debugging
    import traceback
    traceback.print_exc()
    
    # Print TPU diagnostic information
    print("\nTPU diagnostic information at error:")
    print(met.metrics_report())
    
    # Try to save checkpoint if possible
    try:
        print("\nAttempting to save checkpoint after error...")
        if xm.is_master_ordinal():
            trainer.save_model("./phi3_tpu_swift_model_checkpoint_after_error")
            print("Emergency checkpoint saved to ./phi3_tpu_swift_model_checkpoint_after_error")
    except:
        print("Could not save emergency checkpoint")
    
    # Monitor resources after error
    print("Resources after error:")
    monitor_tpu_resources()
    
    raise

# %%
# Test the model with Swift code examples (TPU-specific)
try:
    print("Testing the model with Swift code examples on TPU...")
    
    # Function to generate responses for test examples on TPU
    def generate_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Use TPU-specific generation
        with torch.no_grad():
            # Move inputs to TPU
            input_ids = inputs.input_ids
            
            # Generate with TPU-aware operations
            outputs = model.generate(
                input_ids,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Ensure TPU operations complete
            xm.mark_step()
            
        # Return to host for decoding
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        return response
    
    # Test prompts for different Swift language tasks
    test_prompts = [
        # Explain Swift syntax
        "<|user|>\nExplain the key features of Swift's optional unwrapping syntax:\n\n```swift\nfunc processName(_ name: String?) {\n    guard let unwrappedName = name else {\n        print(\"No name provided\")\n        return\n    }\n    print(\"Hello, \\(unwrappedName)!\")\n}\n```\n<|assistant|>",
        
        # Code completion
        "<|user|>\nComplete this Swift function that calculates the factorial of a number:\n\n```swift\nfunc factorial(_ n: Int) -> Int {\n    // Add implementation here\n}\n```\n<|assistant|>",
        
        # Debugging help
        "<|user|>\nWhat's wrong with this Swift code and how can I fix it?\n\n```swift\nclass Person {\n    var name: String\n    var age: Int\n    \n    func greet() {\n        print(\"Hello, my name is \\(name) and I am \\(age) years old.\")\n    }\n}\n\nlet person = Person()\nperson.greet()\n```\n<|assistant|>",
        
        # Swift best practices
        "<|user|>\nExplain Swift best practices for error handling:\n<|assistant|>"
    ]
    
    # Generate and print responses
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest {i+1}:\n{'-'*40}")
        print(f"Prompt: {prompt.split('<|assistant|>')[0].replace('<|user|>', '')}")
        response = generate_response(prompt)
        print(f"\nResponse:\n{response}\n")
    
    print("\nTPU testing complete")
except Exception as e:
    print(f"Error during TPU testing: {e}")
    import traceback
    traceback.print_exc()
