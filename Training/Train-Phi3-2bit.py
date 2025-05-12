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
# # Training Phi-3-mini-128k-instruct to Learn Swift Programming Language
#
# This notebook trains Microsoft's Phi-3-mini-128k-instruct model to understand and work with Swift code using a dataset of real Swift files.
#
# ## Execution Flow for Kaggle
# 
# This notebook is designed to run smoothly on Kaggle or in any Jupyter environment. Cells will execute sequentially in the following order:
# 
# 1. Setup and library installation
# 2. Data preparation
# 3. Model initialization
# 4. Trainer setup
# 5. Training process
# 6. Model testing

# %%
# EXECUTION TRACKING SYSTEM - helps ensure proper execution on Kaggle and Jupyter
# This cell must be executed first

# Create execution tracker
EXECUTION_STATUS = {
    "setup_complete": False,
    "data_loaded": False,
    "model_initialized": False,
    "trainer_created": False,
    "training_complete": False,
    "testing_complete": False
}

def update_status(stage):
    """Update execution status and print progress"""
    global EXECUTION_STATUS
    EXECUTION_STATUS[stage] = True
    
    # Calculate progress
    completed = sum(1 for status in EXECUTION_STATUS.values() if status)
    total = len(EXECUTION_STATUS)
    progress = completed / total * 100
    
    print(f"✓ {stage.replace('_', ' ').title()} - Progress: {progress:.1f}%")
    
    return True

# Check if we're running on Kaggle
try:
    import kaggle
    IS_KAGGLE = True
    print("✓ Detected Kaggle environment - Sequential execution mode active")
except ImportError:
    IS_KAGGLE = False
    print("✓ Standard Jupyter environment detected")

# Flag to begin execution
print("Starting Phi-3 training pipeline...")

# %%
# SECTION 1: SETUP - Install required libraries and configure environment
print("📦 Installing required libraries...")
!pip install transformers datasets evaluate torch scikit-learn tqdm dropbox requests accelerate peft bitsandbytes

# Set PyTorch memory management environment variables to avoid fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Explicitly set to use 2 GPUs

# Update execution status
if 'update_status' in globals():
    update_status("setup_complete")

# %%
# SECTION 1 (cont): Import required libraries
print("📚 Importing libraries and setting up environment...")
import torch
import numpy as np
import random
import time
import collections
import psutil
import os
import gc
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from transformers.trainer_callback import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Import AQLM for 2-bit quantization
try:
    try:
        import aqlm
    except ImportError:
        print("AQLM package not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "-U", "aqlm", "--no-cache-dir"])
        import aqlm
    
    # AQLM correctly imported - we'll use it directly for quantization when loading the model
    print("AQLM imported successfully - version:", aqlm.__version__ if hasattr(aqlm, "__version__") else "unknown")
except Exception as e:
    print(f"Error importing AQLM: {e}")
    print("Will fallback to 4-bit quantization using BitsAndBytes")

# Define memory cleanup function
def cleanup_memory():
    """Clean up GPU memory to avoid fragmentation."""
    print("Cleaning up memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
# Define resource monitoring function
def monitor_resources():
    """Monitor system and GPU resources."""
    # Monitor CPU and RAM
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"CPU memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # Monitor GPU if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            if hasattr(torch.cuda, 'memory_allocated'):
                print(f"GPU {i} ({torch.cuda.get_device_name(i)})")
                print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
                print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
                if hasattr(torch.cuda, 'memory_stats'):
                    stats = torch.cuda.memory_stats(i)
                    if 'active_bytes.all.current' in stats:
                        print(f"  Active memory: {stats['active_bytes.all.current'] / (1024**3):.2f} GB")
                    if 'reserved_bytes.all.current' in stats:
                        print(f"  Reserved memory: {stats['reserved_bytes.all.current'] / (1024**3):.2f} GB")


# %%
# Configure device (GPU or CPU)
if torch.cuda.is_available():
    # Set up for distributed training on multiple GPUs
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Enable multi-GPU support for T4 x2
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        # For distributed training, we'll use device_map="auto" when loading the model
        print("Multi-GPU training enabled")
        
        # Additional memory management for multi-GPU setup
        torch.cuda.empty_cache()
        # Set memory allocation strategy to reduce fragmentation
        if hasattr(torch.cuda, 'memory_stats'):
            print("Initial GPU memory allocated:", torch.cuda.memory_allocated(0) / (1024**3), "GB")
else:
    device = torch.device('cpu')
    print("Using CPU - Note: Training will be much slower on CPU")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %%
# Dataset configuration - using the same dataset as the original notebook
DATASET_ID = "mvasiliniuc/iva-swift-codeint"

# Model configuration - using Phi-3-mini-128k-instruct
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
MAX_LENGTH = 4096  # Phi-3 can handle long sequences natively
BATCH_SIZE = 2  # Reduced batch size for multi-GPU training (each GPU will process this batch size)
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
GRADIENT_ACCUMULATION_STEPS = 4  # Reduced since we're using 2 GPUs

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
print(f"Effective batch size: {BATCH_SIZE * (2 if torch.cuda.device_count() > 1 else 1) * GRADIENT_ACCUMULATION_STEPS}")
print(f"LoRA rank: {LORA_R}")


# %%
# SECTION 2: DATA PREPARATION - Load and prepare the dataset
print("\n" + "="*80)
print("SECTION 2: DATASET PREPARATION")
print("="*80)

print("📊 Loading and preparing the dataset...")

# Function to load dataset with retry logic
def load_dataset_with_retry(dataset_id, max_retries=3, retry_delay=5):
    """Load a dataset with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Loading dataset (attempt {attempt+1}/{max_retries})...")
            data = load_dataset(dataset_id, trust_remote_code=True)
            print(f"✓ Dataset loaded successfully with {len(data['train'])} examples")
            return data
        except Exception as e:
            print(f"⚠️ Error loading dataset (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ Maximum retries reached. Could not load dataset.")
                raise

# Load the dataset with retry logic
try:
    print(f"📥 Loading dataset: {DATASET_ID}")
    data = load_dataset_with_retry(DATASET_ID)
    print("Dataset structure:")
    print(data)
    
    # If in debug mode, take a small sample of the dataset
    if DEBUG_MODE and 'train' in data:
        print(f"🔍 DEBUG MODE: Sampling {DEBUG_SAMPLE_SIZE} examples from dataset")
        # Take a stratified sample if possible
        data['train'] = data['train'].shuffle(seed=42).select(range(min(DEBUG_SAMPLE_SIZE, len(data['train']))))
        print(f"✓ Reduced dataset size: {len(data['train'])} examples")
    
    # Update execution status
    if 'update_status' in globals():
        update_status("data_loaded")
        
except Exception as e:
    print(f"❌ Fatal error loading dataset: {e}")
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
# FIXED: Tokenize the instruction data with proper handling of padding and truncation
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
# Set up training arguments with optimized settings for multi-GPU training
try:
    # Create output directory if it doesn't exist
    os.makedirs("./phi3_swift_model", exist_ok=True)
    
    # Configure training arguments with distributed training settings
    training_args = TrainingArguments(
        output_dir="./phi3_swift_model",
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
        fp16=True,  # Use mixed precision training
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        # Distributed training parameters
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),  # For distributed training
        ddp_find_unused_parameters=False,  # Optimize DDP
        dataloader_num_workers=4,  # Parallelize data loading
        report_to="none",  # Disable reporting to avoid extra overhead
    )
    
    print(f"Training arguments configured for {'multi-GPU' if torch.cuda.device_count() > 1 else 'single-GPU'} training")
    print(f"Using gradient checkpointing: {training_args.gradient_checkpointing}")
    print(f"Using mixed precision: {training_args.fp16}")
    print(f"Local rank: {training_args.local_rank}")
    
except Exception as e:
    print(f"Error setting up training arguments: {e}")
    raise

# %%
# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)


# %%
# FIXED: Create a custom data collator that properly handles the data
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
# SECTION 3: MODEL INITIALIZATION - Load and quantize the model
print("\n" + "="*80)
print("SECTION 3: MODEL INITIALIZATION")
print("="*80)

print("🤖 Initializing model with quantization...")

# Check if required dependencies are available when running in Jupyter (non-sequential mode)
if not IS_KAGGLE:
    required_vars = ['tokenizer', 'tokenized_train', 'tokenized_val', 'LORA_R', 'LORA_ALPHA', 'LORA_DROPOUT']
    missing_vars = [var for var in required_vars if var not in globals()]
    if missing_vars:
        print(f"⚠️ WARNING: Some required variables are not defined: {', '.join(missing_vars)}")
        print("When running in Jupyter, make sure all previous data preparation cells were executed.")
        print("Proceeding anyway as this might be running in sequential mode...")

# Create a flag to track which quantization method we're using
USING_AQLM = False
QUANT_BITS = 2  # Default to 2-bit quantization

print(f"📥 Loading {MODEL_NAME} with {QUANT_BITS}-bit quantization...")

try:
    # First check if AQLM is available
    if 'aqlm' in globals() or 'aqlm' in locals():
        # Use AQLM's correct approach for 2-bit quantization
        print(f"Using AQLM for {QUANT_BITS}-bit quantization...")
        
        # First load the model normally - we'll apply AQLM quantization after
        print(f"Loading base model {MODEL_NAME}...")
        
        # For GPU or CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            use_cache=False,  # Disable KV cache during training for better memory efficiency
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        )
        
        # Now apply AQLM 2-bit quantization to the model
        print(f"Applying AQLM {QUANT_BITS}-bit quantization...")
        
        try:
            # Get the AQLM quantizer - try different possible module paths
            try:
                # Try different module paths where quantize function might be
                if hasattr(aqlm, 'quantize'):
                    quantize_fn = aqlm.quantize
                elif hasattr(aqlm, 'quantization') and hasattr(aqlm.quantization, 'quantize'):
                    quantize_fn = aqlm.quantization.quantize
                else:
                    # Try to discover the correct module
                    for module_name in dir(aqlm):
                        module = getattr(aqlm, module_name)
                        if hasattr(module, 'quantize'):
                            quantize_fn = module.quantize
                            print(f"Found quantize function in aqlm.{module_name}")
                            break
                    else:
                        raise ImportError("Could not find quantize function in AQLM modules")
                
                # Apply quantization to the model
                model = quantize_fn(
                    model, 
                    bits=QUANT_BITS, 
                    lora_rank=LORA_R,  # Use the same rank as we'll use for LoRA
                )
                USING_AQLM = True
                print(f"Successfully applied AQLM {QUANT_BITS}-bit quantization")
            
            except (ImportError, AttributeError) as e:
                print(f"Error using AQLM quantization: {e}")
                print("Trying alternative AQLM API...")
                
                # Try an alternative approach with explicit package imports
                from aqlm import quantize
                model = quantize(
                    model,
                    bits=QUANT_BITS,
                    lora_rank=LORA_R
                )
                USING_AQLM = True
                print(f"Successfully applied AQLM {QUANT_BITS}-bit quantization using direct import")
                
        except Exception as quant_error:
            print(f"AQLM {QUANT_BITS}-bit quantization failed: {quant_error}")
            
            # If 2-bit failed, try 4-bit with AQLM before falling back to BitsAndBytes
            if QUANT_BITS == 2:
                print("Trying AQLM with 4-bit quantization instead...")
                QUANT_BITS = 4
                try:
                    from aqlm import quantize
                    model = quantize(
                        model,
                        bits=QUANT_BITS,
                        lora_rank=LORA_R
                    )
                    USING_AQLM = True
                    print(f"Successfully applied AQLM {QUANT_BITS}-bit quantization")
                except Exception as e:
                    print(f"AQLM {QUANT_BITS}-bit quantization also failed: {e}")
                    raise  # Let it fall through to BitsAndBytes fallback
            else:
                raise  # Let it fall through to BitsAndBytes fallback
    else:
        raise ImportError("AQLM not available")
        
except Exception as e:
    # Fallback to using BitsAndBytes for 4-bit quantization
    print(f"Falling back to BitsAndBytes 4-bit quantization: {e}")
    QUANT_BITS = 4
    USING_AQLM = False
    
    # Configure BitsAndBytes for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model for GPU or CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False
    )
    print("Successfully loaded model with BitsAndBytes 4-bit quantization")

# Configure LoRA for fine-tuning (same for both quantization methods)
print("Setting up LoRA fine-tuning...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Prepare the model for training with LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Print information about the quantized model
quant_method = "AQLM" if USING_AQLM else "BitsAndBytes"
print(f"✅ Model loaded and configured with {QUANT_BITS}-bit {quant_method} quantization and LoRA (rank={LORA_R})")
print(f"Model architecture: {model.__class__.__name__}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

# Update execution status
if 'update_status' in globals():
    update_status("model_initialized")

# %%
# SECTION 4: TRAINER SETUP - Configure the training parameters
print("\n" + "="*80)
print("SECTION 4: TRAINER SETUP")
print("="*80)

print("🔧 Creating trainer and configuring training parameters...")

# Verify dependencies when running in Jupyter mode
if not IS_KAGGLE:
    required_vars = ['model', 'training_args', 'tokenized_train', 'tokenized_val', 
                     'tokenizer', 'data_collator', 'early_stopping_callback']
    missing_vars = [var for var in required_vars if var not in globals()]
    if missing_vars:
        print(f"⚠️ WARNING: Missing required variables: {', '.join(missing_vars)}")
        print("When running in Jupyter, make sure to run all previous cells first.")
        print("Proceeding anyway as this might be running in sequential mode...")

# Create trainer for GPU/CPU
print("Setting up trainer...")
    
# Standard trainer for GPU/CPU
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback]
)

# Verify model device
model_device = next(model.parameters()).device
device_type = "GPU" if torch.cuda.is_available() else "CPU"
print(f"Model is on {device_type} device: {model_device}")

print("✅ Trainer setup complete")

# Update execution status
if 'update_status' in globals():
    update_status("trainer_created")
    
print("Ready to start training...")


# %%
# SECTION 5: TRAINING - Train the model on the dataset
print("\n" + "="*80)
print("SECTION 5: TRAINING PROCESS")
print("="*80)

# Function to monitor system resources during training
def monitor_resources():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    print(f"\nSystem Resources:")
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Process Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"System Memory: {mem.percent}% used, {mem.available / 1024 / 1024:.2f} MB available")
    
    # Show memory metrics for the active device
    if torch.cuda.is_available():
        # GPU memory monitoring
        num_gpus = torch.cuda.device_count()
        print(f"\nGPU Memory Usage ({num_gpus} GPUs detected):")
        
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            properties = torch.cuda.get_device_properties(i)
            total_memory = properties.total_memory / (1024**3)
            free_memory = total_memory - allocated
            
            print(f"  GPU {i} ({properties.name}):")
            print(f"    Total memory: {total_memory:.2f} GB")
            print(f"    Allocated: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
            print(f"    Free: {free_memory:.2f} GB ({free_memory/total_memory*100:.1f}%)")
    
    print("")  # Add blank line for readability

# Verify trainer is initialized when running in Jupyter mode
if not IS_KAGGLE and ('trainer' not in globals() or trainer is None):
    print("⚠️ ERROR: Trainer not initialized. Please run the trainer setup cell first.")
    if 'model' not in globals():
        print("⚠️ ERROR: Model not initialized. Please run the model initialization cell first.")
    raise RuntimeError("Required components not initialized. Please run previous cells first.")

print("🚀 Starting training process...")
print("This will take some time. Training progress will be displayed below.")

# Run training with enhanced memory monitoring for multi-GPU setup
try:
    # Monitor resources before training
    print("Resources before training:")
    monitor_resources()
    
    # Additional memory cleanup before training
    cleanup_memory()
    
    # Set hardware-specific optimizations
    if TPU_AVAILABLE:
        print("Configuring PyTorch for TPU training...")
        # TPU-specific optimizations
        # TPUs work best with bfloat16 precision
        print("TPU training starting with bfloat16 precision...")
        
    elif torch.cuda.device_count() > 1:
        print("Configuring PyTorch for multi-GPU training...")
        # Enable TF32 precision for faster training (on Ampere GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve some memory for system
    
    # Start training with a timeout
    start_time = time.time()
    
    # Run training with device-specific approach
    if TPU_AVAILABLE:
        print("\n🚀 Starting TPU training...")
        train_result = trainer.train()
        
        # Additional TPU verification during training
        print("\nTPU Training Statistics:")
        if hasattr(xm, 'get_memory_info'):
            mem_info = xm.get_memory_info(device)
            print(f"TPU Memory - Free: {mem_info['kb_free']/1024:.2f} MB, Total: {mem_info['kb_total']/1024:.2f} MB")
        print(f"TPU Device: {xm.get_device_type()}")
    else:
        # Standard training for GPU/CPU
        print("\n🚀 Starting training on {device}...")
        train_result = trainer.train()
    
    # Monitor resources after training
    print("Resources after training:")
    monitor_resources()
    
    # Print training results
    print(f"Training completed in {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Training loss: {train_result.metrics['train_loss']:.4f}")
    
    # Save the model with appropriate method based on quantization used
    print("\n💾 Saving trained model...")
    
    # Ensure proper model saving based on the hardware
    if TPU_AVAILABLE:
        # For TPU, ensure we're on CPU before saving to avoid XLA tensor issues
        print("Moving model to CPU before saving (TPU compatibility)...")
        # Synchronize TPU operations before saving
        if hasattr(xm, 'rendezvous'):
            xm.rendezvous("save_checkpoint")
        
    trainer.save_model("./phi3_swift_model")
    
    # Determine the quantization method for display
    quant_method = "AQLM" if USING_AQLM else "BitsAndBytes"
    print(f"✅ Model saved to ./phi3_swift_model ({QUANT_BITS}-bit {quant_method} quantized)")
    print(f"   Trained on: {'TPU' if TPU_AVAILABLE else 'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Save model configuration details
    with open("./phi3_swift_model/quantization_config.json", "w") as f:
        config_data = {
            "quantization_method": quant_method,
            "bits": QUANT_BITS,
            "lora_rank": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "original_model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "training_dataset": DATASET_ID,
            "training_date": time.strftime("%Y-%m-%d")
        }
        json.dump(config_data, f, indent=2)
        print("✅ Model configuration saved")
    
    # Create appropriate loading instructions based on quantization method
    if USING_AQLM:
        loading_code = """```python
from aqlm import quantize
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./phi3_swift_model")

# Load the base model first (to apply quantization)
base_model = AutoModelForCausalLM.from_pretrained("./phi3_swift_model")

# Apply AQLM quantization
model = quantize(base_model, bits=QUANT_BITS, lora_rank=LORA_R)
```"""
    else:
        loading_code = """```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./phi3_swift_model")

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained(
    "./phi3_swift_model",
    quantization_config=bnb_config,
    device_map="auto"
)
```"""
        
    # Also save a README with information about the quantization
    print("📝 Creating model documentation...")
    with open("./phi3_swift_model/README.md", "w") as f:
        f.write(f"""# Phi-3-mini Quantized Model for Swift

This model is a {QUANT_BITS}-bit quantized version of `{MODEL_NAME}` trained for Swift programming.

## Quantization Details
- Method: {quant_method}
- Bits: {QUANT_BITS} 
- Training dataset: {DATASET_ID}
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- LoRA rank: {LORA_R}
- LoRA alpha: {LORA_ALPHA}
- Training date: {time.strftime("%Y-%m-%d")}

## Usage

To load this model:

{loading_code}

This quantized model reduces memory usage significantly while maintaining most of the capabilities of the original model.
""")
        print("✅ Model documentation created")
    
    # Update execution status
    if 'update_status' in globals():
        update_status("training_complete")
    
    # Clean up memory
    print("🧹 Cleaning up memory...")
    cleanup_memory()
    print("✅ Training complete!")
    
except Exception as e:
    print(f"❌ Error during training: {e}")
    
    # Print stack trace for debugging
    import traceback
    traceback.print_exc()
    
    # Monitor resources after error
    print("Resources after error:")
    monitor_resources()
    
    # Update status to indicate failure
    if 'update_status' in globals():
        EXECUTION_STATUS["training_error"] = True
        print("Training failed. Please check the error message above.")
    
    raise

# %%
# SECTION 6: TESTING - Evaluate the trained model
print("\n" + "="*80)
print("SECTION 6: MODEL TESTING")
print("="*80)

# Verify required components are available
if not IS_KAGGLE:
    required_test_vars = ['model', 'tokenizer', 'device', 'QUANT_BITS', 'quant_method']
    missing_vars = [var for var in required_test_vars if var not in globals()]
    if missing_vars:
        print(f"⚠️ WARNING: Missing required variables: {', '.join(missing_vars)}")
        print("When running in Jupyter, make sure you've completed the training process first.")
        print("Proceeding anyway as this might be running in sequential mode...")

try:
    print(f"🧪 Testing the {QUANT_BITS}-bit {quant_method} quantized model with Swift code examples...")
    print(f"This will generate responses to evaluate the model's capabilities.")
    
    # For testing, we use the model we already have loaded
    test_model = model
    
    # Function to generate responses for test examples
    def generate_response(prompt):
        print(f"Generating response for: {prompt.split('<|assistant|>')[0].split('<|user|>')[-1].strip()[:50]}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            # Generate with the quantized model
            outputs = test_model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
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
        print(f"\n📝 Test {i+1}/{len(test_prompts)}:\n{'-'*40}")
        print(f"Prompt: {prompt.split('<|assistant|>')[0].replace('<|user|>', '')}")
        response = generate_response(prompt)
        print(f"\nResponse:\n{response}\n")
        
        # Add a small delay for better readability in logs
        time.sleep(0.5)
    
    print("\n✅ Testing complete! If the responses look good, your model has been trained successfully.")
    print("If you're not satisfied with the quality, you might want to train for more epochs or adjust the training parameters.")
    
    # Update execution status
    if 'update_status' in globals():
        update_status("testing_complete")
        
except Exception as e:
    print(f"❌ Error during testing: {e}")
    print("Detailed error information:")
    import traceback
    traceback.print_exc()
    
    # Update status to indicate testing error
    if 'update_status' in globals():
        EXECUTION_STATUS["testing_error"] = True

# %%
# SECTION 7: EXECUTION SUMMARY
print("\n" + "="*80)
print("SECTION 7: EXECUTION SUMMARY")
print("="*80)

# Print final execution status
if 'EXECUTION_STATUS' in globals():
    print("\n📊 Execution Status Summary:")
    for stage, status in EXECUTION_STATUS.items():
        if 'error' not in stage:  # Skip error flags in the summary view
            icon = "✅" if status else "❌"
            print(f"{icon} {stage.replace('_', ' ').title()}")
    
    # Check if we completed successfully
    core_stages = ['setup_complete', 'data_loaded', 'model_initialized', 
                  'trainer_created', 'training_complete', 'testing_complete']
    success = all(EXECUTION_STATUS.get(stage, False) for stage in core_stages)
    
    if success:
        print("\n🎉 SUCCESS: Complete training pipeline executed successfully!")
    else:
        print("\n⚠️ INCOMPLETE: Some stages of the pipeline did not complete.")
        # Find the first incomplete stage
        for stage in core_stages:
            if not EXECUTION_STATUS.get(stage, False):
                print(f"First incomplete stage: {stage.replace('_', ' ').title()}")
                break

print("\n📋 Final Summary:")
print(f"- Model: {MODEL_NAME}")
print(f"- Quantization: {QUANT_BITS}-bit {quant_method if 'quant_method' in globals() else 'quantization'}")
print(f"- Dataset: {DATASET_ID}")
print(f"- Saved model location: ./phi3_swift_model")
print(f"- Status: {'Successfully trained and tested' if 'success' in locals() and success else 'Incomplete training process'}")

print("\n🚀 Next Steps:")
print("1. Use your trained model for Swift programming tasks")
print("2. Deploy the model to your application")
print("3. Continue fine-tuning with more data if needed")
print("4. Experiment with different quantization settings")

print("\n" + "="*80)
print("Thank you for using the Phi-3 training pipeline!")
print("="*80)
