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
# Install required libraries with specific versions to ensure compatibility
!pip install -q -U numpy
!pip install -q torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
!pip install -q 'torch_xla[tpu]==2.1.0' -f https://storage.googleapis.com/libtpu-releases/index.html
!pip install -q transformers==4.38.1 datasets==2.16.1 evaluate==0.4.1 scikit-learn tqdm dropbox requests accelerate==0.28.0 peft==0.7.1

# Set environment variables for TPU
import os
# Check for TPU environment and set appropriate variables
if "COLAB_TPU_ADDR" in os.environ:
    print(f"Setting up Colab TPU: {os.environ['COLAB_TPU_ADDR']}")
    os.environ["XLA_USE_BF16"] = "1"  # Enable bfloat16 for better performance
elif "TPU_NAME" in os.environ:  
    print(f"Using Cloud TPU: {os.environ['TPU_NAME']}")
else:
    print("No TPU environment detected, setting to local")
    os.environ["TPU_NAME"] = "local"

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
import sys
import logging
import traceback
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TPU detection and imports with error handling
TPU_AVAILABLE = False
try:
    # Try to import TPU-specific libraries
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.distributed.parallel_loader import ParallelLoader
    import torch_xla.debug.metrics as met
    
    # Verify TPU is actually available
    if torch_xla.xla_device_hw(devkind='TPU') is not None:
        TPU_AVAILABLE = True
        logger.info("TPU detected and PyTorch XLA successfully imported")
    else:
        logger.warning("PyTorch XLA imported but no TPU devices detected")
        
except ImportError as e:
    logger.warning(f"Could not import PyTorch XLA. TPU will not be available. Error: {e}")
    # Create placeholder functions/objects for graceful fallback
    class XMPlaceholder:
        @staticmethod
        def xla_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        @staticmethod
        def mark_step(): pass
        @staticmethod
        def xrt_world_size(): return 1
        @staticmethod
        def get_ordinal(): return 0
        @staticmethod
        def get_local_ordinal(): return 0
        @staticmethod
        def get_device_type(): return 'CPU' if not torch.cuda.is_available() else 'GPU'
        @staticmethod
        def is_master_ordinal(): return True
        @staticmethod
        def master_print(msg): print(msg)
        @staticmethod
        def rendezvous(tag): pass
        
    class MetPlaceholder:
        @staticmethod
        def metrics_report(): return "TPU metrics not available"
    
    # Set up fallback placeholders
    xm = XMPlaceholder()
    met = MetPlaceholder()
    TPU_AVAILABLE = False

# Define memory cleanup function for TPU
def cleanup_memory():
    """Clean up memory to avoid fragmentation."""
    logger.info("Cleaning up memory...")
    
    # Force garbage collection
    gc.collect()
    
    # Empty CUDA cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")
    
    # Synchronize TPU operations if TPU is available
    if TPU_AVAILABLE and xm.xrt_world_size() > 0:
        try:
            xm.mark_step()
            logger.info("TPU memory synchronized")
        except Exception as e:
            logger.warning(f"Error during TPU synchronization: {e}")
        
# Define resource monitoring function for TPU and GPU
def monitor_resources():
    """Monitor system and accelerator (TPU/GPU) resources."""
    # Monitor CPU and RAM
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    mem = psutil.virtual_memory()
    
    logger.info(f"CPU memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    logger.info(f"System memory: {mem.percent}% used, {mem.available / 1024 / 1024:.2f} MB available")
    
    # Monitor GPU if available
    if torch.cuda.is_available() and not TPU_AVAILABLE:
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}): "
                  f"{torch.cuda.memory_allocated(i) / 1024 / 1024:.2f} MB allocated, "
                  f"{torch.cuda.memory_reserved(i) / 1024 / 1024:.2f} MB reserved")
    
    # Monitor TPU if available
    if TPU_AVAILABLE and xm.xrt_world_size() > 0:
        try:
            logger.info(f"TPU device: {xm.get_device_type()}")
            logger.info(f"Number of TPU devices: {xm.xrt_world_size()}")
            
            # Get TPU memory metrics
            tpu_metrics = met.metrics_report()
            
            # Extract and log TPU memory-related metrics
            metrics_str = str(tpu_metrics)
            memory_metrics = [m for m in metrics_str.split('\n') 
                             if any(word in m.lower() for word in ['mem', 'memory', 'ram'])]
            
            if memory_metrics:
                logger.info("TPU Memory Metrics:")
                for metric in memory_metrics:
                    logger.info(f"  {metric}")
            
            # Check for potential issues or errors in metrics
            error_indicators = ['oom', 'error', 'out of memory']
            errors = [m for m in metrics_str.split('\n') 
                     if any(indicator in m.lower() for indicator in error_indicators)]
            
            if errors:
                logger.warning("Potential TPU issues detected:")
                for error in errors:
                    logger.warning(f"  {error}")
                    
        except Exception as e:
            logger.error(f"Error monitoring TPU resources: {e}")
            logger.debug(traceback.format_exc())


# %%
# Configure device (TPU, GPU, or CPU)
def setup_device():
    """Set up the appropriate device based on what's available."""
    if TPU_AVAILABLE:
        try:
            device = xm.xla_device()
            logger.info(f"Using TPU device: {device}")
            logger.info(f"Number of TPU devices: {xm.xrt_world_size()}")
            logger.info(f"TPU type: {xm.get_device_type()}")
            
            # Print TPU configuration details
            logger.info("TPU Configuration:")
            logger.info(f"  TPU cores: {xm.xrt_world_size()}")
            logger.info(f"  Local ordinal: {xm.get_local_ordinal()}")
            logger.info(f"  Global ordinal: {xm.get_ordinal()}")
            
            # Verify TPU is actually working with a small tensor operation
            test_tensor = torch.randn(2, 2)
            test_tensor = test_tensor.to(device)
            _ = test_tensor + test_tensor  # Simple operation to verify device works
            xm.mark_step()  # Sync TPU
            logger.info("✅ TPU verification passed")
            
            return device, "tpu"
            
        except Exception as e:
            logger.warning(f"TPU initialization failed: {e}")
            logger.warning("Falling back to GPU or CPU")
            TPU_AVAILABLE = False
    
    # Fall back to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        return device, "gpu"
    
    # Fall back to CPU as last resort
    device = torch.device("cpu")
    logger.info("Using CPU device")
    return device, "cpu"

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
# Setup device
device, device_type = setup_device()
logger.info(f"Device setup complete. Using: {device_type.upper()}")

# Monitor initial resource usage
monitor_resources()

# %%
# Dataset configuration - using the same dataset as the original notebook
DATASET_ID = "mvasiliniuc/iva-swift-codeint"

# Model configuration - using Phi-3-mini-128k-instruct
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"

# Set device-specific parameters based on what's available
if device_type == "tpu":
    MAX_LENGTH = 2048  # Reduced from 4096 to save memory on TPU
    BATCH_SIZE = 1  # Reduced batch size to avoid OOM errors on TPU
    GRADIENT_ACCUMULATION_STEPS = 8  # Increased to compensate for smaller batch size
    # Use bfloat16 on TPU for better performance/memory usage
    MODEL_DTYPE = torch.bfloat16
elif device_type == "gpu":
    # GPU configuration - can handle more if GPU has enough VRAM
    if torch.cuda.get_device_properties(0).total_memory > 16e9:  # >16GB VRAM
        MAX_LENGTH = 2048
        BATCH_SIZE = 2
        GRADIENT_ACCUMULATION_STEPS = 4
    else:  # Smaller GPU
        MAX_LENGTH = 1024
        BATCH_SIZE = 1
        GRADIENT_ACCUMULATION_STEPS = 8
    # Use float16 on GPU for better performance
    MODEL_DTYPE = torch.float16
else:  # CPU fallback
    MAX_LENGTH = 512  # Much smaller for CPU
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    MODEL_DTYPE = torch.float32  # Full precision on CPU

# Common parameters
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03

# LoRA configuration - helps with memory efficiency on all devices
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Debug mode for testing with smaller dataset
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 100

# Calculate effective batch size
if device_type == "tpu":
    num_devices = xm.xrt_world_size()
elif device_type == "gpu":
    num_devices = torch.cuda.device_count()
else:
    num_devices = 1

EFFECTIVE_BATCH_SIZE = BATCH_SIZE * num_devices * GRADIENT_ACCUMULATION_STEPS

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Max sequence length: {MAX_LENGTH}")
logger.info(f"Batch size: {BATCH_SIZE} per device")
logger.info(f"Number of devices: {num_devices}")
logger.info(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")
logger.info(f"Using model dtype: {MODEL_DTYPE}")
logger.info(f"LoRA rank: {LORA_R}")


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
# Set up training arguments based on device type
try:
    # Create output directory if it doesn't exist
    model_output_dir = f"./phi3_{device_type}_swift_model"
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Define training argument dict with common parameters
    training_args_dict = {
        "output_dir": model_output_dir,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "logging_dir": f"./logs/{device_type}",
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 2,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "load_best_model_at_end": True,
        "report_to": "none",  # Disable reporting to reduce overhead
        "gradient_checkpointing": True,  # Enable gradient checkpointing to save memory
    }
    
    # Add device-specific settings
    if device_type == "tpu":
        # TPU-specific settings
        training_args_dict.update({
            "tpu_num_cores": 8,  # Specify the number of TPU cores
            "dataloader_num_workers": 2,
            "dataloader_pin_memory": False,
            "fp16": False,  # TPUs prefer bfloat16 over fp16
            "bf16": True,  # Enable bfloat16 for TPUs
        })
    elif device_type == "gpu":
        # GPU-specific settings
        training_args_dict.update({
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "fp16": True,  # Enable mixed precision training on GPUs
            "bf16": False,
            "half_precision_backend": "auto",
        })
    else:  # CPU
        # CPU-specific settings
        training_args_dict.update({
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "fp16": False,
            "bf16": False,
        })
    
    # Create training arguments
    training_args = TrainingArguments(**training_args_dict)
    
    logger.info(f"Training arguments configured for {device_type.upper()} training")
    logger.info(f"Using gradient checkpointing: {training_args.gradient_checkpointing}")
    
    if device_type == "tpu":
        logger.info(f"TPU cores: {training_args.tpu_num_cores}")
    
except Exception as e:
    logger.error(f"Error setting up training arguments: {e}")
    logger.error(traceback.format_exc())
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

# Load and prepare the model with device-specific optimizations
try:
    logger.info(f"Loading {MODEL_NAME} for {device_type.upper()}...")
    
    # Clean up memory before model loading
    cleanup_memory()
    
    # Define model loading parameters based on device
    model_loading_kwargs = {
        "trust_remote_code": True,
        "use_cache": False,  # Disable KV cache during training for better memory efficiency
    }
    
    # Set device-specific model loading parameters
    if device_type == "tpu":
        model_loading_kwargs.update({
            "torch_dtype": MODEL_DTYPE,  # bfloat16 for TPU
            "low_cpu_mem_usage": True,
            "load_in_8bit": False,
        })
    elif device_type == "gpu":
        model_loading_kwargs.update({
            "torch_dtype": MODEL_DTYPE,  # float16 for GPU
            "low_cpu_mem_usage": True,
            # For GPUs with limited memory, add device_map for better memory efficiency
            "device_map": "auto" if torch.cuda.get_device_properties(0).total_memory < 24e9 else None,
        })
    else:  # cpu
        model_loading_kwargs.update({
            "torch_dtype": MODEL_DTYPE,  # float32 for CPU
            "low_cpu_mem_usage": True,
        })
    
    # Load the model with the specified parameters
    logger.info(f"Loading model with parameters: {model_loading_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_loading_kwargs)
    
    # For TPU or Standard GPU, we need to explicitly move model to device
    if device_type in ["tpu", "gpu"] and "device_map" not in model_loading_kwargs:
        logger.info(f"Moving model to {device}...")
        model = model.to(device)
    
    # Verify model is loaded on the correct device
    model_device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {model_device}")
    
    # Configure LoRA for efficient fine-tuning
    logger.info("Setting up LoRA fine-tuning...")
    
    # Get label names from category names
    label_names = [category_names[i] for i in range(len(category_names))]
    
    # Configure optimized target modules based on model architecture
    # Identify what modules are in the model to avoid errors
    model_modules = [name for name, _ in model.named_modules()]
    
    # Default set of modules to target
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Ensure we only target modules that actually exist in the model
    target_modules = [name for name in target_modules if any(name in module for module in model_modules)]
    
    logger.info(f"Targeting LoRA modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        label_names=label_names
    )
    
    # Apply LoRA to model
    logger.info("Applying LoRA adapter to model...")
    model = get_peft_model(model, lora_config)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    logger.info(f"Model loaded and configured with LoRA (rank={LORA_R})")
    logger.info(f"Model architecture: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params:.2f}M")
    logger.info(f"Parameter efficiency: {trainable_params/total_params*100:.2f}%")
    
    # Monitor device status after model loading
    monitor_resources()
    
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())
    
    # Try fallback to CPU if model loading on accelerator failed
    if device_type != "cpu":
        logger.warning(f"Attempting fallback to CPU...")
        try:
            device = torch.device("cpu")
            device_type = "cpu"
            MODEL_DTYPE = torch.float32
            
            # Adjust training parameters for CPU
            MAX_LENGTH = 512
            BATCH_SIZE = 1
            GRADIENT_ACCUMULATION_STEPS = 16
            
            # Load model on CPU
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                use_cache=False
            )
            
            logger.info("Successfully loaded model on CPU as fallback")
        except Exception as cpu_e:
            logger.error(f"CPU fallback also failed: {cpu_e}")
            raise
    else:
        raise

# %%
# Create a device-optimized data collator for efficient training
class DeviceOptimizedDataCollator(DataCollatorForLanguageModeling):
    """Data collator optimized for different device types (TPU, GPU, CPU)"""
    
    def __init__(self, tokenizer, mlm=False, device_type="tpu"):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.device_type = device_type
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        logger.info(f"Initialized {device_type.upper()}-optimized data collator")
    
    def __call__(self, features):
        # Ensure all features have the same keys
        if not all(k in features[0] for k in ["input_ids", "attention_mask", "labels"]):
            logger.warning("Some features are missing required keys in batch")
            # Attempt to fix by ensuring all features have the required keys
            for feature in features:
                for key in ["input_ids", "attention_mask", "labels"]:
                    if key not in feature:
                        if key == "labels":
                            feature[key] = feature.get("input_ids", []).copy()
                        else:
                            # Create empty tensor with right dtype
                            feature[key] = torch.tensor([], dtype=torch.long)
        
        try:
            # Create batch dictionary
            batch = {
                "input_ids": torch.stack([f["input_ids"] for f in features]),
                "attention_mask": torch.stack([f["attention_mask"] for f in features]),
                "labels": torch.stack([f["labels"] for f in features])
            }
            
            # Device-specific optimizations
            if self.device_type == "tpu":
                # For TPU, we need to ensure tensors have proper shapes and dtypes
                # XLA prefers static shapes
                for key in batch:
                    # Ensure int64 dtype for TPU
                    if batch[key].dtype != torch.int64:
                        batch[key] = batch[key].to(torch.int64)
            
            return batch
            
        except Exception as e:
            logger.error(f"Error in data collation: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback: Process one example at a time to identify the problematic feature
            logger.warning("Attempting fallback collation...")
            
            # Find the max length in this batch to use for padding
            max_length = max(len(f["input_ids"]) for f in features)
            
            # Manually create batches with careful padding
            input_ids = []
            attention_mask = []
            labels = []
            
            for feature in features:
                # Pad input_ids
                padded_input_ids = feature["input_ids"].tolist() + [self.pad_token_id] * (max_length - len(feature["input_ids"]))
                input_ids.append(torch.tensor(padded_input_ids, dtype=torch.long))
                
                # Pad attention_mask
                padded_attention_mask = feature["attention_mask"].tolist() + [0] * (max_length - len(feature["attention_mask"]))
                attention_mask.append(torch.tensor(padded_attention_mask, dtype=torch.long))
                
                # Pad labels
                padded_labels = feature["labels"].tolist() + [-100] * (max_length - len(feature["labels"]))
                labels.append(torch.tensor(padded_labels, dtype=torch.long))
            
            batch = {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_mask),
                "labels": torch.stack(labels)
            }
            
            logger.info("Fallback collation successful")
            return batch

# Create data collator optimized for the current device
data_collator = DeviceOptimizedDataCollator(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked language modeling
    device_type=device_type
)
logger.info(f"Created {device_type}-optimized data collator")

# %%
# Create device-optimized callbacks based on device type
class DeviceMetricsCallback(transformers.TrainerCallback):
    """Callback to monitor metrics for different device types"""
    
    def __init__(self, device_type):
        self.device_type = device_type
        logger.info(f"Initialized {device_type} metrics callback")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        # Log metrics based on device type
        if self.device_type == "tpu" and TPU_AVAILABLE:
            xm.master_print(f"TPU Step {state.global_step}: {logs}")
            
            # Log TPU metrics periodically
            if state.global_step % 50 == 0:
                try:
                    tpu_metrics = met.metrics_report()
                    metrics_str = str(tpu_metrics)
                    
                    # Extract and print TPU memory metrics
                    memory_metrics = [m for m in metrics_str.split('\n') 
                                     if any(word in m.lower() for word in ['mem', 'memory'])]
                    
                    if memory_metrics:
                        xm.master_print("\nTPU Memory Metrics:")
                        for metric in memory_metrics:
                            xm.master_print(f"  {metric}")
                except Exception as e:
                    logger.warning(f"Error getting TPU metrics: {e}")
                
        elif self.device_type == "gpu":
            if state.global_step % 50 == 0 and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}): "
                         f"{torch.cuda.memory_allocated(i) / 1024 / 1024:.2f} MB allocated, "
                         f"{torch.cuda.memory_reserved(i) / 1024 / 1024:.2f} MB reserved")
                    
    def on_step_end(self, args, state, control, **kwargs):
        # Device-specific syncing and memory management
        if self.device_type == "tpu" and TPU_AVAILABLE:
            # TPU-specific step end operations
            try:
                xm.mark_step()  # Ensure TPU operations complete
                
                # Check if we're hitting memory pressure
                if state.global_step % 100 == 0:
                    metrics = met.metrics_report()
                    metrics_str = str(metrics)
                    
                    # Look for memory pressure indicators
                    if any(warning in metrics_str.lower() for warning in ["oom", "out of memory", "resource exhausted"]):
                        logger.warning("⚠️ TPU memory pressure detected!")
                        # Force garbage collection
                        gc.collect()
                        xm.mark_step()  # Ensure sync after cleanup
            except Exception as e:
                logger.warning(f"Error in TPU step end operations: {e}")
                
        elif self.device_type == "gpu" and torch.cuda.is_available():
            # GPU-specific operations for memory monitoring
            if state.global_step % 100 == 0:
                # Empty cache periodically to reduce fragmentation
                torch.cuda.empty_cache()

# Initialize early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

# Create device metrics callback
device_metrics_callback = DeviceMetricsCallback(device_type)

# Configure callbacks based on device type
callbacks = [early_stopping_callback, device_metrics_callback]

# Create the trainer with device-optimized settings
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks
)

# Verify model device
model_device = next(model.parameters()).device
logger.info(f"Model is on device: {model_device}")

logger.info(f"{device_type.upper()} training setup complete")


# %%
# Function to monitor device resources during training
def monitor_training_resources():
    """Monitor system and accelerator resources during training."""
    logger.info(f"\n{device_type.upper()} Resources:")
    
    # Monitor CPU memory (for host)
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    logger.info(f"CPU Usage: {cpu_percent}%")
    logger.info(f"Process Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    logger.info(f"System Memory: {mem.percent}% used, {mem.available / 1024 / 1024:.2f} MB available")
    
    # Device-specific monitoring
    if device_type == "tpu" and TPU_AVAILABLE:
        try:
            logger.info("\nTPU Metrics:")
            tpu_metrics = met.metrics_report()
            metrics_str = str(tpu_metrics)
            
            # Extract and log important metrics
            metric_categories = {
                "Memory": ['mem', 'memory', 'ram'],
                "Compilation": ['compile', 'xla', 'execution'],
                "Errors": ['error', 'timeout', 'fail', 'oom', 'out of memory']
            }
            
            for category, keywords in metric_categories.items():
                filtered_metrics = [m for m in metrics_str.split('\n') 
                                  if any(word in m.lower() for word in keywords)]
                
                if filtered_metrics:
                    prefix = "⚠️ " if category == "Errors" else ""
                    logger.info(f"\n{prefix}TPU {category} Metrics:")
                    for metric in filtered_metrics:
                        logger.info(f"  {metric}")
        except Exception as e:
            logger.warning(f"Error getting TPU metrics: {e}")
    
    elif device_type == "gpu" and torch.cuda.is_available():
        try:
            logger.info("\nGPU Metrics:")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                mem_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                mem_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                mem_usage_percent = 100 * mem_allocated / mem_reserved if mem_reserved > 0 else 0
                
                logger.info(f"GPU {i} ({gpu_name}):")
                logger.info(f"  Memory Allocated: {mem_allocated:.2f} MB")
                logger.info(f"  Memory Reserved: {mem_reserved:.2f} MB")
                logger.info(f"  Memory Usage: {mem_usage_percent:.2f}%")
                
                # Check for potential memory pressure
                if mem_usage_percent > 90:
                    logger.warning(f"⚠️ High GPU memory usage on GPU {i}: {mem_usage_percent:.2f}%")
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
    
    logger.info("")  # Empty line for readability

# %%
# Create a device-optimized training loop with proper error handling
try:
    logger.info(f"Starting training with {device_type.upper()}-optimized settings...")
    
    # Monitor resources before training
    logger.info("Resources before training:")
    monitor_training_resources()
    
    # Additional cleanup before training
    cleanup_memory()
    
    # Add device-specific guard callback
    class DeviceGuardCallback(transformers.TrainerCallback):
        """Device-specific monitoring and error prevention callback"""
        def __init__(self, device_type):
            self.device_type = device_type
            logger.info(f"Initialized {device_type} guard callback")
        
        def on_step_end(self, args, state, control, **kwargs):
            # Device-specific operations after each step
            if self.device_type == "tpu" and TPU_AVAILABLE:
                try:
                    # Perform a TPU sync after each step
                    xm.mark_step()
                    
                    # Periodically monitor TPU resources
                    if state.global_step % 50 == 0:
                        logger.info(f"TPU status at step {state.global_step}:")
                        metrics = met.metrics_report()
                        
                        # Check for memory issues
                        metrics_str = str(metrics)
                        if any(warning in metrics_str.lower() for warning in ["oom", "out of memory", "resource exhausted"]):
                            logger.warning("⚠️ TPU memory pressure detected!")
                            # Force garbage collection
                            gc.collect()
                            xm.mark_step()  # Ensure sync
                            
                except Exception as e:
                    logger.warning(f"Error in TPU step end handling: {e}")
            
            elif self.device_type == "gpu" and torch.cuda.is_available():
                # Periodically monitor GPU usage
                if state.global_step % 100 == 0 and state.global_step > 0:
                    # Clear cache periodically to prevent fragmentation
                    torch.cuda.empty_cache()
            
            # Do deeper analysis every 200 steps for all device types
            if state.global_step % 200 == 0 and state.global_step > 0:
                monitor_training_resources()
                    
        def on_epoch_end(self, args, state, control, **kwargs):
            # Clean up between epochs
            logger.info(f"Completed epoch {state.epoch}")
            gc.collect()
            
            if self.device_type == "tpu" and TPU_AVAILABLE:
                xm.mark_step()
                logger.info("TPU sync point at epoch end")
            elif self.device_type == "gpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared at epoch end")
    
    # Add our device monitoring callback
    trainer.add_callback(DeviceGuardCallback(device_type))
    
    # Start training with a timeout
    max_training_time = 6 * 60 * 60  # 6 hours max
    start_time = time.time()
    
    # Run training with device-specific error handling
    try:
        # Train the model
        logger.info(f"\nStarting {device_type.upper()} training...")
        train_result = trainer.train()
        
        # Ensure device-specific operations are complete
        if device_type == "tpu" and TPU_AVAILABLE:
            xm.mark_step()
        elif device_type == "gpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
    except Exception as e:
        logger.error(f"\n🛑 Training error on {device_type}: {e}")
        logger.error(traceback.format_exc())
        
        # Try to diagnose the issue
        logger.info("Diagnostic information:")
        monitor_training_resources()
        
        # Clean up and try to restart if possible
        gc.collect()
        if device_type == "tpu" and TPU_AVAILABLE:
            xm.mark_step()
        elif device_type == "gpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # For common memory issues, try reducing parameters and retry
        if any(err in str(e).lower() for err in ["memory", "resource", "cuda out of memory"]):
            logger.warning("\nAttempting recovery with reduced parameters...")
            
            # Reduce sequence length if needed
            old_max_len = MAX_LENGTH
            if MAX_LENGTH > 1024:
                MAX_LENGTH = 1024
            elif MAX_LENGTH > 512:
                MAX_LENGTH = 512
            else:
                MAX_LENGTH = 256
                
            logger.info(f"Reducing sequence length from {old_max_len} to {MAX_LENGTH}")
            
            # Try with smaller batch size and more gradient accumulation
            old_batch_size = training_args.per_device_train_batch_size
            old_grad_accum = training_args.gradient_accumulation_steps
            
            training_args.per_device_train_batch_size = 1
            training_args.gradient_accumulation_steps = old_grad_accum * 2
            
            logger.info(f"Reduced batch size from {old_batch_size} to 1 and increased gradient accumulation "
                  f"from {old_grad_accum} to {training_args.gradient_accumulation_steps}")
            
            # Re-create trainer with updated settings
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=callbacks + [DeviceGuardCallback(device_type)]
            )
            
            # Try again with new settings
            logger.info(f"\nRetrying training with reduced memory settings...")
            train_result = trainer.train()
            
            # Sync operations after recovery training
            if device_type == "tpu" and TPU_AVAILABLE:
                xm.mark_step()
    
    # Monitor resources after training
    logger.info("\nResources after training:")
    monitor_training_resources()
    
    # Print training results
    elapsed_time = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed_time/60:.2f} minutes")
    logger.info(f"Training loss: {train_result.metrics['train_loss']:.4f}")
    
    # Save the model - with device-specific handling
    logger.info("\nSaving model...")
    
    # Define output paths
    model_output_path = f"./phi3_{device_type}_swift_model"
    adapter_output_path = f"./phi3_{device_type}_swift_model_adapter"
    
    if device_type == "tpu" and TPU_AVAILABLE:
        # Ensure TPU operations complete before saving
        xm.mark_step()
        
        # Save on TPU process 0 only to avoid conflicts
        if xm.is_master_ordinal():
            trainer.save_model(model_output_path)
            
            # Save adapter separately for easier loading
            if hasattr(model, "save_pretrained"):
                try:
                    model.save_pretrained(adapter_output_path)
                    logger.info(f"Saved LoRA adapter to {adapter_output_path}")
                except Exception as save_error:
                    logger.error(f"Error saving adapter: {save_error}")
            
            logger.info("Model saved successfully")
        
        # Final TPU synchronization
        xm.rendezvous("training_complete")
    else:
        # For GPU/CPU we can save directly
        trainer.save_model(model_output_path)
        
        # Save adapter separately for easier loading
        if hasattr(model, "save_pretrained"):
            try:
                model.save_pretrained(adapter_output_path)
                logger.info(f"Saved LoRA adapter to {adapter_output_path}")
            except Exception as save_error:
                logger.error(f"Error saving adapter: {save_error}")
        
        logger.info("Model saved successfully")
    
    logger.info(f"{device_type.upper()} training complete")
    
    # Clean up memory
    cleanup_memory()
    
except Exception as e:
    logger.error(f"\n❌ Error during {device_type} training: {e}")
    logger.error(traceback.format_exc())
    
    # Device-specific diagnostics
    if device_type == "tpu" and TPU_AVAILABLE:
        logger.info("\nTPU diagnostic information at error:")
        logger.info(met.metrics_report())
    elif device_type == "gpu" and torch.cuda.is_available():
        logger.info("\nGPU diagnostic information at error:")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / 1024 / 1024:.2f} MB allocated")
    
    # Try to save checkpoint if possible
    try:
        logger.info("\nAttempting to save emergency checkpoint...")
        emergency_path = f"./phi3_{device_type}_swift_model_emergency_checkpoint"
        
        if device_type == "tpu" and TPU_AVAILABLE:
            if xm.is_master_ordinal():
                trainer.save_model(emergency_path)
        else:
            trainer.save_model(emergency_path)
            
        logger.info(f"Emergency checkpoint saved to {emergency_path}")
    except Exception as save_error:
        logger.error(f"Could not save emergency checkpoint: {save_error}")
    
    # Monitor resources after error
    logger.info("Resources after error:")
    monitor_training_resources()
    
    raise

# %%
# Test the model with Swift code examples (device-agnostic approach)
try:
    logger.info(f"Testing the model with Swift code examples on {device_type.upper()}...")
    
    # Function to generate responses for test examples with device-specific handling
    def generate_response(prompt, max_length=200):
        """Generate model response with appropriate device handling"""
        try:
            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to the appropriate device
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            
            # Set generation parameters
            generation_config = {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            # Device-specific handling for generation
            with torch.no_grad():
                # Generate with proper device handling
                if device_type == "tpu" and TPU_AVAILABLE:
                    # TPU-specific generation
                    outputs = model.generate(**inputs, **generation_config)
                    # Ensure TPU operations complete
                    xm.mark_step()
                else:
                    # GPU/CPU generation
                    outputs = model.generate(**inputs, **generation_config)
                
            # Return to host for decoding
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return f"Error: Could not generate response due to: {str(e)}"
    
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
    
    # Generate and print responses with proper error handling
    for i, prompt in enumerate(test_prompts):
        try:
            logger.info(f"\nTest {i+1}:\n{'-'*40}")
            
            # Extract user part of prompt for logging
            user_prompt = prompt.split('<|assistant|>')[0].replace('<|user|>', '')
            logger.info(f"Prompt: {user_prompt}")
            
            # Generate response with device-appropriate handling
            logger.info(f"Generating response on {device_type}...")
            response = generate_response(prompt)
            
            # Log and display the response
            logger.info(f"\nResponse:\n{response}\n")
            
            # Clean up after each test
            cleanup_memory()
            
        except Exception as test_error:
            logger.error(f"Error in test {i+1}: {test_error}")
    
    logger.info(f"\n{device_type.upper()} testing complete")
    
    # Save example responses
    example_output_path = f"./phi3_{device_type}_swift_examples.txt"
    try:
        with open(example_output_path, "w") as f:
            f.write(f"Example outputs from Phi-3 fine-tuned on Swift ({device_type}):\n\n")
            
            for i, prompt in enumerate(test_prompts):
                user_prompt = prompt.split('<|assistant|>')[0].replace('<|user|>', '')
                response = generate_response(prompt, max_length=100)  # Shorter for examples
                
                f.write(f"Example {i+1}:\n")
                f.write(f"Prompt: {user_prompt}\n")
                f.write(f"Response: {response}\n\n")
                f.write("-" * 80 + "\n\n")
                
        logger.info(f"Saved example outputs to {example_output_path}")
    except Exception as save_error:
        logger.error(f"Error saving example outputs: {save_error}")
    
except Exception as e:
    logger.error(f"Error during {device_type} testing: {e}")
    logger.error(traceback.format_exc())
