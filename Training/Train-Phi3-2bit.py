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

# %%
# EXECUTION TRACKING SYSTEM
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

# Flag to begin execution
print("Starting Phi-3 training pipeline...")

# %%
# SECTION 1: SETUP - Install required libraries and configure environment
print("📦 Installing required libraries...")

# First, install bitsandbytes with multi-backend support for CPU offloading during training
print("👉 Installing BitsAndBytes with multi-backend support for CPU offloading...")
!pip uninstall -y bitsandbytes  # Remove any existing installation
!pip install git+https://github.com/TimDettmers/bitsandbytes.git  # Install latest version
!pip install --upgrade transformers datasets evaluate torch scikit-learn tqdm dropbox requests accelerate peft

# Verify bitsandbytes installation for training with offloading
print("👉 Verifying BitsAndBytes installation...")
try:
    import bitsandbytes as bnb
    import torch
    print(f"✓ BitsAndBytes version: {bnb.__version__}")
    
    # Check if compiled with CUDA support (different ways depending on version)
    cuda_available = False
    
    # Method 1: Check COMPILED_WITH_CUDA attribute (older versions)
    if hasattr(bnb, "COMPILED_WITH_CUDA"):
        cuda_available = bnb.COMPILED_WITH_CUDA
        print(f"✓ Compiled with CUDA: {cuda_available}")
    
    # Method 2: Check cuda_specs module (newer versions)
    elif hasattr(bnb, "cuda_specs") and hasattr(bnb.cuda_specs, "CUDA_AVAILABLE"):
        cuda_available = bnb.cuda_specs.CUDA_AVAILABLE
        print(f"✓ CUDA available (from cuda_specs): {cuda_available}")
    
    # Method 3: Check if CUDA is available through torch
    elif hasattr(torch, "cuda") and torch.cuda.is_available():
        cuda_available = True
        print(f"✓ CUDA available through PyTorch: {torch.cuda.is_available()}")
    else:
        print("⚠️ CUDA support not detected in bitsandbytes")
    
    # Check for multi-backend support (different ways depending on version)
    has_multi_backend = False
    
    # Method 1: Check for get_available_modules function
    if "get_available_modules" in dir(bnb):
        has_multi_backend = True
        print("✓ Multi-backend support detected (get_available_modules)")
    
    # Method 2: Check for cuda module with has_cuda_extension
    elif hasattr(bnb, "cuda"):
        # Only check for has_cuda_extension if bnb.cuda exists
        if hasattr(bnb.cuda, "has_cuda_extension"):
            has_multi_backend = True
            print("✓ Multi-backend support detected (cuda.has_cuda_extension)")
    
    # Method 3: Check for diagnostics.cuda module
    elif hasattr(bnb, "diagnostics") and hasattr(bnb.diagnostics, "cuda"):
        print("✓ CUDA diagnostics module detected")
        has_multi_backend = True
    
    if not has_multi_backend:
        print("⚠️ Multi-backend support not detected, but continuing anyway")
    
    print("BitsAndBytes installation looks good enough to proceed!")
    
except ImportError as e:
    print(f"⚠️ BitsAndBytes import failed: {e}")
    print("Installing fallback version...")
    !pip install bitsandbytes

# Set PyTorch memory management environment variables to avoid fragmentation and OOM issues
import os
# Configure CUDA memory allocation for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Explicitly set to use 2 GPUs
# Enable better memory handling with CPU offloading
os.environ["ACCELERATE_USE_CPU_OFFLOAD"] = "1"
os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
# Enable disk offloading if needed
os.environ["ACCELERATE_ENABLE_DISK_OFFLOAD"] = "1"
# Enable training with CPU offloading
os.environ["BNB_OFFLOAD_TRAINING"] = "1"  # Critical for training with CPU offload

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
import sys
import json
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_dataset

# Set environment variables to disable unnecessary features
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

# Skip unnecessary imports in transformers
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Temporarily set offline mode
os.environ["TRANSFORMERS_SKIP_TORCH_VISION_IMPORT"] = "1"  # Skip image-related components

# Import transformers components individually to avoid problematic dependencies
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.utils.quantization_config import BitsAndBytesConfig

# Reset offline mode after imports
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from transformers.trainer_callback import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Import GGUF for model quantization
try:
    try:
        import ctransformers
    except ImportError:
        print("ctransformers package not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "-U", "ctransformers", "--no-cache-dir"])
        import ctransformers

    # Also install llama-cpp-python for additional GGUF compatibility
    try:
        import llama_cpp
    except ImportError:
        print("llama-cpp-python package not found. Installing...")
        subprocess.check_call(["pip", "install", "-U", "llama-cpp-python", "--no-cache-dir"])
        import llama_cpp
    
    print("GGUF libraries imported successfully - ctransformers version:", 
          ctransformers.__version__ if hasattr(ctransformers, "__version__") else "unknown")
    print("llama-cpp-python version:", 
          llama_cpp.__version__ if hasattr(llama_cpp, "__version__") else "unknown")
except Exception as e:
    print(f"Error importing GGUF libraries: {e}")
    print("Will fallback to 4-bit quantization using BitsAndBytes")

# Focus on language model training without unnecessary dependencies

# Define Kaggle-optimized memory cleanup function
def cleanup_memory():
    """
    Clean up GPU memory with aggressive management optimized for Kaggle environments.
    Handles GPU memory, system temp files, and Python memory with multiple strategies.
    """
    print("Cleaning up memory with Kaggle-optimized strategies...")
    
    # Clear Python's garbage collector multiple times
    for _ in range(3):
        gc.collect()
    
    # Force Python to release memory to OS if possible
    if hasattr(gc, 'mem_free'):
        gc.mem_free()  # Some Python installations support this
    
    if torch.cuda.is_available():
        # Empty CUDA cache with multiple strategies
        torch.cuda.empty_cache()  # Standard cleanup
        torch.cuda.synchronize()  # Wait for all CUDA operations to finish
        
        # Try to release all CUDA memory and reinitialize if necessary
        for i in range(torch.cuda.device_count()):
            # Force deallocate any unused memory
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats(i)
            if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                try:
                    torch.cuda.reset_accumulated_memory_stats(i)
                except:
                    pass
            
            # Print memory info after cleanup
            if hasattr(torch.cuda, 'memory_allocated'):
                alloc = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  GPU {i} after cleanup: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Try to explicitly clear CUDA memory pools
        try:
            # This works on newer PyTorch versions
            torch.cuda._cached_memory_pool.empty_cache()
        except:
            pass
    
    # Explicitly delete any large objects that might be in memory
    # First identify all large objects
    large_objects = []
    
    # Look for larger threshold on tensors and arrays which are more common in ML
    for var_name in list(globals().keys()):
        var = globals()[var_name]
        try:
            # Calculate size more accurately for common ML objects
            if isinstance(var, torch.Tensor):
                size = var.element_size() * var.nelement()
                if size > 1e6:  # 1MB
                    large_objects.append(var_name)
            elif isinstance(var, (list, dict, set)):
                size = sys.getsizeof(var)
                if size > 5e6:  # 5MB
                    large_objects.append(var_name)
            # Look for numpy arrays too
            elif 'numpy' in str(type(var)):
                try:
                    size = var.nbytes
                    if size > 1e6:
                        large_objects.append(var_name)
                except:
                    pass
        except:
            continue
    
    # Delete identified large objects
    for obj in large_objects:
        if obj in globals():
            print(f"  Deleting large object: {obj}")
            try:
                del globals()[obj]
            except:
                pass
    
    # Clean Kaggle temp directories if they exist
    try:
        # Common Kaggle temp directories that might accumulate files
        kaggle_temp_dirs = [
            "/tmp/transformers_cache",
            "/tmp/torch_cache",
            "/tmp/huggingface",
            "/kaggle/working/tmp",
            "/kaggle/temp"
        ]
        
        import os
        import shutil
        
        for temp_dir in kaggle_temp_dirs:
            if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                print(f"Cleaning Kaggle temp directory: {temp_dir}")
                try:
                    # Delete files older than 1 hour
                    for root, dirs, files in os.walk(temp_dir):
                        for f in files:
                            try:
                                full_path = os.path.join(root, f)
                                # Only delete if older than 1 hour and not a necessary file
                                if os.path.getmtime(full_path) < time.time() - 3600:
                                    os.remove(full_path)
                            except:
                                pass
                except Exception as e:
                    print(f"Error cleaning temp dir {temp_dir}: {e}")
    except Exception as e:
        print(f"Error during Kaggle temp cleanup: {e}")
    
    # Run gc again at the end
    gc.collect()
    
    # Try to force system to release memory to OS
    try:
        # Some systems support this call to release memory to the OS
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass
        
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
BATCH_SIZE = 1  # Reduced batch size to 1 to minimize memory usage per GPU
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
GRADIENT_ACCUMULATION_STEPS = 8  # Increased gradient accumulation to compensate for smaller batch size

# Output configuration
OUTPUT_DIR = "./phi3_swift_model"  # Single output directory for all model artifacts

# LoRA configuration
LORA_R = 8  # Reduced from 16 to save memory
LORA_ALPHA = 16  # Reduced from 32 to save memory
LORA_DROPOUT = 0.05

# Debug mode for testing with smaller dataset
DEBUG_MODE = True
DEBUG_SAMPLE_SIZE = 100

# Memory optimization flags
USE_CPU_OFFLOAD = True
USE_MEMORY_EFFICIENT_ATTENTION = True
USE_ACTIVATION_CHECKPOINTING = True
USE_SEQUENTIAL_OFFLOAD = True
OFFLOAD_FOLDER = "./offload_folder"  # For disk offloading
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

print(f"Using model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Batch size: {BATCH_SIZE} per device")
print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * (2 if torch.cuda.device_count() > 1 else 1) * GRADIENT_ACCUMULATION_STEPS}")
print(f"LoRA rank: {LORA_R}")
print(f"Memory optimizations: CPU Offload={USE_CPU_OFFLOAD}, Memory-Efficient Attention={USE_MEMORY_EFFICIENT_ATTENTION}")


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
# Set up for training without unnecessary dependencies

# Set up training arguments with optimized settings for multi-GPU training
try:
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Configure training arguments with enhanced memory optimizations for multi-GPU training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
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
        
        # Memory optimization settings
        fp16=True,                    # Use mixed precision training
        bf16=False,                   # Don't use bfloat16 (T4 GPUs don't support it)
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        optim="adamw_torch_fused",    # Use memory-efficient fused optimizer
        
        # Advanced memory settings
        max_grad_norm=0.3,            # Reduce gradient norm for stability
        group_by_length=True,         # Group sequences of similar length to reduce padding
        dataloader_pin_memory=False,  # Disable pinning memory for less RAM usage
        
        # Distributed training parameters
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),  # For distributed training
        ddp_find_unused_parameters=True,                   # Required for Phi-3 models in multi-GPU setup
        ddp_bucket_cap_mb=50,                             # Limit communication buffer size
        dataloader_num_workers=1,                          # Reduced from 4 to save memory
        dataloader_prefetch_factor=2,                      # Limit prefetching to save memory
        report_to="none",                                  # Disable reporting to avoid overhead
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

# Create a flag to track which quantization method we're using
USING_GGUF = False
QUANT_BITS = 2  # Default to 2-bit quantization

print(f"📥 Loading {MODEL_NAME} with {QUANT_BITS}-bit quantization...")

try:
    # First check if GGUF libraries are available
    if ('ctransformers' in globals() or 'ctransformers' in locals()) and ('llama_cpp' in globals() or 'llama_cpp' in locals()):
        # Use GGUF for 2-bit quantization
        print(f"Using GGUF for {QUANT_BITS}-bit quantization...")
        
        # First load the model with HF transformers
        print(f"Loading base model {MODEL_NAME} with standard transformers to prepare for GGUF conversion...")
        
        # Standard loading with memory optimization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            use_cache=False,  # Disable KV cache for better memory efficiency
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        )
        
        # Create temporary directory for GGUF conversion
        import tempfile
        import os
        from pathlib import Path
        
        temp_dir = Path(tempfile.mkdtemp())
        hf_model_path = temp_dir / "hf_model"
        gguf_output_path = temp_dir / "model.gguf"
        
        print(f"Saving model to temporary directory for GGUF conversion: {hf_model_path}")
        
        # Save the model in HF format first
        model.save_pretrained(hf_model_path)
        tokenizer.save_pretrained(hf_model_path)
        
        # Now apply GGUF 2-bit quantization using conversion tools
        print(f"Converting to GGUF with {QUANT_BITS}-bit quantization...")
        
        try:
            # First method: Use ctransformers for quantization
            from ctransformers.lib import convert_hf_to_gguf
            
            # Define GGUF quantization type based on bit level
            if QUANT_BITS == 2:
                quant_type = "q2_k"  # 2-bit quantization with k-quants
            elif QUANT_BITS == 3:
                quant_type = "q3_k"  # 3-bit quantization
            elif QUANT_BITS == 4:
                quant_type = "q4_k"  # 4-bit quantization
            else:
                quant_type = "q2_k"  # Default to 2-bit if not specified
                
            print(f"Using GGUF quantization type: {quant_type}")
            
            # Convert the model to GGUF format with the specified quantization
            convert_hf_to_gguf(
                str(hf_model_path),
                str(gguf_output_path),
                quantization_type=quant_type
            )
            
            # Load the GGUF model for verification
            from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
            
            gguf_model = CTAutoModelForCausalLM.from_pretrained(
                str(gguf_output_path),
                model_type="phi",  # Specify it's a Phi model
                gpu_layers=24,     # Use GPU for most layers
                context_length=MAX_LENGTH
            )
            
            print(f"Successfully converted and loaded GGUF {QUANT_BITS}-bit quantized model")
            
            # Now we need to integrate the GGUF model with our HF training pipeline
            # We'll keep the original model for training with LoRA but save the GGUF model for inference
            print("Saving the GGUF quantized model for inference after training")
            
            # Create directory for the GGUF model
            os.makedirs("./phi3_swift_model_gguf", exist_ok=True)
            
            # Copy the GGUF model to the output directory
            import shutil
            shutil.copy(gguf_output_path, "./phi3_swift_model_gguf/model.gguf")
            
            # Continue with HF model for training but set the flag to indicate we'll use GGUF for inference
            USING_GGUF = True
            print(f"Will use GGUF {QUANT_BITS}-bit quantization for inference after training")
            
        except Exception as e:
            print(f"Error with ctransformers GGUF conversion: {e}")
            print("Trying alternative GGUF conversion method...")
            
            # Try alternative method: Use llama.cpp for conversion if available
            try:
                import llama_cpp
                from llama_cpp import Llama
                
                # Determine the appropriate quantization type
                if QUANT_BITS == 2:
                    quant_type = "Q2_K"
                elif QUANT_BITS == 3:
                    quant_type = "Q3_K" 
                elif QUANT_BITS == 4:
                    quant_type = "Q4_K"
                else:
                    quant_type = "Q2_K"  # Default to 2-bit
                
                # First check if llama-cpp-python includes the conversion tool directly
                if hasattr(llama_cpp, "convert_hf_to_gguf"):
                    print("Using llama-cpp-python's built-in converter")
                    llama_cpp.convert_hf_to_gguf(
                        str(hf_model_path),
                        str(gguf_output_path),
                        quantization_type=quant_type
                    )
                else:
                    # Otherwise, use the command-line tool
                    print("Using command-line llama-cpp-python converter")
                    import subprocess
                    subprocess.check_call([
                        "python", "-m", "llama_cpp.convert_hf_to_gguf",
                        str(hf_model_path),
                        "--outfile", str(gguf_output_path),
                        "--quantize", quant_type
                    ])
                
                # Verify the model can be loaded
                model_gguf = Llama(
                    model_path=str(gguf_output_path),
                    n_ctx=MAX_LENGTH,
                    n_gpu_layers=24
                )
                
                print(f"Successfully converted and loaded GGUF {QUANT_BITS}-bit quantized model with llama-cpp")
                
                # Save the model for later use
                os.makedirs("./phi3_swift_model_gguf", exist_ok=True)
                shutil.copy(gguf_output_path, "./phi3_swift_model_gguf/model.gguf")
                
                USING_GGUF = True
                print(f"Will use GGUF {QUANT_BITS}-bit quantization for inference after training")
                
            except Exception as llama_cpp_error:
                print(f"llama-cpp-python GGUF conversion also failed: {llama_cpp_error}")
                raise  # Let it fall through to BitsAndBytes fallback
    else:
        raise ImportError("GGUF libraries not available")
        
except Exception as e:
    # Fallback to using BitsAndBytes for 4-bit quantization
    print(f"Falling back to BitsAndBytes 4-bit quantization: {e}")
    QUANT_BITS = 4
    USING_GGUF = False
    
    # BitsAndBytes 4-bit quantization with CPU offloading for training
    # Note: This requires the multi-backend version of BitsAndBytes we installed earlier
    print("Using 4-bit quantization with CPU offloading support for training")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # Add CPU offloading settings for training
        bnb_4bit_compute_dtype_for_cpu_offload=torch.float32,
        # Additional settings to help with CPU offloaded training
        llm_int8_enable_fp32_cpu_offload=True,  # Critical for CPU offloading during training
        llm_int8_has_fp16_weight=True,  # Help with mixed precision during offloading
        llm_int8_threshold=6.0  # Adjust threshold for better quantization quality
    )
    
    # We'll use hybrid GPU/CPU configuration with multi-backend BitsAndBytes support
    print("Initializing with CPU offloading for multi-backend BitsAndBytes training...")
    
    # First initialize with empty weights to avoid OOM during loading
    with init_empty_weights():
        config = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            return_dict=True
        )
        # Add memory-efficient attention method to config if available
        if USE_MEMORY_EFFICIENT_ATTENTION and hasattr(config, "attention_implementation"):
            print("Enabling memory-efficient attention...")
            config.attention_implementation = "flash_attention_2"
    
    # Create hybrid device map that allows CPU offloading with multi-backend BitsAndBytes
    offload_folder = OFFLOAD_FOLDER if USE_SEQUENTIAL_OFFLOAD else None
    print(f"Using offload folder: {offload_folder}")
    
    # Configure device map for multi-backend BitsAndBytes training
    device_map = {
        "model.embed_tokens": 0,  # Keep embeddings on GPU 0
        "model.norm": 0,          # Keep normalization on GPU 0
        "lm_head": 0,             # Keep LM head on GPU 0
    }
    
    # Set additional environment variables specific to bitsandbytes multi-backend training
    os.environ["BNB_ENABLE_TRAINING_OFFLOAD"] = "1"  # Critical for bitsandbytes multi-backend support
        
        # Distribute layers across devices with CPU/disk offloading
        # Get number of layers with compatibility for different model types
        n_layers = None
        
        # First, try direct model name detection which is most reliable for Phi-3 models
        if "phi-3" in MODEL_NAME.lower() or "phi3" in MODEL_NAME.lower():
            # Use hardcoded values for known Phi-3 variants
            if "mini" in MODEL_NAME.lower():
                n_layers = 32  # Phi-3-mini has 32 layers
                print(f"Direct model name detection: Phi-3-mini has {n_layers} layers")
            elif "medium" in MODEL_NAME.lower():
                n_layers = 60  # Phi-3-medium has 60 layers
                print(f"Direct model name detection: Phi-3-medium has {n_layers} layers")
            elif "small" in MODEL_NAME.lower():
                n_layers = 26  # Phi-3-small has 26 layers
                print(f"Direct model name detection: Phi-3-small has {n_layers} layers") 
            else:
                n_layers = 32
                print(f"Direct model name detection: Unknown Phi-3 variant, assuming {n_layers} layers")
        
        # If we couldn't determine layers by name, try config attributes
        if n_layers is None:
            # Try different attribute names used by various model architectures
            for attr_name in ["num_hidden_layers", "n_layer", "num_layers", "n_blocks"]:
                if hasattr(config, attr_name):
                    n_layers = getattr(config, attr_name)
                    print(f"Found layers count using config.{attr_name}: {n_layers}")
                    break
        
        # As fallback for Phi-3 models, try to detect layers by architecture patterns
        if n_layers is None:
            # For Phi-3 models specifically
            if hasattr(config, "model_type") and "phi" in config.model_type.lower():
                # Phi-3 typically has 32 layers in the mini version
                if "mini" in MODEL_NAME.lower():
                    n_layers = 32
                # Phi-3-medium typically has 60 layers
                elif "medium" in MODEL_NAME.lower():
                    n_layers = 60
                # Phi-3-small typically has 26 layers
                elif "small" in MODEL_NAME.lower():
                    n_layers = 26
                else:
                    # Default to 32 for unknown Phi-3 variants
                    n_layers = 32
                print(f"Using estimated layers for Phi-3 model: {n_layers}")
            else:
                # Last resort default
                n_layers = 24
                print(f"Warning: Could not determine number of layers, using default: {n_layers}")
        
        # With multi-backend BitsAndBytes, we can use CPU offloading for training
        # Split layers across GPUs and CPU optimally
        gpu0_layers = n_layers // 3      # ~33% on GPU 0
        gpu1_layers = n_layers // 3      # ~33% on GPU 1
        cpu_layers = n_layers - gpu0_layers - gpu1_layers  # ~34% on CPU with offloading
        print(f"Using hybrid configuration with multi-backend BitsAndBytes: {gpu0_layers} layers on GPU 0, {gpu1_layers} layers on GPU 1, {cpu_layers} layers on CPU")
        
        # Determine the correct layer naming pattern for different model architectures
        # For Phi-3 models, the pattern might be different than standard transformers
        layer_prefix = "model.layers"  # Default pattern
        
        # Try to determine correct layer prefix from model config
        if hasattr(config, "model_type"):
            model_type = config.model_type.lower()
            if "phi" in model_type:
                # Phi models may use different naming conventions
                # We'll try a few common patterns for Phi-3
                layer_prefix_options = [
                    "model.layers",         # Standard pattern
                    "transformer.h",        # Some Phi models use this
                    "model.decoder.layers", # Another common pattern
                    "model.transformer.h",  # Yet another pattern
                ]
                
                # Log the possible patterns we're going to try
                print(f"Phi model detected, will try these layer prefix patterns: {layer_prefix_options}")
                
                # Use the first one by default, the device_map will be adjusted at runtime if needed
                layer_prefix = layer_prefix_options[0]
        
        print(f"Using layer prefix pattern: {layer_prefix}")
        
        # Assign layers to devices with multiple pattern fallbacks
        for i in range(n_layers):
            layer_device = None
            if i < gpu0_layers:
                layer_device = 0
            elif i < gpu0_layers + gpu1_layers:
                layer_device = 1
            else:
                layer_device = "cpu"
            
            # Add all possible layer naming patterns to device map to ensure we catch the right one
            if hasattr(config, "model_type") and "phi" in config.model_type.lower():
                # Add all possible patterns for Phi models - optimized for Phi-3
                # Primary pattern for Phi-3
                device_map[f"model.layers.{i}"] = layer_device
                
                # Additional patterns with Phi-3-specific paths
                device_map[f"transformer.h.{i}"] = layer_device
                device_map[f"model.decoder.layers.{i}"] = layer_device
                device_map[f"model.transformer.h.{i}"] = layer_device
                
                # Phi-3-specific naming patterns based on model inspection
                device_map[f"phi_model.layers.{i}"] = layer_device
                device_map[f"layers.{i}"] = layer_device
                
                # For absolute reliability, add all possible nested paths where Phi-3 layers could be
                device_map[f"base_model.model.layers.{i}"] = layer_device
                device_map[f"model.model.layers.{i}"] = layer_device
            else:
                # Standard pattern for other models
                device_map[f"{layer_prefix}.{i}"] = layer_device
        
        print(f"Device map: GPU 0: {gpu0_layers} layers, GPU 1: {gpu1_layers} layers, CPU: {cpu_layers} layers")
        
        # Load with offloading and quantization - with error handling and fallbacks
        try:
            print("Attempting to load model with custom device map and CPU offloading...")
            
            # Create a modified version of BnB config that enables CPU offloading
            # This is required when using quantization with CPU offloading 
            cpu_offload_bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading with quantized models
                llm_int8_threshold=6.0,  # Increase threshold for more efficient offloading
            )
            
            # Kaggle-optimized max memory allocation - be more conservative
            max_memory = {
                0: "9GB",          # Reserve more headroom on GPU 0
                1: "9GB",          # Reserve more headroom on GPU 1
                "cpu": "24GB",     # Limit CPU memory usage for Kaggle
            }
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=cpu_offload_bnb_config,  # Use modified config with CPU offloading enabled
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=True,
                max_memory=max_memory,  # Add explicit memory limits
                low_cpu_mem_usage=True,  # Enable more aggressive CPU memory optimization
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_cache=False,
                attn_implementation="eager"  # Always use eager implementation for compatibility
            )
        except Exception as e:
            print(f"Error loading with custom device map: {e}")
            print("Falling back to automatic device map...")
            
            # Perform deep cleanup for Kaggle environment
            for _ in range(3):  # Multiple cleanup passes
                cleanup_memory()  
                time.sleep(1)  # Give the system time to reclaim memory
            
            # Try again with much simpler configuration optimized for Kaggle
            print("Using simplified auto device mapping for maximum compatibility...")
            
            # Create simpler BnB config with multi-backend support for auto device mapping
            simple_bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                # Keep multi-backend support even in fallback mode
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=True
            )
            
            # Set multi-backend environment variables for fallback as well
            os.environ["BNB_ENABLE_TRAINING_OFFLOAD"] = "1"
            
            # Kaggle-optimized max memory settings - be very conservative
            kaggle_max_memory = {
                0: "8GB",          # Very conservative GPU 0 limit for Kaggle 
                1: "8GB",          # Very conservative GPU 1 limit for Kaggle
                "cpu": "24GB",     # Limited CPU memory for Kaggle
            }
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=simple_bnb_config,
                device_map="auto",  # Let HF decide automatically - most compatible option
                max_memory=kaggle_max_memory,  # Conservative memory limits for Kaggle
                low_cpu_mem_usage=True,  # More aggressive CPU memory optimization
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_cache=False
            )
    else:
        # Standard loading with basic memory optimization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=False
        )
    print("Successfully loaded model with BitsAndBytes 4-bit quantization")

# Configure LoRA for fine-tuning with memory optimizations
print("Setting up memory-optimized LoRA fine-tuning...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    # Target only essential modules to save memory
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj"],
    modules_to_save=None,  # Don't save any modules fully to save memory
)

# Prepare the model for training with LoRA and additional memory optimizations
print("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# Enable activation checkpointing to save memory if requested
if USE_ACTIVATION_CHECKPOINTING:
    print("Enabling activation checkpointing to save memory...")
    try:
        # For transformers models
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled via model method")
        # For torch models
        elif hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            print("Input require grads enabled")
            
        # Enable checkpointing on specific modules (for newer transformers versions)
        for module in model.modules():
            if hasattr(module, "_use_gradient_checkpointing"):
                module._use_gradient_checkpointing = True
    except Exception as e:
        print(f"Warning: Could not enable activation checkpointing: {e}")

# Apply LoRA and free memory
print("Applying LoRA adapter...")
model = get_peft_model(model, lora_config)

# Run explicit memory cleanup after model initialization
cleanup_memory()

# Report model parameter counts
print(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Model parameters using memory: {sum(p.numel() * (2 if p.dtype == torch.float16 else 4) for p in model.parameters()) / (1024**2):.2f} MB")

# Print information about the quantized model
quant_method = "GGUF" if USING_GGUF else "BitsAndBytes"
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

# Create trainer with memory optimizations for GPU/CPU
print("Setting up memory-optimized trainer...")

# Ensure environment variables are set for transformers
os.environ["TRANSFORMERS_SKIP_TORCH_VISION_IMPORT"] = "1"

# Add additional memory optimization callbacks
from transformers.trainer_callback import TrainerCallback

class OOMGuardCallback(TrainerCallback):
    """Custom callback that detects and prevents OOM errors"""
    def on_step_end(self, args, state, control, **kwargs):
        # Check memory usage on each GPU after step
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory
            if allocated > 0.92:  # Over 92% memory usage
                # Force immediate garbage collection and cache clearing
                gc.collect()
                torch.cuda.empty_cache()
                print(f"\n⚠️ Warning: GPU {i} memory usage high ({allocated*100:.1f}%). Forcing cache clear.")
                # Pause briefly to allow memory release
                time.sleep(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Clean up between epochs for better stability
        gc.collect()
        torch.cuda.empty_cache()
        return control

class MemoryOptimizationCallback(TrainerCallback):
    """Custom callback to optimize memory usage during training."""
    
    def on_step_end(self, args, state, control, **kwargs):
        """Run cleanup after each optimization step."""
        # Free memory every few steps
        if state.global_step % 10 == 0:
            cleanup_memory()
            
            # Monitor GPU memory usage every 100 steps
            if state.global_step % 100 == 0:
                print(f"\n--- Memory status at step {state.global_step} ---")
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                # Also monitor CPU memory
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                print(f"CPU memory: {memory_info.rss / (1024**3):.2f}GB")
        
        return control
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Run before evaluation begins."""
        print("\nRunning memory cleanup before evaluation...")
        cleanup_memory()
        return control
    
    def on_save(self, args, state, control, **kwargs):
        """Run before model is saved."""
        print("\nRunning memory cleanup before saving model...")
        cleanup_memory()
        return control
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Run at the end of each epoch."""
        print(f"\nCompleted epoch. Running thorough memory cleanup...")
        cleanup_memory()
        return control

# Create custom memory-efficient trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[
        early_stopping_callback,
        MemoryOptimizationCallback(),
        OOMGuardCallback()
    ]
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


print("🚀 Starting training process with enhanced memory management...")
print("This will take some time. Training progress will be displayed below.")

# Run training with aggressive memory optimization for multi-GPU setup
try:
    # Monitor resources before training
    print("Resources before training:")
    monitor_resources()
    
    # Super aggressive memory cleanup before training
    print("Performing deep memory cleanup before training...")
    # Force multiple garbage collection passes
    for _ in range(3):
        cleanup_memory()
        time.sleep(0.5)  # Brief pause to allow OS to reclaim memory
    
    # Set environment variables for even more aggressive memory management
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Disable CUDA caching
    
    # Configure PyTorch for memory-efficient multi-GPU training
    if torch.cuda.device_count() > 1:
        print("Configuring PyTorch for memory-efficient multi-GPU training...")
        # Enable TF32 precision for faster training (on Ampere GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # More conservative memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.85)  # Reserve more memory (15%) for system
        
        # For Kaggle, use the most reliable attention mechanism rather than Flash Attention
        print("Using standard attention for maximum Kaggle compatibility")
            
        # Configure CUDA memory usage more conservatively for Kaggle
        if torch.cuda.is_available():
            # Set more conservative memory fraction for Kaggle's T4 GPUs
            torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve 20% for Kaggle system
                
            # For Kaggle's T4 GPUs, avoid enabling TF32 which can consume more memory
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
                
            # Explicitly disable any caching mechanisms that could leak memory
            torch.backends.cudnn.benchmark = False
                
            print("Configured CUDA memory settings for Kaggle environment")
                
        # Set additional environment variables specifically for Kaggle
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism to save memory
    
    # Set up a memory monitor thread for continuous monitoring during training
    def memory_monitoring_thread():
        print("Starting memory monitoring thread...")
        while True:
            try:
                # Only log if we're close to OOM
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    if allocated / total > 0.85:  # If using more than 85% of memory
                        print(f"⚠️ WARNING: GPU {i} memory usage high: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")
                        cleanup_memory()  # Try to free up memory
            except Exception as e:
                print(f"Error in memory monitoring: {e}")
            time.sleep(10)  # Check every 10 seconds

    # Start memory monitoring in a separate thread if not in debug mode
    if not DEBUG_MODE:
        import threading
        monitor_thread = threading.Thread(target=memory_monitoring_thread, daemon=True)
        monitor_thread.start()
    
    # Start training 
    start_time = time.time()
    
    # Configure training to catch OOM errors and retry with more aggressive settings
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\n🚀 Starting training (attempt {attempt+1}/{max_retries})...")
            
            # Additional cleanup right before training
            cleanup_memory()
            
            # Run training
            train_result = trainer.train()
            
            # If we get here, training succeeded
            print("✅ Training completed successfully!")
            break
            
        except RuntimeError as e:
            # Check if this is an OOM error
            if "CUDA out of memory" in str(e):
                print(f"❌ CUDA out of memory error (attempt {attempt+1}/{max_retries})")
                
                # More aggressive cleanup
                cleanup_memory()
                
                if attempt < max_retries - 1:
                    # Try more aggressive memory saving for next attempt
                    print("Applying more aggressive memory optimizations for next attempt...")
                    
                    # Reduce batch size if possible
                    if trainer.args.per_device_train_batch_size > 1:
                        trainer.args.per_device_train_batch_size //= 2
                        trainer.args.per_device_eval_batch_size //= 2
                        print(f"Reduced batch size to {trainer.args.per_device_train_batch_size}")
                    
                    # Increase gradient accumulation steps
                    trainer.args.gradient_accumulation_steps *= 2
                    print(f"Increased gradient accumulation to {trainer.args.gradient_accumulation_steps}")
                    
                    # Wait a moment for memory to be fully reclaimed
                    time.sleep(10)
                else:
                    print("Maximum retry attempts reached. Training failed.")
                    raise
            else:
                # This is not an OOM error, re-raise
                raise
    
    # Monitor resources after training
    print("Resources after training:")
    monitor_resources()
    
    # Print training results
    print(f"Training completed in {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Training loss: {train_result.metrics['train_loss']:.4f}")
    
    # Save the model with appropriate method based on quantization used
    print("\n💾 Saving trained model...")
    
    # Save the model
    trainer.save_model(OUTPUT_DIR)
    
    # Determine the quantization method for display
    quant_method = "GGUF" if USING_GGUF else "BitsAndBytes"
    print(f"✅ Model saved to {OUTPUT_DIR} ({QUANT_BITS}-bit {quant_method} quantized)")
    if USING_GGUF:
        print(f"✅ GGUF model also saved to {OUTPUT_DIR}/model.gguf")
    print(f"   Trained on: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Save model configuration details
    with open(f"{OUTPUT_DIR}/quantization_config.json", "w") as f:
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
    
    # Skip creating loading instructions - we just want the model files
    
    # Add a minimal config file for reference
    with open(f"{OUTPUT_DIR}/model_info.txt", "w") as f:
        f.write(f"""MODEL INFO
Model: {MODEL_NAME}
Quantization: {quant_method} {QUANT_BITS}-bit
Trained on: {DATASET_ID}
Date: {time.strftime("%Y-%m-%d")}
""")
        print("✅ Model info saved")
    
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
