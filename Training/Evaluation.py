#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phi-3 Swift Model Evaluation Script

This script evaluates and compares a fine-tuned Phi-3 model against the original pre-trained model,
specifically for Swift programming language capabilities. It provides both quantitative metrics 
and qualitative examples to measure how much the model has learned.

Usage:
    python evaluate_phi3_models.py --finetuned_model "YOUR_MODEL_NAME" [--original_model "microsoft/Phi-3-mini-128k-instruct"] 
                                   [--quantization_bits 4] [--quantization_method "BitsAndBytes"] 
                                   [--output_dir "./evaluation_results"]

Example:
    python evaluate_phi3_models.py --finetuned_model "yourusername/phi3-swift-finetuned" --quantization_bits 4
"""

import os
import sys
import json
import time
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)

# Try to import AQLM for 2-bit quantization
try:
    import aqlm
    AQLM_AVAILABLE = True
    print("AQLM package found - 2-bit quantization available")
except ImportError:
    AQLM_AVAILABLE = False
    print("AQLM package not found - only 4-bit quantization available with BitsAndBytes")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Phi-3 models for Swift programming capabilities")
    
    # Model arguments
    parser.add_argument(
        "--finetuned_model", 
        type=str,
        default="", # Leave empty for user to fill in
        help="HuggingFace model ID of the fine-tuned model (e.g., 'yourusername/phi3-swift-finetuned')"
    )
    parser.add_argument(
        "--original_model", 
        type=str,
        default="microsoft/Phi-3-mini-128k-instruct",
        help="HuggingFace model ID of the original model to compare against"
    )
    
    # Quantization arguments
    parser.add_argument(
        "--quantization_bits", 
        type=int, 
        default=4,
        choices=[2, 4, 8, 16],
        help="Number of bits for quantization (2 requires AQLM, 4 uses BitsAndBytes)"
    )
    parser.add_argument(
        "--quantization_method", 
        type=str,
        default="auto",
        choices=["auto", "AQLM", "BitsAndBytes", "none"],
        help="Quantization method to use (auto will use AQLM for 2-bit if available, BitsAndBytes for 4-bit)"
    )
    
    # Dataset and evaluation arguments
    parser.add_argument(
        "--test_dataset_id", 
        type=str,
        default="mvasiliniuc/iva-swift-codeint",
        help="HuggingFace dataset ID for testing"
    )
    parser.add_argument(
        "--max_test_samples", 
        type=int,
        default=100,
        help="Maximum number of test samples to evaluate"
    )
    parser.add_argument(
        "--max_length", 
        type=int,
        default=4096,
        help="Maximum sequence length for tokenizer and generation"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--generate_plots", 
        action="store_true",
        help="Generate and save comparison plots"
    )
    
    return parser.parse_args()

def setup_device():
    """Set up the device for inference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        if torch.cuda.device_count() > 1:
            print(f"Found {torch.cuda.device_count()} GPUs, will use device_map='auto'")
            use_device_map = True
        else:
            use_device_map = True  # Still use device_map for single GPU for better memory management
            
        # Print available GPU memory
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"Available GPU memory: {free_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU - evaluation will be slow")
        use_device_map = False
        
    return device, use_device_map

def load_model(model_name, quantization_bits, quantization_method, use_device_map, device):
    """Load a model with the specified quantization."""
    print(f"Loading model: {model_name}")
    
    if not model_name:
        raise ValueError("Model name cannot be empty. Please provide a valid model name.")
    
    # Determine quantization method if set to auto
    if quantization_method == "auto":
        if quantization_bits == 2 and AQLM_AVAILABLE:
            quantization_method = "AQLM"
        elif quantization_bits == 4:
            quantization_method = "BitsAndBytes"
        else:
            quantization_method = "none"
    
    # Set up loading configuration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure the device map
    device_map = "auto" if use_device_map else None
    
    # Load the model with the specified quantization
    if quantization_method == "AQLM" and quantization_bits == 2:
        if not AQLM_AVAILABLE:
            raise ImportError("AQLM package is required for 2-bit quantization but not found.")
        
        print(f"Loading model with AQLM {quantization_bits}-bit quantization...")
        
        # First load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            use_cache=True,
            low_cpu_mem_usage=True
        )
        
        # Apply AQLM quantization
        try:
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
                        break
                else:
                    raise ImportError("Could not find quantize function in AQLM modules")
            
            # Apply quantization with default settings
            model = quantize_fn(
                base_model, 
                bits=quantization_bits,
                lora_rank=16  # Default LoRA rank value
            )
            print(f"Successfully loaded model with AQLM {quantization_bits}-bit quantization")
            
        except Exception as e:
            print(f"Failed to apply AQLM quantization: {e}")
            print("Falling back to BitsAndBytes 4-bit quantization")
            quantization_method = "BitsAndBytes"
            quantization_bits = 4
    
    if quantization_method == "BitsAndBytes":
        print(f"Loading model with BitsAndBytes {quantization_bits}-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=True
        )
        print(f"Successfully loaded model with BitsAndBytes {quantization_bits}-bit quantization")
    
    elif quantization_method == "none":
        print("Loading model without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=True
        )
        print("Successfully loaded model without quantization")
    
    return model, tokenizer, {"method": quantization_method, "bits": quantization_bits}

def prepare_swift_benchmark():
    """Define a structured benchmark for Swift programming tasks."""
    return [
        {
            "name": "Optional Unwrapping",
            "category": "Language Fundamentals",
            "prompt": "<|user|>\nExplain the key features of Swift's optional unwrapping syntax:\n\n```swift\nfunc processName(_ name: String?) {\n    guard let unwrappedName = name else {\n        print(\"No name provided\")\n        return\n    }\n    print(\"Hello, \\(unwrappedName)!\")\n}\n```\n<|assistant|>",
            "keywords": ["guard", "optional", "unwrap", "if let", "nil", "safety"]
        },
        {
            "name": "Factorial Implementation",
            "category": "Algorithm Implementation",
            "prompt": "<|user|>\nComplete this Swift function that calculates the factorial of a number:\n\n```swift\nfunc factorial(_ n: Int) -> Int {\n    // Add implementation here\n}\n```\n<|assistant|>",
            "keywords": ["recursion", "base case", "return 1", "multiply", "n * factorial"]
        },
        {
            "name": "Class Initialization",
            "category": "Object-Oriented Programming",
            "prompt": "<|user|>\nWhat's wrong with this Swift code and how can I fix it?\n\n```swift\nclass Person {\n    var name: String\n    var age: Int\n    \n    func greet() {\n        print(\"Hello, my name is \\(name) and I am \\(age) years old.\")\n    }\n}\n\nlet person = Person()\nperson.greet()\n```\n<|assistant|>",
            "keywords": ["initializer", "init", "required", "properties", "constructor"]
        },
        {
            "name": "Error Handling Patterns",
            "category": "Best Practices",
            "prompt": "<|user|>\nExplain Swift best practices for error handling:\n<|assistant|>",
            "keywords": ["throws", "do-catch", "try", "Error protocol", "Result type"]
        },
        {
            "name": "Protocol Implementation",
            "category": "Swift Protocols",
            "prompt": "<|user|>\nExplain how to use protocols in Swift and provide an example of protocol conformance:\n<|assistant|>",
            "keywords": ["protocol", "conform", "implement", "requirement", "extension"]
        },
        {
            "name": "SwiftUI View Creation",
            "category": "UI Development",
            "prompt": "<|user|>\nCreate a simple SwiftUI view that displays a list of items and allows selecting one:\n<|assistant|>",
            "keywords": ["SwiftUI", "List", "ForEach", "NavigationView", "State", "onTapGesture"]
        },
        {
            "name": "Concurrency with Swift",
            "category": "Advanced Features",
            "prompt": "<|user|>\nExplain Swift's async/await concurrency model and provide an example:\n<|assistant|>",
            "keywords": ["async", "await", "Task", "MainActor", "structured concurrency", "async let"]
        },
        {
            "name": "Memory Management",
            "category": "Language Fundamentals",
            "prompt": "<|user|>\nExplain Swift's approach to memory management and how to avoid retain cycles:\n<|assistant|>",
            "keywords": ["ARC", "weak", "unowned", "reference cycle", "closure capture", "self"]
        }
    ]

def generate_response(model, tokenizer, prompt, max_tokens=200, device="cuda"):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
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

def score_response(response, keywords):
    """Score a response based on the presence of expected keywords."""
    score = 0
    matched_keywords = []
    
    for keyword in keywords:
        if keyword.lower() in response.lower():
            score += 1
            matched_keywords.append(keyword)
    
    percentage = (score / len(keywords)) * 100 if keywords else 0
    return {
        "score": score,
        "total": len(keywords),
        "percentage": percentage,
        "matched_keywords": matched_keywords
    }

def calculate_perplexity(model, tokenizer, dataset, device, max_samples=100):
    """Calculate perplexity on an evaluation dataset."""
    # Limit dataset size
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    # Tokenize all the examples
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=4, shuffle=False
    )
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Get the number of tokens in the batch
            num_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

def run_model_evaluation(args):
    """Run the full model evaluation process."""
    print("\n" + "="*80)
    print("PHI-3 SWIFT MODEL EVALUATION")
    print("="*80)
    
    # 1. Setup
    device, use_device_map = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Load models
    print("\n[1] Loading models...")
    
    # Check if fine-tuned model name is provided
    if not args.finetuned_model:
        print("\nERROR: Please provide your fine-tuned model name with --finetuned_model")
        print("Example: --finetuned_model \"yourusername/phi3-swift-finetuned\"")
        return
    
    try:
        # Load fine-tuned model
        finetuned_model, tokenizer, quant_config = load_model(
            args.finetuned_model, 
            args.quantization_bits,
            args.quantization_method,
            use_device_map,
            device
        )
        
        # Load original model with same quantization for fair comparison
        original_model, _, _ = load_model(
            args.original_model,
            args.quantization_bits,
            args.quantization_method,
            use_device_map,
            device
        )
        
        models_loaded = True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        models_loaded = False
        return
    
    # 3. Setup Swift benchmark
    swift_benchmark = prepare_swift_benchmark()
    
    # 4. Run benchmark tests
    print("\n[2] Running Swift programming benchmark...")
    benchmark_results = []
    
    for test in swift_benchmark:
        print(f"\nEvaluating: {test['name']} ({test['category']})")
        
        # Generate responses from both models
        finetuned_response = generate_response(finetuned_model, tokenizer, test['prompt'], device=device)
        original_response = generate_response(original_model, tokenizer, test['prompt'], device=device)
        
        # Score responses
        finetuned_score = score_response(finetuned_response, test['keywords'])
        original_score = score_response(original_response, test['keywords'])
        
        # Calculate improvement
        score_diff = finetuned_score["percentage"] - original_score["percentage"]
        
        # Store result
        result = {
            "name": test['name'],
            "category": test['category'],
            "finetuned_score": finetuned_score,
            "original_score": original_score,
            "improvement": score_diff,
            "prompt": test['prompt'],
            "finetuned_response": finetuned_response,
            "original_response": original_response
        }
        
        benchmark_results.append(result)
        
        # Print results
        print(f"  Original model score: {original_score['percentage']:.1f}%")
        print(f"  Fine-tuned model score: {finetuned_score['percentage']:.1f}%")
        print(f"  Improvement: {score_diff:+.1f}%")
    
    # 5. Calculate overall metrics
    print("\n[3] Calculating overall learning metrics...")
    
    # Average scores
    avg_finetuned_score = sum(r["finetuned_score"]["percentage"] for r in benchmark_results) / len(benchmark_results)
    avg_original_score = sum(r["original_score"]["percentage"] for r in benchmark_results) / len(benchmark_results)
    avg_improvement = sum(r["improvement"] for r in benchmark_results) / len(benchmark_results)
    
    print(f"Average fine-tuned model score: {avg_finetuned_score:.2f}%")
    print(f"Average original model score: {avg_original_score:.2f}%")
    print(f"Average improvement: {avg_improvement:+.2f}%")
    
    # 6. Calculate perplexity if requested
    try:
        perplexity_calculated = False
        print("\n[4] Loading Swift test dataset for perplexity calculation...")
        
        # Load a small sample of the dataset
        test_dataset = load_dataset(args.test_dataset_id, split="test")
        
        # Create instruction dataset
        def create_instruction(example):
            return {
                "text": f"<|user|>\n{example['content']}\n<|assistant|>\n"
            }
        
        test_dataset = test_dataset.map(create_instruction)
        test_dataset = test_dataset.select(range(min(args.max_test_samples, len(test_dataset))))
        
        print(f"Loaded {len(test_dataset)} test samples")
        
        # Calculate perplexity
        print("\nCalculating perplexity for fine-tuned model...")
        finetuned_perplexity = calculate_perplexity(finetuned_model, tokenizer, test_dataset, device)
        print(f"Fine-tuned model perplexity: {finetuned_perplexity:.4f}")
        
        print("\nCalculating perplexity for original model...")
        original_perplexity = calculate_perplexity(original_model, tokenizer, test_dataset, device)
        print(f"Original model perplexity: {original_perplexity:.4f}")
        
        # Calculate improvement
        if original_perplexity > finetuned_perplexity:
            perplexity_improvement = ((original_perplexity - finetuned_perplexity) / original_perplexity) * 100
            print(f"Perplexity improvement: {perplexity_improvement:.2f}% better than original model")
        else:
            perplexity_decline = ((finetuned_perplexity - original_perplexity) / original_perplexity) * 100
            print(f"Perplexity decline: {perplexity_decline:.2f}% worse than original model")
            perplexity_improvement = -perplexity_decline
        
        perplexity_calculated = True
        
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        perplexity_calculated = False
    
    # 7. Generate plots if requested
    if args.generate_plots:
        try:
            print("\n[5] Generating evaluation plots...")
            
            # Create a directory for plots
            plots_dir = os.path.join(args.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot 1: Benchmark scores comparison
            plt.figure(figsize=(12, 6))
            categories = [r["name"] for r in benchmark_results]
            finetuned_scores = [r["finetuned_score"]["percentage"] for r in benchmark_results]
            original_scores = [r["original_score"]["percentage"] for r in benchmark_results]
            
            x = range(len(categories))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], original_scores, width, label='Original Model')
            plt.bar([i + width/2 for i in x], finetuned_scores, width, label='Fine-tuned Model')
            
            plt.xlabel('Task')
            plt.ylabel('Score (%)')
            plt.title('Model Performance Comparison by Task')
            plt.xticks(x, categories, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.savefig(os.path.join(plots_dir, "benchmark_comparison.png"))
            
            # Plot 2: Improvement by category
            plt.figure(figsize=(10, 6))
            
            # Group by category
            categories = {}
            for r in benchmark_results:
                if r["category"] not in categories:
                    categories[r["category"]] = []
                categories[r["category"]].append(r["improvement"])
            
            category_names = list(categories.keys())
            category_improvements = [sum(improvements)/len(improvements) for improvements in categories.values()]
            
            plt.bar(category_names, category_improvements)
            plt.xlabel('Category')
            plt.ylabel('Average Improvement (%)')
            plt.title('Model Improvement by Category')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(plots_dir, "category_improvement.png"))
            
            # Plot 3: Overall comparison
            if perplexity_calculated:
                plt.figure(figsize=(8, 6))
                
                metrics = ['Benchmark Score', 'Perplexity\n(lower is better)']
                finetuned_values = [avg_finetuned_score, finetuned_perplexity]
                original_values = [avg_original_score, original_perplexity]
                
                x = range(len(metrics))
                width = 0.35
                
                plt.bar([i - width/2 for i in x], original_values, width, label='Original Model')
                plt.bar([i + width/2 for i in x], finetuned_values, width, label='Fine-tuned Model')
                
                plt.xlabel('Metric')
                plt.ylabel('Value')
                plt.title('Overall Model Performance Comparison')
                plt.xticks(x, metrics)
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plt.savefig(os.path.join(plots_dir, "overall_comparison.png"))
            
            print(f"Plots saved to {plots_dir}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    # 8. Generate summary report
    print("\n" + "="*80)
    print("LEARNING ASSESSMENT SUMMARY")
    print("="*80)
    
    print("\nPerformance by category:")
    categories = set(test["category"] for test in swift_benchmark)
    for category in categories:
        category_tests = [r for r in benchmark_results if r["category"] == category]
        avg_category_score = sum(r["finetuned_score"]["percentage"] for r in category_tests) / len(category_tests)
        print(f"  - {category}: {avg_category_score:.2f}%")
    
    # Learning assessment based on scores
    if avg_finetuned_score > 80:
        learning_assessment = "Excellent"
    elif avg_finetuned_score > 60:
        learning_assessment = "Good"
    elif avg_finetuned_score > 40:
        learning_assessment = "Moderate"
    else:
        learning_assessment = "Limited"
    
    print(f"\nOverall learning assessment: {learning_assessment}")
    
    improvement_assessment = ""
    if avg_improvement > 20:
        improvement_assessment = "Substantial improvement over the original model"
    elif avg_improvement > 10:
        improvement_assessment = "Significant improvement over the original model"
    elif avg_improvement > 0:
        improvement_assessment = "Modest improvement over the original model"
    else:
        improvement_assessment = "No improvement over the original model"
    
    print(f"Comparative assessment: {improvement_assessment}")
    
    # 9. Save evaluation results
    print("\n[6] Saving evaluation results...")
    
    evaluation_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "finetuned": {
                "name": args.finetuned_model,
                "quantization": quant_config
            },
            "original": {
                "name": args.original_model,
                "quantization": quant_config
            }
        },
        "benchmark_results": [
            {
                "name": r["name"],
                "category": r["category"],
                "finetuned_score": {
                    "percentage": float(r["finetuned_score"]["percentage"]),
                    "score": r["finetuned_score"]["score"],
                    "total": r["finetuned_score"]["total"],
                    "matched_keywords": r["finetuned_score"]["matched_keywords"]
                },
                "original_score": {
                    "percentage": float(r["original_score"]["percentage"]),
                    "score": r["original_score"]["score"],
                    "total": r["original_score"]["total"],
                    "matched_keywords": r["original_score"]["matched_keywords"]
                },
                "improvement": float(r["improvement"])
            } for r in benchmark_results
        ],
        "average_metrics": {
            "finetuned_score": float(avg_finetuned_score),
            "original_score": float(avg_original_score),
            "improvement": float(avg_improvement)
        },
        "learning_assessment": learning_assessment,
        "improvement_assessment": improvement_assessment
    }
    
    if perplexity_calculated:
        evaluation_results["perplexity"] = {
            "finetuned": float(finetuned_perplexity),
            "original": float(original_perplexity),
            "improvement": float(perplexity_improvement)
        }
    
    # Save to file
    model_name_safe = args.finetuned_model.replace('/', '-')
    evaluation_file = os.path.join(args.output_dir, f"{model_name_safe}_evaluation.json")
    
    with open(evaluation_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation results saved to {evaluation_file}")
    
    # 10. Generate detailed report with examples
    examples_file = os.path.join(args.output_dir, f"{model_name_safe}_examples.md")
    
    with open(examples_file, "w") as f:
        f.write(f"# Phi-3 Swift Model Evaluation: {args.finetuned_model}\n\n")
        f.write(f"*Evaluation performed on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Fine-tuned model score**: {avg_finetuned_score:.2f}%\n")
        f.write(f"- **Original model score**: {avg_original_score:.2f}%\n")
        f.write(f"- **Average improvement**: {avg_improvement:+.2f}%\n")
        if perplexity_calculated:
            f.write(f"- **Fine-tuned perplexity**: {finetuned_perplexity:.4f}\n")
            f.write(f"- **Original perplexity**: {original_perplexity:.4f}\n")
            f.write(f"- **Perplexity improvement**: {perplexity_improvement:+.2f}%\n")
        f.write(f"- **Learning assessment**: {learning_assessment}\n")
        f.write(f"- **Improvement assessment**: {improvement_assessment}\n\n")
        
        f.write("## Category Performance\n\n")
        for category in categories:
            category_tests = [r for r in benchmark_results if r["category"] == category]
            avg_category_score = sum(r["finetuned_score"]["percentage"] for r in category_tests) / len(category_tests)
            avg_category_improvement = sum(r["improvement"] for r in category_tests) / len(category_tests)
            f.write(f"### {category}\n")
            f.write(f"- Average score: {avg_category_score:.2f}%\n")
            f.write(f"- Average improvement: {avg_category_improvement:+.2f}%\n\n")
        
        f.write("## Detailed Examples\n\n")
        for i, result in enumerate(benchmark_results):
            f.write(f"### Example {i+1}: {result['name']} ({result['category']})\n\n")
            
            # Show prompt
            f.write("**Prompt:**\n\n")
            f.write("```\n")
            f.write(result["prompt"].split("<|assistant|>")[0].replace("<|user|>", ""))
            f.write("\n```\n\n")
            
            # Show scores
            f.write("**Scores:**\n\n")
            f.write(f"- Original model: {result['original_score']['percentage']:.1f}%\n")
            f.write(f"- Fine-tuned model: {result['finetuned_score']['percentage']:.1f}%\n")
            f.write(f"- Improvement: {result['improvement']:+.1f}%\n\n")
            
            # Show matched keywords
            f.write("**Keywords matched by original model:** ")
            f.write(", ".join(result["original_score"]["matched_keywords"]) if result["original_score"]["matched_keywords"] else "None")
            f.write("\n\n")
            
            f.write("**Keywords matched by fine-tuned model:** ")
            f.write(", ".join(result["finetuned_score"]["matched_keywords"]) if result["finetuned_score"]["matched_keywords"] else "None")
            f.write("\n\n")
            
            # Show responses
            f.write("**Original model response:**\n\n")
            f.write("```\n")
            f.write(result["original_response"])
            f.write("\n```\n\n")
            
            f.write("**Fine-tuned model response:**\n\n")
            f.write("```\n")
            f.write(result["finetuned_response"])
            f.write("\n```\n\n")
            
            # Add separator
            f.write("---\n\n")
    
    print(f"Detailed examples report saved to {examples_file}")
    
    # Final message
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nYour fine-tuned model ({args.finetuned_model}) achieved a score of {avg_finetuned_score:.2f}%")
    print(f"This is {avg_improvement:+.2f}% compared to the original model ({args.original_model})")
    print(f"\nOverall learning assessment: {learning_assessment}")
    print(f"Comparative assessment: {improvement_assessment}")
    print("\nSee the output files for detailed results and examples:")
    print(f"  - {evaluation_file}")
    print(f"  - {examples_file}")
    if args.generate_plots:
        print(f"  - {plots_dir}/")
    
    return evaluation_results

if __name__ == "__main__":
    args = parse_arguments()
    run_model_evaluation(args)
