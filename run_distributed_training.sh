#!/bin/bash

# Install required packages
pip install -q accelerate jupyter nbconvert

# Set environment variables for PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

# Convert notebook to Python script
echo "Converting notebook to Python script..."
jupyter nbconvert --to python phi-train.ipynb

# Create accelerate config
echo "Creating accelerate config..."
cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
use_cpu: false
EOF

# Run training with accelerate
echo "Starting distributed training on 2 GPUs..."
accelerate launch --config_file accelerate_config.yaml phi-train.py

echo "Training complete!"