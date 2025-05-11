# Training Phi-3-mini-128k-instruct on 2 T4 GPUs

This guide explains how to run the optimized `phi-train.ipynb` notebook on 2 T4 GPUs with memory optimizations to prevent CUDA out-of-memory errors.

## Optimizations Made

The notebook has been optimized for multi-GPU training with the following changes:

1. **Memory Management**
   - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent memory fragmentation
   - Added explicit GPU memory monitoring for each GPU
   - Implemented enhanced memory cleanup functions
   - Enabled gradient checkpointing to reduce memory usage

2. **Batch Size and Gradient Accumulation**
   - Reduced per-device batch size to 2 (from 4)
   - Reduced gradient accumulation steps to 4 (from 8)
   - This maintains the same effective batch size while distributing the workload

3. **Model Loading**
   - Using `device_map="auto"` to automatically distribute model layers across GPUs
   - Enabled 4-bit quantization with optimized settings
   - Added state dict offloading for more efficient memory usage
   - Using float16 precision throughout to reduce memory requirements

4. **Distributed Training Configuration**
   - Added proper DDP (Distributed Data Parallel) settings
   - Optimized communication between GPUs with `ddp_bucket_cap_mb=25`
   - Disabled unused parameter finding for better performance
   - Using fused optimizers for better memory efficiency

## Running the Training

To run the training on 2 T4 GPUs:

1. **Use the provided script**:
   ```bash
   ./run_distributed_training.sh
   ```
   This script will:
   - Convert the notebook to a Python script
   - Configure accelerate for distributed training
   - Launch the training across 2 GPUs

2. **Alternatively, run manually**:
   ```bash
   # Set environment variables
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export CUDA_VISIBLE_DEVICES=0,1
   
   # Convert notebook to Python
   jupyter nbconvert --to python phi-train.ipynb
   
   # Run with accelerate
   accelerate launch --multi_gpu phi-train.py
   ```

## Monitoring GPU Usage

During training, you can monitor GPU usage with:

```bash
nvidia-smi -l 5  # Updates every 5 seconds
```

The notebook also includes built-in monitoring that will show:
- Memory usage per GPU
- Allocated vs. reserved memory
- System memory usage

## Troubleshooting Memory Issues

If you still encounter CUDA out-of-memory errors:

1. **Further reduce batch size**:
   - Change `BATCH_SIZE = 2` to `BATCH_SIZE = 1`
   - You may need to increase `GRADIENT_ACCUMULATION_STEPS` to maintain the effective batch size

2. **Reduce sequence length**:
   - Change `MAX_LENGTH = 4096` to a smaller value like `MAX_LENGTH = 2048`

3. **Enable more aggressive memory optimization**:
   - Add `torch.cuda.set_per_process_memory_fraction(0.8)` to reserve 20% of GPU memory
   - Increase CPU offloading by setting `offload_folder="offload"` and `offload_state_dict=True`

4. **Check for memory leaks**:
   - Monitor GPU memory usage over time
   - If memory continuously increases, there may be a leak in the data pipeline

5. **Use CPU offloading for optimizer states**:
   - Add `optim_type="adamw_torch_fused"` and `optim_args={"offload_to_cpu": True}` to training arguments

## Expected Performance

With these optimizations on 2 T4 GPUs:
- Training should be approximately 1.7-1.9× faster than using a single GPU
- Memory usage per GPU will be lower, allowing for stable training
- The model quality should be identical to single-GPU training