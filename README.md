# modded-llama2.c

An updated version of BabyLlama that runs in a Jupyter notebook, building on llama2.c and incorporating training improvements from modded-nanogpt. Focuses on training with the TinyStories dataset.

## Overview

This project is:
- An updated version of [BabyLlama](https://github.com/EN10/BabyLlama) - Simplified LLaMA for Jupyter notebooks
- Based on [llama2.c](https://github.com/karpathy/llama2.c) - Pure C implementation of LLaMA 2
- Incorporates training improvements from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- Focuses on efficient training with the TinyStories dataset

## Key Improvements from modded-nanogpt

### Architecture Improvements (via model.py modifications)
- ReLU² activation instead of SwiGLU
- QK-Normalization for improved training stability
- Zero-initialized projections for better convergence
- Untied & padded vocabulary head
- Logit Softcapping to control output distributions

### Training Improvements
- Parameter-grouped AdamW optimizer with different learning rates for:
  - Embedding layers
  - Hidden matrix weights
  - Norm/scalar parameters
  - Head layer
- Improved learning rate schedule: stable period followed by cosine decay
- Efficient gradient accumulation with mixed precision
- TinyStories dataset focus with smaller vocabulary (4096 tokens)

## Features

- Simplified implementation designed to run in Jupyter notebooks
- Pure C implementation for maximum performance and minimal dependencies
- Efficient training with gradient accumulation and mixed precision
- Support for distributed training (DDP)
- Cosine learning rate scheduling with warmup
- Gradient clipping and optimization techniques
- Checkpointing and model export capabilities
- Wandb integration for experiment tracking

## Requirements

- C compiler (gcc/clang)
- CUDA toolkit (for GPU support)
- Python 3.8+ (for training script)
- PyTorch
- Wandb (optional, for experiment tracking)

## Building

```bash
make
```

## Training

To train a model:

```bash
python train.py --out_dir=out --batch_size=64 --gradient_accumulation_steps=4
```

Key training parameters:
- `--out_dir`: Output directory for checkpoints
- `--batch_size`: Batch size for training
- `--gradient_accumulation_steps`: Number of steps for gradient accumulation
- `--base_learning_rate`: Initial learning rate (different LRs will be applied to parameter groups)
- `--max_iters`: Maximum number of training iterations
- `--cooldown_frac`: Fraction of training to spend cooling down LR
- `--vocab_size`: Vocabulary size (default: 4096 for TinyStories)

## Inference

To run inference with a trained model:

```bash
./main -m models/model.bin -n 256 -i "Your prompt here"
```

## Model Architecture

The implementation follows the LLaMA 2 architecture with:
- RMSNorm for layer normalization
- RoPE positional embeddings
- SwiGLU activation function
- Grouped-Query Attention (GQA)
- Sliding window attention

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [BabyLlama](https://github.com/EN10/BabyLlama) by EN10 - This project is an updated version of BabyLlama
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) by Keller Jordan