"""
This training script integrates improvements from modded-nanogpt:
- ReLU^2 activation (requires model.py modification)
- QK-Norm (requires model.py modification)
- Zero-initialized projections (requires model.py modification)
- Untied & padded vocabulary head (requires model.py modification)
- Logit Softcapping (requires model.py modification)
- AdamW with parameter grouping and specific LRs
- modded-nanogpt LR schedule (stable then decay)
- TinyStories dataset focus
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial, lru_cache # Added lru_cache

import torch
import torch.nn.functional as F # Added for potential use in softcap/loss if needed directly here

# Assume model.py is in the same directory and has been modified as described above
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task # Uses tinystories dataloader
from export import model_export

# -----------------------------------------------------------------------------
# Helper function from modded-nanogpt
def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

# -----------------------------------------------------------------------------
# I/O
out_dir = "out" # Changed output dir
eval_interval = 500 # Eval more often for smaller dataset/faster training
log_interval = 10 # Log more often
eval_iters = 50 # Fewer eval iterations needed for smaller dataset
eval_only = False
always_save_checkpoint = True # Save checkpoints more reliably
init_from = "scratch"  # 'scratch' or 'resume'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "tinystories-modded" # Changed project name
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data - TinyStories specific
# vocab_source = "llama2" # Keep if using llama2 tokenizer with tinystories
# vocab_size = 32000
vocab_source = "custom" # Specify custom for TinyStories vocab
vocab_size = 4096 # Example: Use the 4k vocab trained on TinyStories (make sure data/tok4096.model exists)
# dataset = "tinystories" # Implicit via Task.iter_batches
batch_size = 64  # Smaller batch size might be okay for smaller models/data
max_seq_len = 512 # Max sequence length for TinyStories can be shorter
# model - Adjust based on desired model size (e.g., ~15M, ~42M from llama2.c README)
dim = 512 # Example: For ~42M model
n_layers = 8
n_heads = 8
n_kv_heads = 8 # Can be same as n_heads or fewer for MQA/GQA
multiple_of = 32 # Keep, from Llama 2
dropout = 0.0 # Or 0.1 as in modded-nanogpt's 110M model
# adamw optimizer - Settings inspired by modded-nanogpt's Adam groups
gradient_accumulation_steps = 4 # Adjust as needed based on memory/batch size target
# Base LR - will be scaled per group. This is just a reference.
base_learning_rate = 5e-4 # Starting point, modded-nanogpt used different LRs per group
weight_decay = 1e-1
beta1 = 0.9 # modded-nanogpt used 0.8 for Adam part
beta2 = 0.95
grad_clip = 1.0
adam_eps = 1e-10 # From modded-nanogpt
# learning rate decay settings - From modded-nanogpt
decay_lr = True
# warmup_iters = 1000 # Original llama2.c style warmup - replaced by stable period
max_iters  = 100000  # Set max_iters here
min_lr = base_learning_rate / 10.0 # Standard min LR
cooldown_frac = 0.1 # Fraction of training to spend cooling down LR (modded-nanogpt used 0.4, adjust based on max_iters)
# system
device = "cuda"
dtype = "float16"  # bfloat16|float16|float32
compile = True
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read()) # Can still use configurator if needed
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# derived hyperparameters
# Calculate the point where LR starts decaying
stable_iters = int(max_iters * (1.0 - cooldown_frac))

# Calculate padded vocab size for model init
padded_vocab_size = next_multiple_of_n(vocab_size, n=128)
print(f"Original vocab size: {vocab_size}, Padded vocab size: {padded_vocab_size}")

# DDP setup (identical to original llama2-train.txt)
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# data loader
# NOTE: Ensure tinystories.py pretokenizes with the correct vocab_source and vocab_size
# You might need to run: python tinystories.py pretokenize --vocab_size=4096 (or relevant size)
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size, # Use ORIGINAL vocab size for data loading
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

# init these up here
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=padded_vocab_size, # Use PADDED vocab size for model
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
    # Add flags if needed in model.py to enable/disable modded features
    # e.g., use_qk_norm=True, use_relu_squared=True, untie_embeddings=True ...
)
if init_from == "scratch":
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    # Ensure zero-init and untied head happened inside model.py
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # Force necessary args to match checkpoint, others can be overridden
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "multiple_of", "max_seq_len"]:
         model_args[k] = checkpoint_model_args[k]
    # Handle potential vocab size mismatch carefully if resuming
    model_args["vocab_size"] = checkpoint_model_args["vocab_size"] # Use checkpoint's vocab size

    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer - using AdamW with parameter groups inspired by modded-nanogpt
# The model's configure_optimizers method should set this up.
# Example of how model.configure_optimizers might look:
# def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
#     param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
#
#     # Separate params: embeddings, hidden matmuls, norms/scalars, head
#     embed_params = [p for n, p in param_dict.items() if "tok_embeddings" in n]
#     head_params = [p for n, p in param_dict.items() if "output" in n] # lm_head called 'output' here
#     hidden_matrix_params = [p for n, p in param_dict.items() if p.ndim >= 2 and "tok_embeddings" not in n and "output" not in n and "norm" not in n]
#     scalar_norm_params = [p for n, p in param_dict.items() if p.ndim < 2 or "norm" in n] # Includes biases if any, norms
#
#     # Assign different LRs (example values, tune these!)
#     # modded-nanogpt used specific values for its Adam groups: head=0.22, embed=0.6, scalar=0.04 (relative to base?)
#     # Let's use multipliers on the base learning_rate
#     lr_mult_head = 1.0 # Tune these multipliers
#     lr_mult_embed = 1.0
#     lr_mult_hidden = 1.0
#     lr_mult_scalar_norm = 0.1 # Often beneficial to use lower LR for norms/biases
#
#     optim_groups = [
#         {"params": head_params, "lr": learning_rate * lr_mult_head, "weight_decay": 0.0}, # Often disable WD for head/embeddings/norms
#         {"params": embed_params, "lr": learning_rate * lr_mult_embed, "weight_decay": 0.0},
#         {"params": scalar_norm_params, "lr": learning_rate * lr_mult_scalar_norm, "weight_decay": 0.0},
#         {"params": hidden_matrix_params, "lr": learning_rate * lr_mult_hidden, "weight_decay": weight_decay}, # Apply WD only to hidden weights
#     ]
#
#     # Create optimizer
#     # Use fused AdamW if available
#     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
#     use_fused = fused_available and device_type == 'cuda'
#     extra_args = dict(fused=True) if use_fused else dict()
#     optimizer = torch.optim.AdamW(optim_groups, betas=betas, eps=adam_eps, **extra_args) # Use small eps
#     print(f"using fused AdamW: {use_fused}")
#
#     # Store initial LRs
#     for group in optimizer.param_groups:
#         group['initial_lr'] = group['lr']
#
#     return optimizer

# Assume model has configure_optimizers method as described above
optimizer = model.configure_optimizers(weight_decay, base_learning_rate, (beta1, beta2), device_type)

if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None # free up memory

# Ensure initial_lr is stored in each param group for the scheduler
# This handles both scratch training and resuming from checkpoints without initial_lr saved
for group in optimizer.param_groups:
    # Store the current LR as the initial LR if it's not already there.
    # This works because at this point, 'lr' holds the correct starting LR
    # either from configure_optimizers (scratch) or from the loaded checkpoint (resume).
    if 'initial_lr' not in group:
         group['initial_lr'] = group['lr']

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer if it exists and using compile
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y) # Assuming model returns logits now
                # Loss calculation might happen inside model or here
                # If inside model (like original llama2.c):
                loss = raw_model.last_loss
                # If needs to be calculated here (ensure logits have correct shape and use padded size):
                # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (modded-nanogpt style: stable then decay)
@lru_cache(1) # Cache the LR multiplier
def get_lr_multiplier(step: int):
    # 1) linear warmup (Optional, modded-nanogpt didn't use warmup, just stable period)
    # warmup_iters = 100 # Example if you want warmup
    # if step < warmup_iters:
    #     return float(step) / float(max(1, warmup_iters))
    # For stable-then-decay:
    if step < stable_iters:
        return 1.0
    # 2) if it > lr_decay_iters, return min learning rate ratio
    if step > max_iters: # Should equal max_iters
        return min_lr / base_learning_rate # Return ratio relative to base
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - stable_iters) / (max_iters - stable_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    # Final LR = min_lr + coeff * (max_lr - min_lr)
    # We want multiplier relative to initial_lr for each group
    # Return the coefficient that scales (initial_lr - min_lr_for_group)
    # Simpler: return the final LR ratio relative to the base_learning_rate
    final_lr_ratio = (min_lr / base_learning_rate) + coeff * (1.0 - (min_lr / base_learning_rate))
    return final_lr_ratio

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0 # MFU calculation might need adjustment for new arch
while True:
    # determine and set the learning rate for this iteration
    lr_mult = get_lr_multiplier(iter_num) if decay_lr else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["initial_lr"] * lr_mult # Scale the initial LR for the group

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        current_lr = optimizer.param_groups[0]['lr'] # Get LR of first group for logging
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "lr": current_lr,
                        # "mfu": running_mfu * 100, # MFU might be inaccurate
                    }, step = iter_num
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    # Use padded vocab size in saved args
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args, # Already uses padded_vocab_size
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config, # Original config might have unpadded size, model_args is better
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                # Export requires the non-padded vocab size for tokenizer compatibility
                # Temporarily set model args for export if needed, or handle in export.py
                # export_model_args = model_args.copy()
                # export_model_args['vocab_size'] = vocab_size # Pass original size if export needs it
                # model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0, export_model_args)
                # Simpler: Let export.py handle loading checkpoint and extracting original size if possible
                model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)


    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits = model(X, Y) # Assume loss calculation is inside model
            loss = raw_model.last_loss # Get loss from model attribute
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps # loss as float, scale up
        current_lr = optimizer.param_groups[0]['lr'] # Get LR for logging
        # if local_iter_num >= 5: # let the training loop settle a bit
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            # f"{iter_num} | loss {lossf:.4f} | lr {current_lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            f"{iter_num} | loss {lossf:.4f} | lr {current_lr:e} | {dt*1000:.2f}ms" # Simpler log without MFU
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
