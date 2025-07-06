# tensor-parallel

Minimal tensor parallelism, pipeline parallelism, and sequence parallelism from scratch over `torch.distributed` + NCCL.

Built to understand how model parallelism actually works under the hood — no magic, just explicit collectives and autograd functions.

## What's here

### Tensor Parallelism (Megatron-style)
- **`ColumnParallelLinear`** — Splits weight along output dim. Each rank computes a slice of the output, optionally all-gathers.
- **`RowParallelLinear`** — Splits weight along input dim. Consumes split input, all-reduces the output.
- **`VocabParallelEmbedding`** — Splits the embedding table across ranks. Each rank handles a vocabulary range.

### Communication
- **`all_reduce`** / **`all_gather`** / **`reduce_scatter`** — Autograd-aware wrappers that handle forward/backward pairing correctly.
- **`AsyncAllReduce`** — Async all-reduce for overlapping communication with compute.

### Sequence Parallelism
- Replaces all-reduce with reduce-scatter + all-gather to split norm/dropout along sequence dim.
- Reduces activation memory by `tp_size` in non-parallel regions.

### Pipeline Parallelism
- **1F1B schedule** — Warmup → steady-state (1 forward, 1 backward) → cooldown.
- Minimizes pipeline bubble vs GPipe's all-forward-then-all-backward.

### Utilities
- **`weight_loader`** — Converts HuggingFace checkpoints to TP-sharded weights.

## Architecture

The key idea from Megatron-LM is that a transformer block needs only **2 all-reduces** for full tensor parallelism:

```
Input (replicated)
  |
  ├── Attention:
  │     Column-Parallel Q/K/V → local SDPA → Row-Parallel output → [all-reduce]
  │
  ├── Residual + RMSNorm (replicated, or seq-parallel)
  │
  ├── MLP:
  │     Column-Parallel gate/up → SwiGLU → Row-Parallel down → [all-reduce]
  │
  └── Residual + RMSNorm
  |
Output (replicated)
```

## Benchmark Results (2x A100 NVLink)

### All-Reduce Latency
| Size (MB) | Latency (µs) | Bandwidth (GB/s) | Algo BW (GB/s) |
|-----------|-------------|-------------------|-----------------|
| 0.01      | 8.2         | 1.2               | 1.2             |
| 0.1       | 11.4        | 8.8               | 8.8             |
| 1.0       | 28.3        | 35.3              | 35.3            |
| 4.0       | 68.1        | 58.7              | 58.7            |
| 16.0      | 217.5       | 73.6              | 73.6            |
| 64.0      | 812.3       | 78.8              | 78.8            |

Approaching ~79 GB/s bus bandwidth at large sizes (NVLink theoretical max is ~300 GB/s bidirectional per link).

## Examples

```bash
# Tensor-parallel MLP
torchrun --nproc_per_node=2 examples/tp_mlp.py

# Full transformer block
torchrun --nproc_per_node=2 examples/tp_transformer.py

# Pipeline parallelism
torchrun --nproc_per_node=2 examples/pipeline_example.py
```

## Setup

```bash
pip install -e ".[dev]"
```

Requires PyTorch 2.1+ with NCCL. For CPU testing, gloo backend is used.

## Tests

```bash
python -m pytest tests/ -v
```

## References

- [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019
- [Reducing Activation Recomputation (Sequence Parallelism)](https://arxiv.org/abs/2205.05198) — Korthikanti et al., 2022
- [PipeDream](https://arxiv.org/abs/1806.03377) — Narayanan et al., 2019
- [GPipe](https://arxiv.org/abs/1811.06965) — Huang et al., 2019
