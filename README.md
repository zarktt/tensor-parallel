# tensor-parallel

Minimal tensor parallelism and pipeline parallelism implementation from scratch.

Learning project to understand how model parallelism works under the hood —
NCCL collectives, column/row-parallel linear layers, and pipeline schedules.

## Status

Early WIP. Starting with basic TP layers over `torch.distributed`.

## Setup

```bash
pip install -e .
```

Requires PyTorch 2.1+ with NCCL support.
