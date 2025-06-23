"""
Weight loading utilities for converting HuggingFace model checkpoints
to tensor-parallel sharded weights.

Given a HF model state_dict, this module splits the weights across TP
ranks according to the Megatron-style sharding convention:
  - QKV projections: split along output dim (column-parallel)
  - Output projections: split along input dim (row-parallel)
  - MLP gate/up: split along output dim
  - MLP down: split along input dim
  - Embeddings: split along vocab dim
  - Norms: replicated
"""

import torch
from typing import Dict
from tensor_parallel.utils import get_tp_rank, get_tp_world_size


def shard_column(weight: torch.Tensor, bias=None) -> tuple:
    """Split weight along dim 0 (output features) for column-parallel."""
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()
    chunks = weight.chunk(tp_size, dim=0)
    w = chunks[tp_rank].clone()
    b = None
    if bias is not None:
        b = bias.chunk(tp_size, dim=0)[tp_rank].clone()
    return w, b


def shard_row(weight: torch.Tensor) -> torch.Tensor:
    """Split weight along dim 1 (input features) for row-parallel."""
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()
    chunks = weight.chunk(tp_size, dim=1)
    return chunks[tp_rank].clone()


def shard_embedding(weight: torch.Tensor) -> torch.Tensor:
    """Split embedding table along vocab dim."""
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()
    vocab_size = weight.shape[0]
    per_rank = vocab_size // tp_size
    start = tp_rank * per_rank
    end = start + per_rank
    return weight[start:end].clone()


def shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    column_parallel_keys: list,
    row_parallel_keys: list,
    embedding_keys: list = None,
) -> Dict[str, torch.Tensor]:
    """
    Shard a full state_dict for tensor parallelism.

    Args:
        state_dict: full model state dict
        column_parallel_keys: list of weight name patterns to split as column-parallel
        row_parallel_keys: list of weight name patterns to split as row-parallel
        embedding_keys: list of embedding weight name patterns

    Returns:
        Sharded state dict for this rank
    """
    embedding_keys = embedding_keys or []
    sharded = {}

    for key, param in state_dict.items():
        is_col = any(k in key for k in column_parallel_keys)
        is_row = any(k in key for k in row_parallel_keys)
        is_emb = any(k in key for k in embedding_keys)

        if is_col:
            if "bias" in key:
                _, b = shard_column(param.unsqueeze(0), param)
                sharded[key] = b
            else:
                w, _ = shard_column(param)
                sharded[key] = w
        elif is_row:
            sharded[key] = shard_row(param)
        elif is_emb:
            sharded[key] = shard_embedding(param)
        else:
            # replicate (norms, etc.)
            sharded[key] = param.clone()

    return sharded
