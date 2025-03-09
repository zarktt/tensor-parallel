"""
Distributed setup utilities for tensor parallelism.

Handles process group initialization and provides convenience functions
for getting rank/world_size within the tensor-parallel group.

Usage:
    # In each process:
    init_distributed(tp_size=4)
    rank = get_tp_rank()
"""

import os
import torch
import torch.distributed as dist
from typing import Optional

# global state for the TP process group
_TP_GROUP: Optional[dist.ProcessGroup] = None
_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1


def init_distributed(
    tp_size: int = -1,
    backend: str = "nccl",
):
    """
    Initialize distributed and create the tensor-parallel process group.

    Args:
        tp_size: tensor parallel degree. If -1, uses all GPUs.
        backend: torch.distributed backend ('nccl' for GPU, 'gloo' for CPU testing)
    """
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if tp_size == -1:
        tp_size = world_size

    assert world_size % tp_size == 0, (
        f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
    )

    # create TP groups: ranks [0,1,...,tp_size-1], [tp_size,...,2*tp_size-1], etc.
    num_tp_groups = world_size // tp_size
    for i in range(num_tp_groups):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TP_GROUP = group
            _TP_RANK = ranks.index(rank)
            _TP_WORLD_SIZE = tp_size

    # set device for this rank
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def get_tp_group() -> dist.ProcessGroup:
    """Get the tensor-parallel process group."""
    assert _TP_GROUP is not None, "Call init_distributed() first"
    return _TP_GROUP


def get_tp_rank() -> int:
    """Get this process's rank within the TP group."""
    return _TP_RANK


def get_tp_world_size() -> int:
    """Get the TP group size (number of devices for tensor parallelism)."""
    return _TP_WORLD_SIZE
