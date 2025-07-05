"""
Sequence parallelism for LayerNorm/RMSNorm and Dropout.

In standard tensor parallelism, the LayerNorm and Dropout are replicated
across all ranks — each rank computes the exact same thing, wasting compute
and memory.

Sequence parallelism (Korthikanti et al., 2022) fixes this by splitting
these operations along the sequence dimension. Instead of replicating:

    Standard TP:
        all-reduce -> LayerNorm (replicated) -> ColumnParallel
        RowParallel -> all-reduce -> LayerNorm (replicated) -> ...

    With Sequence Parallelism:
        reduce-scatter -> LayerNorm (split by seq) -> all-gather -> ColumnParallel
        RowParallel -> reduce-scatter -> LayerNorm (split by seq) -> all-gather -> ...

The all-reduce is replaced by reduce-scatter + all-gather, and the norm/dropout
in between operates on 1/tp_size of the sequence.

Benefits:
    - Reduces activation memory by tp_size for norm and dropout
    - Same total communication volume as all-reduce
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from tensor_parallel.utils import get_tp_group, get_tp_world_size


class _ScatterToSequenceParallel(torch.autograd.Function):
    """reduce-scatter in forward (to scatter activations along seq dim),
    all-gather in backward."""

    @staticmethod
    def forward(ctx, x):
        tp_size = get_tp_world_size()
        if tp_size == 1:
            return x

        # scatter along sequence dimension (dim 1 for [B, S, D])
        seq_len = x.shape[1]
        assert seq_len % tp_size == 0

        # reduce-scatter: sum partial results and scatter
        chunk_size = seq_len // tp_size
        output = torch.empty(
            x.shape[0], chunk_size, x.shape[2],
            dtype=x.dtype, device=x.device
        )
        input_list = list(x.chunk(tp_size, dim=1))
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=get_tp_group())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tp_size = get_tp_world_size()
        if tp_size == 1:
            return grad_output

        # all-gather in backward
        gathered = [torch.empty_like(grad_output) for _ in range(tp_size)]
        dist.all_gather(gathered, grad_output.contiguous(), group=get_tp_group())
        return torch.cat(gathered, dim=1)


class _GatherFromSequenceParallel(torch.autograd.Function):
    """all-gather in forward (to reconstruct full sequence),
    reduce-scatter in backward."""

    @staticmethod
    def forward(ctx, x):
        tp_size = get_tp_world_size()
        if tp_size == 1:
            return x

        gathered = [torch.empty_like(x) for _ in range(tp_size)]
        dist.all_gather(gathered, x.contiguous(), group=get_tp_group())
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        tp_size = get_tp_world_size()
        if tp_size == 1:
            return grad_output

        seq_len = grad_output.shape[1]
        chunk_size = seq_len // tp_size
        output = torch.empty(
            grad_output.shape[0], chunk_size, grad_output.shape[2],
            dtype=grad_output.dtype, device=grad_output.device,
        )
        input_list = list(grad_output.chunk(tp_size, dim=1))
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=get_tp_group())
        return output


def scatter_to_sp(x: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter activations to enter sequence-parallel region."""
    return _ScatterToSequenceParallel.apply(x)


def gather_from_sp(x: torch.Tensor) -> torch.Tensor:
    """All-gather activations to exit sequence-parallel region."""
    return _GatherFromSequenceParallel.apply(x)


class SequenceParallelRMSNorm(nn.Module):
    """RMSNorm that operates on sequence-parallel (split) activations.

    Input is (B, S/tp_size, D), output is (B, S/tp_size, D).
    Each rank normalizes its chunk of the sequence independently.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (x * self.weight).to(x.dtype)
