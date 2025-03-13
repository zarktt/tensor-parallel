"""
Communication primitives for tensor parallelism.

These are thin wrappers around torch.distributed that operate on the TP
process group and handle the autograd integration (so gradients flow
through all-reduce and all-gather correctly).

The key insight for TP is:
  - Forward: all-reduce or all-gather to combine partial results
  - Backward: the conjugate operation to split gradients

Reference: Megatron-LM (Shoeybi et al., 2019)
"""

import torch
import torch.distributed as dist
from tensor_parallel.utils import get_tp_group, get_tp_world_size, get_tp_rank


class _AllReduceFunc(torch.autograd.Function):
    """All-reduce in forward, identity in backward."""

    @staticmethod
    def forward(ctx, x):
        if get_tp_world_size() == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=get_tp_group())
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # identity — each rank gets the full gradient
        return grad_output


class _AllReduceBackwardFunc(torch.autograd.Function):
    """Identity in forward, all-reduce in backward.

    Used before column-parallel layers so that gradients are reduced
    before being sent back.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if get_tp_world_size() == 1:
            return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=get_tp_group())
        return grad_output


class _AllGatherFunc(torch.autograd.Function):
    """All-gather in forward, reduce-scatter in backward."""

    @staticmethod
    def forward(ctx, x):
        tp_size = get_tp_world_size()
        if tp_size == 1:
            return x

        # gather along the last dimension
        gathered = [torch.empty_like(x) for _ in range(tp_size)]
        dist.all_gather(gathered, x, group=get_tp_group())
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        tp_size = get_tp_world_size()
        if tp_size == 1:
            return grad_output

        # reduce-scatter: split grad and sum across ranks
        # equivalent to: each rank gets the sum of its chunk across all ranks
        chunks = grad_output.chunk(tp_size, dim=-1)
        my_chunk = chunks[get_tp_rank()].contiguous()
        dist.all_reduce(my_chunk, op=dist.ReduceOp.SUM, group=get_tp_group())
        return my_chunk


class _ReduceScatterFunc(torch.autograd.Function):
    """Reduce-scatter in forward, all-gather in backward."""

    @staticmethod
    def forward(ctx, x):
        tp_size = get_tp_world_size()
        if tp_size == 1:
            return x

        # split along last dim, reduce, each rank gets its chunk
        chunks = list(x.chunk(tp_size, dim=-1))
        output = torch.empty_like(chunks[0])
        dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM, group=get_tp_group())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tp_size = get_tp_world_size()
        if tp_size == 1:
            return grad_output

        gathered = [torch.empty_like(grad_output) for _ in range(tp_size)]
        dist.all_gather(gathered, grad_output, group=get_tp_group())
        return torch.cat(gathered, dim=-1)


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """All-reduce sum in forward pass, identity in backward."""
    return _AllReduceFunc.apply(x)


def all_reduce_backward(x: torch.Tensor) -> torch.Tensor:
    """Identity in forward, all-reduce sum in backward."""
    return _AllReduceBackwardFunc.apply(x)


def all_gather(x: torch.Tensor) -> torch.Tensor:
    """All-gather along last dim in forward, reduce-scatter in backward."""
    return _AllGatherFunc.apply(x)


def reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter along last dim in forward, all-gather in backward."""
    return _ReduceScatterFunc.apply(x)
