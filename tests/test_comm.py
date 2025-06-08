"""
Tests for communication primitives.

Spawns processes with gloo backend to test that autograd-aware
all-reduce / all-gather / reduce-scatter work correctly.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    from tensor_parallel.utils import init_distributed
    init_distributed(tp_size=world_size, backend="gloo")


def _run_all_reduce_test(rank, world_size):
    _setup(rank, world_size)
    from tensor_parallel.comm import all_reduce

    # each rank creates a tensor with its rank value
    x = torch.full((4, 8), float(rank), requires_grad=True)

    out = all_reduce(x)

    # all-reduce sum: should be 0 + 1 = 1 for world_size=2
    expected = sum(range(world_size))
    if not torch.allclose(out, torch.full_like(out, float(expected))):
        raise AssertionError(f"rank {rank}: all_reduce output wrong")

    # test backward
    out.sum().backward()
    # gradient should be identity (no reduction in backward for all_reduce)
    if not torch.allclose(x.grad, torch.ones_like(x)):
        raise AssertionError(f"rank {rank}: all_reduce backward wrong")

    print(f"[rank {rank}] all_reduce: PASS")
    dist.destroy_process_group()


def _run_all_reduce_backward_test(rank, world_size):
    _setup(rank, world_size)
    from tensor_parallel.comm import all_reduce_backward

    x = torch.full((4, 8), float(rank), requires_grad=True)
    out = all_reduce_backward(x)

    # forward should be identity
    if not torch.allclose(out, x):
        raise AssertionError(f"rank {rank}: all_reduce_backward forward wrong")

    # backward should all-reduce gradients
    loss = out.sum()
    loss.backward()

    expected_grad = float(world_size)  # sum of all 1s
    if not torch.allclose(x.grad, torch.full_like(x, expected_grad)):
        raise AssertionError(f"rank {rank}: all_reduce_backward backward wrong")

    print(f"[rank {rank}] all_reduce_backward: PASS")
    dist.destroy_process_group()


def test_all_reduce():
    mp.spawn(_run_all_reduce_test, args=(2,), nprocs=2)


def test_all_reduce_backward():
    mp.spawn(_run_all_reduce_backward_test, args=(2,), nprocs=2)


if __name__ == "__main__":
    test_all_reduce()
    test_all_reduce_backward()
    print("All comm tests passed!")
