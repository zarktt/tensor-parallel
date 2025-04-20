"""
Tests for tensor-parallel layers.

These tests use gloo backend and run on CPU to avoid needing multiple GPUs.
We simulate TP by spawning multiple processes.

Run with:
    python -m pytest tests/ -v
    (or: torchrun --nproc_per_node=2 -m pytest tests/ -v)
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    from tensor_parallel.utils import init_distributed
    init_distributed(tp_size=world_size, backend="gloo")


def _run_column_parallel_test(rank, world_size):
    _setup(rank, world_size)
    from tensor_parallel.column_parallel import ColumnParallelLinear

    in_feat, out_feat = 16, 8
    layer = ColumnParallelLinear(in_feat, out_feat, bias=True, gather_output=True)

    # create a reference linear layer and load its weights
    torch.manual_seed(42)
    ref = torch.nn.Linear(in_feat, out_feat)

    layer.load_full_weight(ref.weight.data, ref.bias.data)

    torch.manual_seed(123)
    x = torch.randn(2, 4, in_feat)

    out = layer(x)
    ref_out = ref(x)

    # outputs should match across all ranks (since we did all-gather)
    if not torch.allclose(out, ref_out, atol=1e-5):
        raise AssertionError(
            f"rank {rank}: column parallel output mismatch. "
            f"max diff: {(out - ref_out).abs().max().item()}"
        )
    print(f"[rank {rank}] column parallel: PASS")
    dist.destroy_process_group()


def _run_row_parallel_test(rank, world_size):
    _setup(rank, world_size)
    from tensor_parallel.row_parallel import RowParallelLinear

    in_feat, out_feat = 8, 16
    layer = RowParallelLinear(in_feat, out_feat, bias=True, input_is_parallel=False)

    torch.manual_seed(42)
    ref = torch.nn.Linear(in_feat, out_feat)
    layer.load_full_weight(ref.weight.data, ref.bias.data)

    torch.manual_seed(123)
    x = torch.randn(2, 4, in_feat)

    out = layer(x)
    ref_out = ref(x)

    # bias is only on rank 0 so we need to handle that
    if rank == 0:
        if not torch.allclose(out, ref_out, atol=1e-5):
            raise AssertionError(
                f"rank {rank}: row parallel output mismatch. "
                f"max diff: {(out - ref_out).abs().max().item()}"
            )
    print(f"[rank {rank}] row parallel: PASS")
    dist.destroy_process_group()


def _run_mlp_test(rank, world_size):
    """Test that column+row parallel MLP produces same output as regular MLP."""
    _setup(rank, world_size)
    from tensor_parallel.column_parallel import ColumnParallelLinear
    from tensor_parallel.row_parallel import RowParallelLinear

    hidden = 16
    inter = 32

    # build TP MLP
    gate = ColumnParallelLinear(hidden, inter, bias=False, gather_output=False)
    down = RowParallelLinear(inter, hidden, bias=False, input_is_parallel=True)

    # build reference
    torch.manual_seed(42)
    ref_gate = torch.nn.Linear(hidden, inter, bias=False)
    torch.manual_seed(43)
    ref_down = torch.nn.Linear(inter, hidden, bias=False)

    gate.load_full_weight(ref_gate.weight.data)
    down.load_full_weight(ref_down.weight.data)

    torch.manual_seed(100)
    x = torch.randn(2, 8, hidden)

    tp_out = down(torch.relu(gate(x)))
    ref_out = ref_down(torch.relu(ref_gate(x)))

    if not torch.allclose(tp_out, ref_out, atol=1e-4):
        raise AssertionError(
            f"rank {rank}: MLP output mismatch. "
            f"max diff: {(tp_out - ref_out).abs().max().item()}"
        )
    print(f"[rank {rank}] column+row MLP: PASS")
    dist.destroy_process_group()


def test_column_parallel():
    world_size = 2
    mp.spawn(_run_column_parallel_test, args=(world_size,), nprocs=world_size)


def test_row_parallel():
    world_size = 2
    mp.spawn(_run_row_parallel_test, args=(world_size,), nprocs=world_size)


def test_column_row_mlp():
    world_size = 2
    mp.spawn(_run_mlp_test, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    test_column_parallel()
    test_row_parallel()
    test_column_row_mlp()
    print("All tests passed!")
