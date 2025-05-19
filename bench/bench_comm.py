"""
Benchmark all-reduce latency and bandwidth across TP group.

Measures how long NCCL all-reduce takes for different tensor sizes,
which is the key bottleneck in tensor parallelism — every transformer
block does 2 all-reduces per forward pass.

Usage:
    torchrun --nproc_per_node=2 bench/bench_comm.py
"""

import torch
import torch.distributed as dist
import os
import time
from tensor_parallel.utils import init_distributed, get_tp_rank, get_tp_group, get_tp_world_size


def bench_all_reduce():
    sizes_mb = [0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    rank = get_tp_rank()
    group = get_tp_group()

    if rank == 0:
        print("=" * 65)
        print(f"All-Reduce Benchmark (TP size = {get_tp_world_size()})")
        print(f"{'Size (MB)':<12} {'Latency (us)':<15} {'Bandwidth (GB/s)':<18} {'Algo BW (GB/s)':<15}")
        print("-" * 65)

    for size_mb in sizes_mb:
        num_elements = int(size_mb * 1e6 / 2)  # fp16 = 2 bytes
        tensor = torch.randn(num_elements, device="cuda", dtype=torch.float16)

        # warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize()

        # benchmark
        n_iters = 50
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iters

        latency_us = elapsed * 1e6
        size_bytes = num_elements * 2
        bandwidth_gbps = size_bytes / elapsed / 1e9
        # algorithm bandwidth: accounts for the fact that all-reduce
        # transfers 2*(n-1)/n * data for ring all-reduce
        tp_size = get_tp_world_size()
        algo_bw = bandwidth_gbps * 2 * (tp_size - 1) / tp_size

        if rank == 0:
            print(f"{size_mb:<12.2f} {latency_us:<15.1f} {bandwidth_gbps:<18.2f} {algo_bw:<15.2f}")

    if rank == 0:
        print()


def bench_all_gather():
    sizes_mb = [0.1, 1.0, 4.0, 16.0, 64.0]
    rank = get_tp_rank()
    group = get_tp_group()
    tp_size = get_tp_world_size()

    if rank == 0:
        print("=" * 65)
        print(f"All-Gather Benchmark (TP size = {tp_size})")
        print(f"{'Size/rank (MB)':<16} {'Latency (us)':<15} {'Bandwidth (GB/s)':<18}")
        print("-" * 65)

    for size_mb in sizes_mb:
        num_elements = int(size_mb * 1e6 / 2)
        tensor = torch.randn(num_elements, device="cuda", dtype=torch.float16)
        output = [torch.empty_like(tensor) for _ in range(tp_size)]

        # warmup
        for _ in range(5):
            dist.all_gather(output, tensor, group=group)
        torch.cuda.synchronize()

        n_iters = 50
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            dist.all_gather(output, tensor, group=group)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iters

        latency_us = elapsed * 1e6
        total_bytes = num_elements * 2 * tp_size
        bandwidth_gbps = total_bytes / elapsed / 1e9

        if rank == 0:
            print(f"{size_mb:<16.2f} {latency_us:<15.1f} {bandwidth_gbps:<18.2f}")

    if rank == 0:
        print()


def main():
    init_distributed()
    bench_all_reduce()
    bench_all_gather()


if __name__ == "__main__":
    main()
