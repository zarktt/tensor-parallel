"""
Pipeline parallelism example.

Splits a simple feedforward model into stages and runs the 1F1B schedule.

Usage:
    torchrun --nproc_per_node=2 examples/pipeline_example.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from tensor_parallel.utils import init_distributed
from tensor_parallel.pipeline import PipelineStage, run_1f1b_schedule


class Stage0(nn.Module):
    """First half of the model."""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


class Stage1(nn.Module):
    """Second half of the model."""
    def __init__(self, hidden, out_dim):
        super().__init__()
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc4(torch.relu(self.fc3(x)))


def main():
    init_distributed(tp_size=1, backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cpu")

    hidden = 256
    in_dim = 128
    out_dim = 10
    batch_size = 8
    num_micro = world_size  # one micro-batch per stage

    if rank == 0:
        module = Stage0(in_dim, hidden)
    else:
        module = Stage1(hidden, out_dim)

    stage = PipelineStage(module, rank, world_size, device)

    # create micro-batches (only stage 0 uses inputs, only last uses labels)
    torch.manual_seed(42)
    micro_batches = [torch.randn(batch_size // num_micro, in_dim) for _ in range(num_micro)]
    labels = [torch.randint(0, out_dim, (batch_size // num_micro,)) for _ in range(num_micro)]

    losses = run_1f1b_schedule(
        stage, micro_batches,
        loss_fn=nn.CrossEntropyLoss(),
        labels=labels,
    )

    if rank == world_size - 1:
        total_loss = sum(l.item() for l in losses)
        print(f"[rank {rank}] total loss: {total_loss:.4f}")

    print(f"[rank {rank}] pipeline example done")


if __name__ == "__main__":
    main()
