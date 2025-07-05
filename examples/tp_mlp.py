"""
Tensor-parallel MLP example.

Demonstrates the standard Megatron-style MLP sharding:
  - gate_proj and up_proj are column-parallel (gather_output=False)
  - down_proj is row-parallel (consumes the split activations)

This gives two all-reduces per transformer block (one in attention,
one in MLP) instead of doing the full computation on every rank.

Usage:
    torchrun --nproc_per_node=2 examples/tp_mlp.py
"""

import torch
import torch.nn as nn
from tensor_parallel.utils import init_distributed, get_tp_rank
from tensor_parallel.column_parallel import ColumnParallelLinear
from tensor_parallel.row_parallel import RowParallelLinear


class TensorParallelMLP(nn.Module):
    """SwiGLU-style MLP with tensor parallelism (Llama/Mistral style)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # gate and up projections are column-parallel
        # they output split activations (gather_output=False)
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        # down projection is row-parallel
        # it consumes split input and produces all-reduced output
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True
        )

    def forward(self, x):
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def main():
    init_distributed()
    rank = get_tp_rank()

    hidden = 4096
    intermediate = 11008  # llama-style

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = TensorParallelMLP(hidden, intermediate).to(device)

    # count parameters per rank
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[rank {rank}] params: {total_params:,} "
          f"(~{total_params * 2 / 1e6:.1f} MB in fp16)")

    # test forward
    x = torch.randn(4, 128, hidden, device=device)
    out = model(x)
    print(f"[rank {rank}] input: {x.shape} -> output: {out.shape}")

    # test backward
    loss = out.sum()
    loss.backward()
    print(f"[rank {rank}] backward pass OK")


if __name__ == "__main__":
    main()
