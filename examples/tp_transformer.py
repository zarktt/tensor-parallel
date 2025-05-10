"""
Full transformer block with tensor parallelism.

Shows how to parallelize an entire transformer layer:
  - Self-attention: Q/K/V projections are column-parallel,
    output projection is row-parallel
  - MLP: gate/up are column-parallel, down is row-parallel
  - RMSNorm and residual connections are replicated

This is the standard Megatron-LM approach. Each transformer block
has exactly 2 all-reduce ops: one in attention, one in MLP.

Usage:
    torchrun --nproc_per_node=2 examples/tp_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_parallel.utils import init_distributed, get_tp_rank, get_tp_world_size
from tensor_parallel.column_parallel import ColumnParallelLinear
from tensor_parallel.row_parallel import RowParallelLinear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (x * self.weight).to(x.dtype)


class TPAttention(nn.Module):
    """Multi-head attention with tensor-parallel Q/K/V and output projections."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        tp_size = get_tp_world_size()
        assert num_heads % tp_size == 0, "num_heads must be divisible by tp_size"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_heads_per_rank = num_heads // tp_size

        # Q, K, V are column-parallel (split across heads)
        self.q_proj = ColumnParallelLinear(
            hidden_size, hidden_size, bias=False, gather_output=False
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, hidden_size, bias=False, gather_output=False
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, hidden_size, bias=False, gather_output=False
        )
        # output is row-parallel (all-reduce at the end)
        self.o_proj = RowParallelLinear(
            hidden_size, hidden_size, bias=False, input_is_parallel=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # each rank computes its local heads
        q = self.q_proj(x).view(B, S, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads_per_rank, self.head_dim).transpose(1, 2)

        # standard scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # reshape back: (B, local_heads, S, D) -> (B, S, local_heads * D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)

        # row-parallel output projection -> all-reduce inside
        return self.o_proj(attn_out)


class TPMLP(nn.Module):
    """SwiGLU MLP with tensor parallelism."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TPTransformerBlock(nn.Module):
    """Single transformer block with tensor parallelism."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = TPAttention(hidden_size, num_heads)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = TPMLP(hidden_size, intermediate_size)

    def forward(self, x):
        # pre-norm architecture
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


def main():
    init_distributed()
    rank = get_tp_rank()
    tp_size = get_tp_world_size()

    # Llama-3 8B-ish dimensions
    hidden = 4096
    heads = 32
    intermediate = 11008

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    block = TPTransformerBlock(hidden, heads, intermediate).to(device)

    total_params = sum(p.numel() for p in block.parameters())
    print(f"[rank {rank}/{tp_size}] transformer block params: {total_params:,} "
          f"(full model would have {total_params * tp_size:,})")

    # forward + backward
    x = torch.randn(2, 64, hidden, device=device)
    out = block(x)
    loss = out.sum()
    loss.backward()

    print(f"[rank {rank}] forward: {x.shape} -> {out.shape}")
    print(f"[rank {rank}] backward OK, grad norm: "
          f"{sum(p.grad.norm().item() for p in block.parameters() if p.grad is not None):.4f}")


if __name__ == "__main__":
    main()
