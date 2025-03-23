"""
Row-parallel linear layer.

Splits the weight matrix along the input dimension (rows). Each rank
holds a chunk of the input and computes a partial output, then all-reduce
combines the partial results.

For Y = XW + b with W of shape (in_features, out_features):
  - Each rank holds W_i of shape (in_features // tp_size, out_features)
  - Each rank receives X_i (a split of X along the last dim)
  - Each rank computes Y_i = X_i @ W_i (partial result)
  - All-reduce sum to get Y = sum(Y_i)

This is the "second half" of the Megatron TP pattern. It pairs with
ColumnParallelLinear(gather_output=False) — the column-parallel layer
outputs split activations, and this layer consumes them.

        X_0   X_1   X_2   X_3     <- split input (from column-parallel)
         |     |     |     |
        [W_0] [W_1] [W_2] [W_3]   <- each rank holds one row shard
         |     |     |     |
        Y_0   Y_1   Y_2   Y_3     <- partial outputs
              \\|/
           all-reduce
              |
              Y (full output)
"""

import torch
import torch.nn as nn
from tensor_parallel.utils import get_tp_rank, get_tp_world_size
from tensor_parallel.comm import all_reduce


class RowParallelLinear(nn.Module):
    """Linear layer with weight split along the input dimension.

    Args:
        in_features: total input size (will be divided by tp_size)
        out_features: output size
        bias: whether to include bias (only rank 0 holds the bias to
            avoid double-counting after all-reduce)
        input_is_parallel: if True, assumes input is already split
            across ranks. If False, splits it internally.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()

        assert in_features % tp_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_rank = in_features // tp_size
        self.input_is_parallel = input_is_parallel
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        # each rank holds its row slice
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank)
        )
        # only rank 0 holds the bias to avoid double-counting
        if bias and tp_rank == 0:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def load_full_weight(self, full_weight: torch.Tensor, full_bias: torch.Tensor = None):
        """Load from a non-parallelized weight. Splits along dim 1 (input features)."""
        chunks = full_weight.chunk(self.tp_size, dim=1)
        self.weight.data.copy_(chunks[self.tp_rank])
        if full_bias is not None and self.bias is not None:
            self.bias.data.copy_(full_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # split x along last dim
            chunks = x.chunk(self.tp_size, dim=-1)
            x = chunks[self.tp_rank]

        # partial matmul
        out = torch.nn.functional.linear(x, self.weight)

        # all-reduce to sum partial results
        out = all_reduce(out)

        # add bias (only rank 0 has it, but after all-reduce all ranks
        # have the same output, so adding on rank 0 and broadcasting
        # would be equivalent. since all ranks see the same all-reduced
        # output, we can just add on all ranks if we have it)
        if self.bias is not None:
            out = out + self.bias

        return out
