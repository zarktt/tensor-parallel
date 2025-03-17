"""
Column-parallel linear layer.

Splits the weight matrix along the output dimension (columns), so each
TP rank computes a slice of the output. The full output is assembled
via all-gather (or used directly if the next layer is row-parallel).

For a linear layer Y = XW + b with W of shape (in_features, out_features):
  - Each rank holds W_i of shape (in_features, out_features // tp_size)
  - Each rank computes Y_i = X @ W_i (partial output)
  - If gather_output=True, all-gather Y_i to get full Y

This is the "first half" of the Megatron-LM TP pattern. The second half
is RowParallelLinear, which takes split input and produces reduced output.

            X (replicated)
            |
     [W_0] [W_1] [W_2] [W_3]    <- each rank holds one column shard
            |
     [Y_0] [Y_1] [Y_2] [Y_3]    <- partial outputs
            |
        all-gather (optional)
            |
            Y (full output)
"""

import torch
import torch.nn as nn
from tensor_parallel.utils import get_tp_rank, get_tp_world_size
from tensor_parallel.comm import all_gather, all_reduce_backward


class ColumnParallelLinear(nn.Module):
    """Linear layer with weight split along the output dimension.

    Args:
        in_features: input size
        out_features: total output size (will be divided by tp_size)
        bias: whether to include a bias term
        gather_output: if True, all-gather the output across ranks.
            Set to False if the next layer is RowParallelLinear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()

        assert out_features % tp_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // tp_size
        self.gather_output = gather_output
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        # each rank only holds its column slice
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # init as if it were the full weight, then we only keep our slice
        # this ensures each rank's init is independent but has the right scale
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def load_full_weight(self, full_weight: torch.Tensor, full_bias: torch.Tensor = None):
        """Load from a non-parallelized linear layer's weight.

        Splits the weight along dim 0 (output features) and assigns the
        appropriate shard to this rank.
        """
        chunks = full_weight.chunk(self.tp_size, dim=0)
        self.weight.data.copy_(chunks[self.tp_rank])
        if full_bias is not None and self.bias is not None:
            bias_chunks = full_bias.chunk(self.tp_size, dim=0)
            self.bias.data.copy_(bias_chunks[self.tp_rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x is replicated across all ranks.
        # we need all-reduce in backward for the input gradient.
        x = all_reduce_backward(x)

        # each rank computes its column slice
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.gather_output:
            out = all_gather(out)

        return out
