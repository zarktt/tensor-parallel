"""
Vocabulary-parallel embedding.

For large-vocab models (32k-128k tokens), the embedding table is huge.
This splits it across TP ranks so each rank holds vocab_size // tp_size
rows of the embedding table.

During forward:
  1. Each rank looks up only the tokens that fall in its range
  2. Tokens outside its range get zero embeddings
  3. All-reduce to combine (since each token is in exactly one rank's range)
"""

import torch
import torch.nn as nn
from tensor_parallel.utils import get_tp_rank, get_tp_world_size
from tensor_parallel.comm import all_reduce


class VocabParallelEmbedding(nn.Module):
    """Embedding layer split across TP ranks along the vocabulary dimension.

    Args:
        num_embeddings: total vocabulary size
        embedding_dim: embedding dimension (not split)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()

        assert num_embeddings % tp_size == 0, (
            f"vocab_size ({num_embeddings}) must be divisible by tp_size ({tp_size})"
        )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.vocab_per_rank = num_embeddings // tp_size

        # range of vocab indices this rank is responsible for
        self.vocab_start = tp_rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank

        self.weight = nn.Parameter(
            torch.empty(self.vocab_per_rank, embedding_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def load_full_weight(self, full_weight: torch.Tensor):
        """Load from a non-parallelized embedding weight."""
        self.weight.data.copy_(
            full_weight[self.vocab_start:self.vocab_end]
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # mask out tokens not in our range
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        # shift indices to local range
        local_ids = input_ids - self.vocab_start
        local_ids = local_ids.clamp(0, self.vocab_per_rank - 1)  # clamp OOB

        # lookup
        output = torch.nn.functional.embedding(local_ids, self.weight)
        # zero out positions that weren't in our range
        output = output * mask.unsqueeze(-1).float()

        # all-reduce to combine — each token is only non-zero on one rank
        output = all_reduce(output)
        return output
