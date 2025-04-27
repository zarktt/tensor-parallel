"""
Pipeline parallelism with 1F1B schedule.

Splits a model into stages, one per device. Uses micro-batching with
the 1F1B (one forward, one backward) schedule to keep all devices busy.

The 1F1B schedule works like this for 4 stages and 4 micro-batches:

  Time ->
  Stage 0: F0  F1  F2  F3  B3  B2  B1  B0
  Stage 1:     F0  F1  F2  F3  B3  B2  B1  B0
  Stage 2:         F0  F1  F2  F3  B3  B2  B1  B0
  Stage 3:             F0  F1  F2  F3  B3  B2  B1  B0

The warmup phase fills the pipeline, then we alternate F and B.
This minimizes the pipeline bubble compared to GPipe's all-forward-then-all-backward.

Reference: PipeDream (Narayanan et al., 2019)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional, Callable
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism."""
    num_stages: int
    num_micro_batches: int
    stage_id: int  # which stage this process handles


class PipelineStage:
    """Wraps a model shard as a pipeline stage.

    Handles send/recv of activations between adjacent stages.
    """

    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
    ):
        self.module = module.to(device)
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.device = device
        self.group = group

        self.is_first = stage_id == 0
        self.is_last = stage_id == num_stages - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def send_forward(self, tensor: torch.Tensor, dst: int):
        """Send activations to the next stage."""
        dist.send(tensor.contiguous(), dst=dst, group=self.group)

    def recv_forward(self, shape: tuple, dtype: torch.dtype, src: int) -> torch.Tensor:
        """Receive activations from the previous stage."""
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(tensor, src=src, group=self.group)
        return tensor

    def send_backward(self, tensor: torch.Tensor, dst: int):
        """Send gradients to the previous stage."""
        dist.send(tensor.contiguous(), dst=dst, group=self.group)

    def recv_backward(self, shape: tuple, dtype: torch.dtype, src: int) -> torch.Tensor:
        """Receive gradients from the next stage."""
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(tensor, src=src, group=self.group)
        return tensor


def run_1f1b_schedule(
    stage: PipelineStage,
    micro_batches: List[torch.Tensor],
    loss_fn: Callable,
    labels: Optional[List[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    """
    Execute the 1F1B pipeline schedule.

    Args:
        stage: this process's pipeline stage
        micro_batches: list of micro-batch inputs (only used by stage 0)
        loss_fn: loss function (only used by last stage)
        labels: list of label tensors (only used by last stage)

    Returns:
        List of losses (only meaningful for last stage)
    """
    num_micro = len(micro_batches)
    num_warmup = stage.num_stages - stage.stage_id - 1
    num_warmup = min(num_warmup, num_micro)
    num_steady = num_micro - num_warmup

    # storage for activations (needed for backward)
    input_tensors: List[Optional[torch.Tensor]] = []
    output_tensors: List[Optional[torch.Tensor]] = []
    losses: List[torch.Tensor] = []

    # === Warmup phase: only forward passes ===
    for i in range(num_warmup):
        inp, out = _forward_step(stage, micro_batches, i, loss_fn, labels)
        input_tensors.append(inp)
        output_tensors.append(out)
        if stage.is_last:
            losses.append(out)

    # === Steady state: 1 forward + 1 backward ===
    for i in range(num_steady):
        # forward
        fwd_idx = num_warmup + i
        inp, out = _forward_step(stage, micro_batches, fwd_idx, loss_fn, labels)
        input_tensors.append(inp)
        output_tensors.append(out)
        if stage.is_last:
            losses.append(out)

        # backward for the oldest buffered micro-batch
        bwd_idx = i
        _backward_step(stage, input_tensors[bwd_idx], output_tensors[bwd_idx])

    # === Cooldown: remaining backward passes ===
    for i in range(num_steady, num_micro):
        _backward_step(stage, input_tensors[i], output_tensors[i])

    return losses


def _forward_step(
    stage: PipelineStage,
    micro_batches: List[torch.Tensor],
    micro_idx: int,
    loss_fn: Callable,
    labels: Optional[List[torch.Tensor]],
) -> tuple:
    """Run one forward micro-step."""
    if stage.is_first:
        inp = micro_batches[micro_idx].to(stage.device)
    else:
        # receive from previous stage
        # TODO: need to know the shape somehow — for now, this is a placeholder
        # that would need actual shape negotiation in a real impl
        inp = stage.recv_forward(
            micro_batches[micro_idx].shape,
            micro_batches[micro_idx].dtype,
            src=stage.stage_id - 1,
        )

    inp.requires_grad_(True)
    out = stage.forward(inp)

    if stage.is_last:
        # compute loss
        if labels is not None:
            loss = loss_fn(out, labels[micro_idx].to(stage.device))
        else:
            loss = loss_fn(out)
        return inp, loss
    else:
        # send to next stage
        stage.send_forward(out.detach(), dst=stage.stage_id + 1)
        return inp, out


def _backward_step(
    stage: PipelineStage,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
):
    """Run one backward micro-step."""
    if stage.is_last:
        # output_tensor is the loss — just backward it
        output_tensor.backward()
    else:
        # receive gradient from next stage
        grad = stage.recv_backward(
            output_tensor.shape,
            output_tensor.dtype,
            src=stage.stage_id + 1,
        )
        output_tensor.backward(grad)

    if not stage.is_first:
        # send input gradient to previous stage
        stage.send_backward(input_tensor.grad, dst=stage.stage_id - 1)
