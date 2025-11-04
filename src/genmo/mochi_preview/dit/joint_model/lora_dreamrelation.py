#! /usr/bin/env python3
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    assert bias == "none", f"Only bias='none' is supported"
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    assert bias == "none", f"Only bias='none' is supported"
    my_state_dict = model.state_dict()
    return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}


class LoraLinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        lora_names: list = [],
        test_lora_names: list = [],
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.lora_names = lora_names
        self.test_lora_names = test_lora_names

        # Initialize lora_A and lora_B as ModuleDicts
        if r > 0:
            self.lora_A = nn.ParameterDict({
                name: nn.Parameter(self.weight.new_zeros((r, in_features)).to(torch.float32))
                for name in lora_names
            })
            
            self.lora_B = nn.ParameterDict({
                name: nn.Parameter(self.weight.new_zeros((out_features, r)).to(torch.float32))
                for name in lora_names
            })

            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            for name in self.lora_names:
                # initialize B the same way as the default for nn.Linear and A to zero
                # this is different than what is described in the paper but should not affect performance
                nn.init.kaiming_uniform_(self.lora_A[name], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[name])

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                for name in self.test_lora_names:
                    if self.r > 0 and name in self.lora_A:
                        self.weight.data -= T(self.lora_B[name] @ self.lora_A[name]) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                for name in self.test_lora_names:
                    if self.r > 0 and name in self.lora_A:
                        self.weight.data += T(self.lora_B[name] @ self.lora_A[name]) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor, forward_lora_name: list = []):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0 and not self.merged:
            x = self.lora_dropout(x)

            # Calculate lora contribution for each lora name
            for name in self.lora_names:
                contribution = x @ self.lora_A[name].transpose(0, 1)
                contribution = contribution @ self.lora_B[name].transpose(0, 1)
                contribution = contribution * self.scaling

                # Multiply by zero for any lora name that is not the specified forward_lora_name
                if name not in forward_lora_name:
                    contribution = 0 * contribution

                # Add contribution to result
                result = result + contribution

        return result