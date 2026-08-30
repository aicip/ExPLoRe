"""
Soft Mixture of Experts (Soft MoE) implementation for MEDiC.

Based on "From Sparse to Soft Mixtures of Experts" (Puigcerver et al., 2023)
Paper: https://arxiv.org/abs/2308.00951

Extended with dispatch weight return for token-level dynamic loss weighting.
"""

import math
from typing import Callable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import Mlp
except ImportError:  # minimal CPU verification environments
    from .timm_compat import Mlp


class SoftMoELayer(nn.Module):
    """
    Soft Mixture of Experts layer for token-level dynamic loss weighting.

    Key innovation: Returns dispatch weights D for per-token loss weighting where:
    - Expert 1 (slot 0) weights control distillation loss
    - Expert 2 (slot 1) weights control reconstruction loss

    Architecture:
        - Learnable slot parameters Phi [dim, num_experts * slots_per_expert]
        - Dispatch weights D: softmax over token dimension (aggregate tokens → slots)
        - Combine weights C: softmax over slot dimension (aggregate slots → tokens)
        - Expert networks: Standard MLP for each expert

    Forward pass:
        1. Compute logits: X @ Phi
        2. Get dispatch/combine weights from logits
        3. Dispatch: input_slots = D^T @ X
        4. Process each slot through its expert
        5. Combine: output = C @ output_slots

    Args:
        dim: Hidden dimension (e.g., 768 for ViT-Base)
        num_experts: Number of experts (supports arbitrary N, default: 2)
        slots_per_expert: Number of slots per expert (fixed to 1)
        mlp_ratio: MLP hidden dimension expansion ratio (default: 4.0)
        act_layer: Activation function (default: GELU)
        drop: Dropout rate (default: 0.0)

    Note: For multi-expert MoE collapse mitigation:
        - Set num_experts=4 (or more) to have redundant experts
        - Configure moe_loss_expert_indices=[0, 1] to use only first 2 for loss
        - Remaining experts (2, 3, ..., N-1) are isolated from gradient feedback
        - This prevents collapse while maintaining diverse representations
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 2,
        slots_per_expert: int = 1,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        drop: float = 0.0,
        expert_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.num_slots = num_experts * slots_per_expert  # Total slots = 2 * 1 = 2
        # Expert Dropout (ExD), paper §4.6.3 / Switch Transformer JMLR 22.
        # Applied to per-slot expert outputs during training only (eval = identity).
        # Can be set after construction by the finetune driver (--expert_dropout flag).
        self.expert_dropout_p = float(expert_dropout_p)

        # Learnable slot parameters Phi [dim, num_slots]
        # These define the "queries" for soft routing
        self.phi = nn.Parameter(torch.randn(dim, self.num_slots))
        # Initialize with Kaiming uniform (same as Linear layers)
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

        # Learnable scale parameter for routing sharpness control (matching V-MoE)
        # Initialized to 1.0 to maintain initial behavior
        self.scale = nn.Parameter(torch.ones(()))

        # Expert MLPs (num_experts = 2)
        # Each expert is a standard MLP with expansion ratio
        self.experts = nn.ModuleList([
            Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=drop,
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, return_weights: bool = True) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with soft routing and optional weight return.

        Args:
            x: Input tokens [B, N, D] where:
               - B: batch size
               - N: number of tokens (including CLS)
               - D: hidden dimension
            return_weights: If True, return routing weights dict. If False, return None.
                           Note: Weights are ALWAYS computed (needed for MoE forward pass),
                           this only controls whether they are returned.

        Returns:
            output: Processed tokens [B, N, D] after soft routing through experts
            weights: Dict with routing weights, or None if return_weights=False
                {
                    'combine': [B, N, num_slots] - Combine weights (softmax over experts, dim=2)
                        - Determines how expert outputs are mixed per token
                        - combine[:, :, i] = weight of expert i's contribution to each token
                        - Each row sums to 1 (per-token distribution over experts)
                    'dispatch': [B, N, num_slots] - Dispatch weights (softmax over tokens, dim=1)
                        - Determines how tokens are aggregated into each expert's input
                        - dispatch[:, :, i] = weight of each token's contribution to expert i
                        - Each column sums to 1 (per-expert distribution over tokens)
                }

        Important: Both weights are ALWAYS computed as they are essential for the MoE
        forward pass. The return_weights parameter only controls whether to return them
        for loss weighting and visualization purposes.
        """
        B, N, D = x.shape

        # Normalize inputs and phi to unit norm (matching V-MoE)
        # This stabilizes routing when many similar tokens are present (dense mode with mask tokens).
        # Without normalization, unbounded logits lead to softmax degeneracy and training divergence.
        # Reference: V-MoE paper (https://arxiv.org/abs/2308.00951) lines 59, 75 in router.py
        x_normalized = F.normalize(x, dim=-1, p=2, eps=1e-6)  # [B, N, D] → ||x[b,n,:]||₂ = 1
        phi_normalized = F.normalize(self.phi, dim=0, p=2, eps=1e-6)  # [D, S] → ||phi[:,s]||₂ = 1

        # 1. Compute logits with NORMALIZED inputs: X @ Phi -> [B, N, num_slots]
        logits = torch.einsum('bnd,ds->bns', x_normalized, phi_normalized)

        # Apply learnable scale for routing sharpness control (matching V-MoE lines 78-82)
        logits = logits * self.scale

        # 2. Dispatch weights: softmax over token dimension (route tokens to slots)
        dispatch_weights = F.softmax(logits, dim=1)  # [B, N, num_slots]

        # 3. Combine weights: softmax over slot dimension (mix expert outputs)
        # CRITICAL: These must ALWAYS be computed - they determine how expert
        # outputs are combined. Using uniform weights would change the architecture!
        combine_weights = F.softmax(logits, dim=2)   # [B, N, num_slots]

        # 4. Dispatch: aggregate tokens to slots
        input_slots = torch.einsum('bns,bnd->bsd', dispatch_weights, x)

        # 5. Process each slot through its corresponding expert
        output_slots = []
        for i in range(self.num_experts):
            for j in range(self.slots_per_expert):
                slot_idx = i * self.slots_per_expert + j
                slot_input = input_slots[:, slot_idx, :]  # [B, D]
                slot_output = self.experts[i](slot_input)  # [B, D]
                output_slots.append(slot_output)

        output_slots = torch.stack(output_slots, dim=1)  # [B, num_slots, D]

        # Expert Dropout (ExD): randomly zero entire slot outputs during training
        # so the combine step cannot rely on any single expert.
        # No-op when self.expert_dropout_p == 0.0 (the default) or in eval mode.
        if self.training and self.expert_dropout_p > 0.0:
            keep_prob = 1.0 - self.expert_dropout_p
            mask = torch.empty(
                output_slots.shape[0], output_slots.shape[1], 1,
                device=output_slots.device, dtype=output_slots.dtype,
            ).bernoulli_(keep_prob)
            output_slots = output_slots * mask / keep_prob

        # 6. Combine: aggregate slots back to tokens using LEARNED weights
        output = torch.einsum('bns,bsd->bnd', combine_weights, output_slots)

        # Return weights dict for loss weighting and visualization
        if return_weights:
            weights_dict = {
                'combine': combine_weights,    # [B, N, num_slots] - per-token expert mixture
                'dispatch': dispatch_weights,  # [B, N, num_slots] - per-expert token aggregation
            }
            return output, weights_dict
        else:
            return output, None

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'dim={self.dim}, '
            f'num_experts={self.num_experts}, '
            f'slots_per_expert={self.slots_per_expert}, '
            f'num_slots={self.num_slots})'
        )
