"""ExPLoRe-VTLA multimodal temporal model and objective-coupled training loss."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import VTLAConfig, VTLASequence
from .router import LossCoupledRouter, loss_coupled_reduce, router_balance_loss


@dataclass
class VTLAOutput:
    token_state: torch.Tensor
    routed_state: torch.Tensor
    dispatch: torch.Tensor
    combine: torch.Tensor
    reconstruction: dict[str, torch.Tensor]
    next_prediction: dict[str, torch.Tensor]
    action: torch.Tensor
    contact_logits: torch.Tensor
    slip_logits: torch.Tensor
    feasibility_logits: torch.Tensor
    reliability: torch.Tensor


class MultimodalTokenizer(nn.Module):
    def __init__(self, config: VTLAConfig) -> None:
        super().__init__()
        self.config = config
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                )
                for name, dim in config.modality_dims.items()
            }
        )
        self.modality_embedding = nn.Parameter(
            torch.empty(len(config.modality_dims), config.hidden_dim)
        )
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.quality_projection = nn.Sequential(
            nn.Linear(3, config.hidden_dim),
            nn.Tanh(),
        )

    def _time_encoding(self, timesteps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if timesteps > self.config.max_timesteps:
            raise ValueError(
                f"trajectory length {timesteps} exceeds max_timesteps={self.config.max_timesteps}"
            )
        dim = self.config.hidden_dim
        position = torch.arange(timesteps, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / dim)
        )
        enc = torch.zeros(timesteps, dim, device=device, dtype=dtype)
        enc[:, 0::2] = torch.sin(position * div)
        enc[:, 1::2] = torch.cos(position * div[: enc[:, 1::2].shape[1]])
        return enc

    def forward(self, sequence: VTLASequence) -> tuple[torch.Tensor, torch.Tensor]:
        sequence.validate(
            self.config.modality_order,
            modality_dims=self.config.modality_dims,
            action_dim=self.config.action_dim,
        )
        batch, timesteps, _ = sequence.action.shape
        reliability = sequence.reliability()
        time = self._time_encoding(timesteps, sequence.action.device, sequence.action.dtype)
        tokens = []
        for idx, name in enumerate(self.config.modality_order):
            projected = self.projections[name](sequence.modalities[name])
            raw_gate = reliability[..., idx].unsqueeze(-1)
            token = projected * raw_gate
            token = token + self.modality_embedding[idx].view(1, 1, -1)
            token = token + time.view(1, timesteps, -1)
            token = token + self.quality_projection(sequence.quality[..., idx, :])
            tokens.append(token)
        stacked = torch.stack(tokens, dim=2)
        if stacked.shape != (batch, timesteps, len(tokens), self.config.hidden_dim):
            raise RuntimeError("tokenizer produced unexpected shape")
        return stacked, reliability


class ExPLoReVTLA(nn.Module):
    def __init__(self, config: VTLAConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = MultimodalTokenizer(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers, enable_nested_tensor=False
        )
        self.router = LossCoupledRouter(
            dim=config.hidden_dim,
            num_experts=len(config.objectives),
            hidden_mult=config.expert_hidden_mult,
            dropout=config.dropout,
        )
        self.reconstruction_heads = nn.ModuleDict(
            {name: nn.Linear(config.hidden_dim, dim) for name, dim in config.modality_dims.items()}
        )
        self.world_heads = nn.ModuleDict(
            {name: nn.Linear(config.hidden_dim, dim) for name, dim in config.modality_dims.items()}
        )
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.contact_head = nn.Linear(config.hidden_dim, 1)
        self.slip_head = nn.Linear(config.hidden_dim, 1)
        self.feasibility_head = nn.Linear(config.hidden_dim, 1)

    def _causal_mask(self, timesteps: int, modalities: int, device: torch.device) -> torch.Tensor:
        token_times = torch.arange(timesteps, device=device).repeat_interleave(modalities)
        future = token_times.unsqueeze(0) > token_times.unsqueeze(1)
        mask = torch.zeros((timesteps * modalities, timesteps * modalities), device=device)
        mask = mask.masked_fill(future, float("-inf"))
        return mask

    def forward(self, sequence: VTLASequence) -> VTLAOutput:
        tokens, reliability = self.tokenizer(sequence)
        batch, timesteps, modalities, dim = tokens.shape
        flat = tokens.reshape(batch, timesteps * modalities, dim)
        contextual = self.temporal(flat, mask=self._causal_mask(timesteps, modalities, flat.device))
        routed, weights = self.router(contextual, reliability.reshape(batch, -1))
        routed4 = routed.reshape(batch, timesteps, modalities, dim)

        reconstruction: dict[str, torch.Tensor] = {}
        next_prediction: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.config.modality_order):
            reconstruction[name] = self.reconstruction_heads[name](routed4[:, :, idx, :])
            next_prediction[name] = self.world_heads[name](routed4[:, :, idx, :])

        return VTLAOutput(
            token_state=contextual.reshape(batch, timesteps, modalities, dim),
            routed_state=routed4,
            dispatch=weights["dispatch"].reshape(
                batch, timesteps, modalities, len(self.config.objectives)
            ),
            combine=weights["combine"].reshape(
                batch, timesteps, modalities, len(self.config.objectives)
            ),
            reconstruction=reconstruction,
            next_prediction=next_prediction,
            action=self.action_head(routed4),
            contact_logits=self.contact_head(routed4).squeeze(-1),
            slip_logits=self.slip_head(routed4).squeeze(-1),
            feasibility_logits=self.feasibility_head(routed4).squeeze(-1),
            reliability=reliability,
        )

    def objective_weights(self, output: VTLAOutput, objective: str) -> torch.Tensor:
        """Return per-timestep modality weights for one routed objective."""

        if objective not in self.config.objectives:
            raise ValueError(f"unknown objective {objective!r}")
        idx = tuple(self.config.objectives).index(objective)
        weights = output.dispatch[..., idx]
        return weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-12)

    def aggregate_action(self, output: VTLAOutput) -> torch.Tensor:
        weights = self.objective_weights(output, "action").unsqueeze(-1)
        return (output.action * weights).sum(dim=2)

    def aggregate_probability(self, output: VTLAOutput, objective: str) -> torch.Tensor:
        if objective == "contact":
            logits = output.contact_logits
        elif objective == "slip":
            logits = output.slip_logits
        elif objective == "feasibility":
            logits = output.feasibility_logits
        else:
            raise ValueError("objective must be contact, slip, or feasibility")
        weights = self.objective_weights(output, objective)
        return (torch.sigmoid(logits) * weights).sum(dim=2)

    def training_loss(
        self,
        sequence: VTLASequence,
        output: VTLAOutput | None = None,
        *,
        detach_router: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if output is None:
            output = self(sequence)
        batch, timesteps, modalities, experts = output.dispatch.shape
        if experts != len(self.config.objectives):
            raise RuntimeError("router/objective cardinality mismatch")

        losses: dict[str, torch.Tensor] = {}
        valid: dict[str, torch.Tensor] = {}
        reliability_valid = output.reliability > 0

        rec_parts = []
        world_parts = []
        for idx, name in enumerate(self.config.modality_order):
            target = sequence.modalities[name]
            rec_parts.append(((output.reconstruction[name] - target) ** 2).mean(dim=-1))
            world = torch.zeros(batch, timesteps, device=target.device, dtype=target.dtype)
            if timesteps > 1:
                world[:, :-1] = (
                    (output.next_prediction[name][:, :-1] - target[:, 1:]) ** 2
                ).mean(dim=-1)
            world_parts.append(world)
        losses["reconstruction"] = torch.stack(rec_parts, dim=2)

        # Same-timestep cross-modal alignment is an explicit objective rather than
        # an implicit side effect.  The reliability-weighted consensus prevents a
        # missing modality from becoming the alignment target.
        rel = output.reliability.unsqueeze(-1)
        consensus = (output.routed_state * rel).sum(dim=2) / rel.sum(dim=2).clamp_min(1e-6)
        consensus = consensus.unsqueeze(2).expand_as(output.routed_state)
        losses["alignment"] = 1.0 - F.cosine_similarity(
            output.routed_state, consensus, dim=-1, eps=1e-6
        )
        valid["alignment"] = reliability_valid

        losses["world_model"] = torch.stack(world_parts, dim=2)
        world_valid = reliability_valid.clone()
        world_valid[:, -1, :] = False
        valid["reconstruction"] = reliability_valid
        valid["world_model"] = world_valid

        action_target = sequence.action.unsqueeze(2).expand(-1, -1, modalities, -1)
        losses["action"] = ((output.action - action_target) ** 2).mean(dim=-1)
        valid["action"] = reliability_valid

        contact_target = sequence.contact.unsqueeze(2).expand(-1, -1, modalities)
        slip_target = sequence.slip.unsqueeze(2).expand(-1, -1, modalities)
        feasible_target = sequence.feasible.unsqueeze(2).expand(-1, -1, modalities)
        losses["contact"] = F.binary_cross_entropy_with_logits(
            output.contact_logits, contact_target, reduction="none"
        )
        losses["slip"] = F.binary_cross_entropy_with_logits(
            output.slip_logits, slip_target, reduction="none"
        )
        losses["feasibility"] = F.binary_cross_entropy_with_logits(
            output.feasibility_logits, feasible_target, reduction="none"
        )
        valid["contact"] = reliability_valid
        valid["slip"] = reliability_valid
        valid["feasibility"] = reliability_valid

        ordered_losses = []
        ordered_valid = []
        for name in self.config.objectives:
            if name not in losses:
                raise ValueError(f"objective {name!r} has no implemented loss")
            ordered_losses.append(losses[name])
            ordered_valid.append(valid[name])
        per_token = torch.stack(ordered_losses, dim=-1).reshape(batch, timesteps * modalities, experts)
        mask = torch.stack(ordered_valid, dim=-1).reshape(batch, timesteps * modalities, experts)
        coupled, per_objective = loss_coupled_reduce(
            per_token,
            output.dispatch.reshape(batch, timesteps * modalities, experts),
            mask,
            detach_router=detach_router,
        )
        balance = router_balance_loss(output.combine.reshape(batch, timesteps * modalities, experts))
        total = coupled + self.config.router_balance_weight * balance
        metrics = {
            name: per_objective[idx] for idx, name in enumerate(self.config.objectives)
        }
        metrics["coupled"] = coupled
        metrics["router_balance"] = balance
        metrics["total"] = total
        return total, metrics
