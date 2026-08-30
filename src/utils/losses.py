import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Any, Tuple, Optional, Union, List

# Lazy import patchify_img to allow importing this module standalone
# (needed for finetuning engine which loads via importlib)
patchify_img = None

def _get_patchify_img():
    """Lazy import of patchify_img."""
    global patchify_img
    if patchify_img is None:
        from .utils import patchify_img as _patchify_img
        patchify_img = _patchify_img
    return patchify_img


def maybe_distributed_mean(t: torch.Tensor) -> torch.Tensor:
    """Average tensor across all GPUs if distributed training is active.

    Args:
        t: Tensor to average across GPUs

    Returns:
        Averaged tensor if distributed, original tensor otherwise
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return t

    # Clone to avoid modifying original
    t_clone = t.clone()
    dist.all_reduce(t_clone, op=dist.ReduceOp.SUM)
    return t_clone / dist.get_world_size()


def apply_normalization(tensor: torch.Tensor, method: str = "variance") -> torch.Tensor:
    """Apply different normalization methods to targets or predictions."""
    if method == "variance":
        # Variance normalization (current default)
        mean = tensor.mean(dim=-1, keepdim=True)
        var = tensor.var(dim=-1, keepdim=True)
        # Handle case where variance is NaN (single element)
        var = torch.where(torch.isnan(var), torch.ones_like(var), var)
        return (tensor - mean) / (var + 1.0e-6) ** 0.5
    elif method == "l2":
        # L2 normalization
        return F.normalize(tensor, p=2, dim=-1)
    elif method == "none":
        # No normalization
        return tensor
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor,
                          beta: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    """Computes SmoothL1 loss on masked patches."""
    assert mask.shape == pred.shape[:2], "Masked Smooth L1 Loss: mask rank-2 must match pred B,L"
    mask = mask.unsqueeze(-1).expand_as(
        pred
    )  # Expand mask to the same shape as pred/target

    assert pred.shape == target.shape, "Masked Smooth L1 Loss: pred/target shape mismatch"
    assert mask.size() == target.size()

    # Handle empty tensors
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # Use autocast for mixed precision support while maintaining numerical stability
    with torch.autocast(device_type='cuda' if pred.device.type == 'cuda' else 'cpu', enabled=True):
        loss_fn = torch.nn.SmoothL1Loss(beta=beta, reduction="none")
        assert torch.all(torch.isfinite(pred)), "Masked Smooth L1 Loss: pred has NaN/Inf"
        assert torch.all(torch.isfinite(target)), "Masked Smooth L1 Loss: target has NaN/Inf"
        loss = loss_fn(pred, target)

        # Apply mask and compute mean over masked elements
        # Use more robust summation to prevent overflow
        masked_loss = loss * mask
        loss_sum = masked_loss.sum()
        mask_sum = mask.sum() + eps

        # Ensure the result is finite
        result = loss_sum / mask_sum
        if not torch.isfinite(result):
            # Fallback to float32 for numerical stability
            result = (masked_loss.float().sum() / (mask.float().sum() + eps))

        return result


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes L2/MSE loss on masked patches."""

    assert mask.shape == pred.shape[:2], "Masked MSE Loss: mask rank-2 must match pred B,L"
    mask = mask.unsqueeze(-1).expand_as(
        pred
    )  # Expand mask to the same shape as pred/target
    assert pred.shape == target.shape, "Masked MSE Loss: pred/target shape mismatch"
    assert torch.all(torch.isfinite(pred)), "Masked MSE Loss: pred has NaN/Inf"
    assert torch.all(torch.isfinite(target)), "Masked MSE Loss: target has NaN/Inf"

    # Use autocast for mixed precision support while maintaining numerical stability
    with torch.autocast(device_type='cuda' if pred.device.type == 'cuda' else 'cpu', enabled=True):
        loss = (pred - target) ** 2

        # Lightning order: per-patch average first, then average over patches
        # Collapse feature dimension first (mean over last dim)
        loss = loss.mean(dim=-1)  # Per patch loss

        # Apply mask to 2D tensor (B, L) and compute mean over masked patches
        mask_2d = mask[:, :, 0]  # Take first channel of expanded mask to get (B, L)
        loss_sum = (loss * mask_2d).sum()
        mask_sum = mask_2d.sum() + eps

        # Ensure the result is finite
        result = loss_sum / mask_sum
        if not torch.isfinite(result):
            # Fallback to float32 for numerical stability
            result = (loss.float() * mask_2d.float()).sum() / (mask_2d.float().sum() + eps)

        return result


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes L1/MAE loss on masked patches."""

    assert mask.shape == pred.shape[:2], "Masked L1 Loss: mask rank-2 must match pred B,L"
    mask = mask.unsqueeze(-1).expand_as(
        pred
    )  # Expand mask to the same shape as pred/target
    assert pred.shape == target.shape, "Masked L1 Loss: pred/target shape mismatch"
    assert torch.all(torch.isfinite(pred)), "Masked L1 Loss: pred has NaN/Inf"
    assert torch.all(torch.isfinite(target)), "Masked L1 Loss: target has NaN/Inf"

    # Use autocast for mixed precision support while maintaining numerical stability
    with torch.autocast(device_type='cuda' if pred.device.type == 'cuda' else 'cpu', enabled=True):
        loss = torch.abs(pred - target)

        # Apply mask and compute mean over masked elements
        # Use more robust summation to prevent overflow
        masked_loss = loss * mask
        loss_sum = masked_loss.sum()
        mask_sum = mask.sum() + eps

        # Ensure the result is finite
        result = loss_sum / mask_sum
        if not torch.isfinite(result):
            # Fallback to float32 for numerical stability
            result = (masked_loss.float().sum() / (mask.float().sum() + eps))

        return result


def cls_ce_loss(pred_cls: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
    """Computes cross-entropy loss for the CLS token using soft targets."""
    assert pred_cls.shape == target_cls.shape, "CLS Token Loss: pred/target shape mismatch"
    assert torch.all(torch.isfinite(pred_cls)), "CLS Token Loss: pred has NaN/Inf"
    assert torch.all(torch.isfinite(target_cls)), "CLS Token Loss: target has NaN/Inf"

    # Use autocast for mixed precision support while maintaining numerical stability
    with torch.autocast(device_type='cuda' if pred_cls.device.type == 'cuda' else 'cpu', enabled=True):
        target_prob = F.softmax(target_cls, dim=-1)
        pred_log_prob = F.log_softmax(pred_cls, dim=-1)
        loss_cls = -(target_prob * pred_log_prob).sum(dim=-1).mean()

        # Ensure the result is finite
        if not torch.isfinite(loss_cls):
            # Fallback to float32 for numerical stability
            target_prob_f32 = F.softmax(target_cls.float(), dim=-1)
            pred_log_prob_f32 = F.log_softmax(pred_cls.float(), dim=-1)
            loss_cls = -(target_prob_f32 * pred_log_prob_f32).sum(dim=-1).mean()

        return loss_cls


def cls_cosine_loss(pred_cls: torch.Tensor, target_cls: torch.Tensor,
                    temperature: float = 0.1) -> torch.Tensor:
    """
    Computes cosine similarity loss for CLS tokens.

    This loss is more appropriate for normalized embeddings from CLIP models,
    as it directly measures the angular distance between embeddings.

    Args:
        pred_cls: Student CLS token embeddings [B, D]
        target_cls: Teacher CLS token embeddings [B, D] (normalized)
        temperature: Temperature scaling factor (lower = sharper gradients)

    Returns:
        Scalar loss value
    """
    assert pred_cls.shape == target_cls.shape, "CLS Cosine Loss: pred/target shape mismatch"
    assert torch.all(torch.isfinite(pred_cls)), "CLS Cosine Loss: pred has NaN/Inf"
    assert torch.all(torch.isfinite(target_cls)), "CLS Cosine Loss: target has NaN/Inf"
    assert temperature > 0, f"CLS Cosine Loss: temperature must be positive, got {temperature}"

    # Use autocast for mixed precision support
    with torch.autocast(device_type='cuda' if pred_cls.device.type == 'cuda' else 'cpu', enabled=True):
        # L2 normalize both embeddings
        pred_cls_norm = F.normalize(pred_cls, dim=-1, p=2)
        target_cls_norm = F.normalize(target_cls, dim=-1, p=2)

        # Compute cosine similarity (dot product of normalized vectors)
        cosine_sim = (pred_cls_norm * target_cls_norm).sum(dim=-1)

        # Clamp to avoid numerical issues
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

        # Convert to loss: (1 - similarity) / temperature
        # This gives loss in range [0, 2/temperature]
        loss = (1 - cosine_sim) / temperature

        # Take mean over batch
        loss_cls = loss.mean()

        # Ensure the result is finite
        if not torch.isfinite(loss_cls):
            # Fallback to float32 for numerical stability
            pred_cls_f32 = F.normalize(pred_cls.float(), dim=-1, p=2)
            target_cls_f32 = F.normalize(target_cls.float(), dim=-1, p=2)
            cosine_sim_f32 = (pred_cls_f32 * target_cls_f32).sum(dim=-1)
            cosine_sim_f32 = torch.clamp(cosine_sim_f32, -1.0, 1.0)
            loss_cls = ((1 - cosine_sim_f32) / temperature).mean()

        return loss_cls


def apply_moe_loss_weighting(
    loss_per_token: torch.Tensor,
    combine_weights: Union[torch.Tensor, Dict[str, torch.Tensor]],
    mask: torch.Tensor,
    expert_idx: int,
    use_mask_tokens: bool,
    ids_restore: Optional[torch.Tensor] = None,
    normalize_weights: bool = True,
    loss_expert_indices: Optional[list] = None,
    weight_type: str = "combine",
    normalize_per_image: bool = True,
    detach_weights: bool = False
) -> torch.Tensor:
    """
    Apply MoE routing weights to per-token loss with optional normalization.

    Supports two weight types:
    - 'combine': Softmax over experts (dim=2) - per-token expert mixture weights
                 Semantically: "how much each expert contributes to each token's output"
    - 'dispatch': Softmax over tokens (dim=1) - per-expert token aggregation weights
                  Semantically: "how much each token contributes to each expert's input"

    Dispatch weights prevent degeneracy (router can't zero entire loss by setting weights to 0).

    Multi-expert isolation: When loss_expert_indices is provided, only the specified expert
    subset participates in loss computation. Other experts remain isolated from gradient
    feedback, preventing MoE collapse while maintaining diverse representations.

    Args:
        loss_per_token: Per-token loss [B, N, D] or [B, N]
        combine_weights: Routing weights, supports two formats for backward compatibility:
                        - Dict format (NEW): {'combine': [B, N, E], 'dispatch': [B, N, E]}
                        - Tensor format (OLD): [B, N, E] (treated as combine weights)
                        In sparse mode: N < full_N (only visible patches)
                        In dense mode: N = full_N (all patches)
        mask: Binary mask [B, N] indicating which patches to weight
        expert_idx: Which expert slot to use from loss_expert_indices (0 = first loss expert, 1 = second, etc.)
        use_mask_tokens: Whether using dense mode (True) or sparse mode (False)
        ids_restore: Optional [B, N] or [N] tensor for unshuffling in sparse mode with shuffle_patches=True
                    Maps from shuffled indices back to spatial indices
        normalize_weights: If True, normalize weights to mean=1.0 to prevent loss magnitude manipulation
        loss_expert_indices: Optional list of expert indices to use for loss (e.g., [0, 1] for first 2 experts)
                            If None, defaults to [0, 1, ..., num_experts-1] (all experts)
        weight_type: Which weights to use: "combine" or "dispatch" (only used if combine_weights is dict)
        normalize_per_image: If True, normalize weights independently per image (more stable, recommended)
                            If False, normalize globally across batch (legacy behavior)
                            Only used when normalize_weights=True

    Returns:
        Weighted loss scalar
    """
    # === Backward Compatibility: Handle both dict and tensor formats ===
    if isinstance(combine_weights, dict):
        # New format: extract requested weight type
        if weight_type not in combine_weights:
            raise ValueError(
                f"weight_type='{weight_type}' not found in combine_weights dict. "
                f"Available keys: {list(combine_weights.keys())}"
            )
        moe_weights = combine_weights[weight_type]
    else:
        # Old format: treat tensor as combine weights (backward compatible)
        moe_weights = combine_weights

    B, N = mask.shape
    num_total_experts = moe_weights.size(2)  # Total number of experts in the model

    # Handle expert index mapping for multi-expert isolation
    # If loss_expert_indices is provided, only use specified experts for loss
    # Otherwise, use all experts (default behavior for backward compatibility)
    if loss_expert_indices is None:
        # Default: use all experts (backward compatible)
        loss_expert_indices = list(range(num_total_experts))

    # Validate expert_idx is within loss expert range
    if expert_idx >= len(loss_expert_indices):
        raise ValueError(
            f"expert_idx={expert_idx} is out of range for loss_expert_indices={loss_expert_indices}. "
            f"expert_idx must be in [0, {len(loss_expert_indices)-1}]"
        )

    # Map from loss expert index (0, 1, ...) to actual model expert slot index
    actual_expert_idx = loss_expert_indices[expert_idx]

    # Validate actual expert index is valid
    if actual_expert_idx >= num_total_experts:
        raise ValueError(
            f"loss_expert_indices contains invalid index {actual_expert_idx}. "
            f"Model has {num_total_experts} experts (indices 0-{num_total_experts-1})"
        )

    # Handle CLS token removal first
    # Dense mode: moe_weights has all N patches + CLS → size N+1
    # Sparse mode: moe_weights has num_visible patches + CLS → size num_visible+1 (where num_visible < N)
    # CLS is ALWAYS present (even when all patches masked), so remove if any tokens exist
    has_cls = moe_weights.size(1) == mask.size(1) + 1 or (not use_mask_tokens and moe_weights.size(1) > 0)

    if has_cls:
        # Remove CLS token (first position) in both dense and sparse modes
        moe_weights = moe_weights[:, 1:, :]

    # Unshuffle moe_weights for DENSE mode with shuffling
    # In dense mode with shuffling, moe_weights are in shuffled order but mask is spatial
    # Need to reorder moe_weights to match spatial mask
    if use_mask_tokens and ids_restore is not None:
        # Dense mode: simple reordering of all N patches from shuffled → spatial
        # ids_restore[shuffled_idx] gives spatial_idx
        # So: spatial = shuffled[ids_restore]
        if ids_restore.dim() == 1:
            # Single ids_restore for all batches
            ids_restore_batch = ids_restore.unsqueeze(0).expand(B, -1)
        else:
            # Already batched
            ids_restore_batch = ids_restore

        # Reorder moe_weights: [B, N, num_experts]
        # Use gather: output[i,j,k] = input[i, index[i,j,k], k]
        # We want: output[i, spatial_idx, k] = input[i, shuffled_idx, k]
        # where ids_restore[shuffled_idx] = spatial_idx
        # So: output = input gathered by ids_restore
        num_experts = moe_weights.size(2)
        ids_expanded = ids_restore_batch.unsqueeze(-1).expand(-1, -1, num_experts)
        moe_weights = torch.gather(moe_weights, dim=1, index=ids_expanded)
        # Now moe_weights are in spatial order, matching mask

    # Handle sparse mode: expand moe_weights AND loss_per_token to full patch size
    if not use_mask_tokens and moe_weights.size(1) < N:
        # Sparse mode: moe_weights and loss_per_token only contain visible patches
        # Need to expand to full size by inserting zeros for masked positions
        num_visible_in_weights = moe_weights.size(1)  # Number of visible patches in moe_weights
        num_experts = moe_weights.size(2)

        # Get indices of visible patches (mask == 0 for visible)
        visible_mask = (mask == 0).float()  # [B, N]

        # Handle shuffling: if ids_restore provided, weights/loss are in shuffled order
        # We need to unshuffle them before expanding to spatial positions
        if ids_restore is not None:
            # ids_restore maps: shuffled_idx -> spatial_idx
            # We need inverse: spatial_idx -> shuffled_idx
            # But we only care about visible patches

            # First, unshuffle moe_weights and loss_per_token to spatial order
            if ids_restore.dim() == 1:
                # Single ids_restore for all batches - expand to batch
                ids_restore_batch = ids_restore.unsqueeze(0).expand(B, -1)
            else:
                # Already batched
                ids_restore_batch = ids_restore

            # OPTIMIZATION: Check if all batches have same visible count (common case)
            num_visible_per_batch = visible_mask.sum(dim=1)  # [B]
            uniform_visible = torch.all(num_visible_per_batch == num_visible_per_batch[0])

            if uniform_visible and num_visible_per_batch[0] > 0:
                # FAST PATH: Vectorized unshuffling for uniform visible counts
                # This is 5-10x faster than the batch loop below
                num_visible = int(num_visible_per_batch[0].item())

                # Step 1: Create inverse mapping (spatial → shuffled) for all batches
                batch_indices = torch.arange(B, device=mask.device).unsqueeze(1).expand(-1, N)
                spatial_indices = torch.arange(N, device=mask.device).unsqueeze(0).expand(B, -1)

                spatial_to_shuffled = torch.zeros(B, N, dtype=torch.long, device=mask.device)
                spatial_to_shuffled[batch_indices, ids_restore_batch] = spatial_indices

                # Step 2: Get visible spatial indices (batched)
                visible_indices_flat = visible_mask.nonzero(as_tuple=False)
                visible_spatial_pos = visible_indices_flat[:, 1]
                visible_spatial_indices = visible_spatial_pos.reshape(B, num_visible)

                # Step 3: Find shuffled positions of visible patches
                visible_shuffled_positions = torch.gather(
                    spatial_to_shuffled, dim=1, index=visible_spatial_indices
                )

                # Step 4: Compute ranks (batched argsort)
                ranks = torch.argsort(torch.argsort(visible_shuffled_positions, dim=1), dim=1)

                # Step 5: Reorder moe_weights (batched gather)
                ranks_expanded = ranks.unsqueeze(-1).expand(-1, -1, num_experts)
                moe_weights = torch.gather(moe_weights, dim=1, index=ranks_expanded)

                # Step 6: Reorder loss_per_token (batched gather)
                if loss_per_token.dim() == 3:
                    D = loss_per_token.size(2)
                    ranks_expanded_loss = ranks.unsqueeze(-1).expand(-1, -1, D)
                    loss_per_token = torch.gather(loss_per_token, dim=1, index=ranks_expanded_loss)
                else:
                    loss_per_token = torch.gather(loss_per_token, dim=1, index=ranks)

            else:
                raise Exception(
                    "Sparse MoE loss weighting with shuffling and non-uniform visible patch counts "
                    "is not supported in the optimized path. Please ensure uniform visible counts "
                    "or disable shuffling."
                )

            # Now moe_weights and loss_per_token are in spatial order (for visible patches)

        # Expand moe_weights and loss_per_token to full patch size
        # OPTIMIZATION: Check for uniform visible counts again for expansion
        num_visible_per_batch = visible_mask.sum(dim=1)  # [B]
        uniform_visible = torch.all(num_visible_per_batch == num_visible_per_batch[0])

        if uniform_visible and num_visible_per_batch[0] > 0:
            # FAST PATH: Vectorized expansion
            num_visible = int(num_visible_per_batch[0].item())

            # Initialize full tensors
            full_weights = torch.zeros(B, N, num_experts, device=moe_weights.device, dtype=moe_weights.dtype)
            if loss_per_token.dim() == 3:
                D = loss_per_token.size(2)
                full_loss = torch.zeros(B, N, D, device=loss_per_token.device, dtype=loss_per_token.dtype)
            else:
                full_loss = torch.zeros(B, N, device=loss_per_token.device, dtype=loss_per_token.dtype)

            # Get visible indices (batched)
            visible_indices_flat = visible_mask.nonzero(as_tuple=False)
            visible_spatial_pos = visible_indices_flat[:, 1]
            visible_spatial_indices = visible_spatial_pos.reshape(B, num_visible)

            # Scatter values to visible positions (batched)
            visible_indices_expanded = visible_spatial_indices.unsqueeze(-1).expand(-1, -1, num_experts)
            full_weights.scatter_(dim=1, index=visible_indices_expanded, src=moe_weights)

            if loss_per_token.dim() == 3:
                visible_indices_expanded_loss = visible_spatial_indices.unsqueeze(-1).expand(-1, -1, D)
                full_loss.scatter_(dim=1, index=visible_indices_expanded_loss, src=loss_per_token)
            else:
                full_loss.scatter_(dim=1, index=visible_spatial_indices, src=loss_per_token)

        else:
            raise Exception(
                "Sparse MoE loss weighting with non-uniform visible patch counts "
                "is not supported in the optimized path. Please ensure uniform visible counts."
            )

        moe_weights = full_weights
        loss_per_token = full_loss

    # Now moe_weights should match mask shape
    assert moe_weights.size(1) == N, \
        f"After expansion, combine weights {moe_weights.shape} must match mask size [B, {N}]"

    # Get weights for the specified expert using mapped index
    # actual_expert_idx maps from loss expert index to model expert slot
    expert_weights = moe_weights[:, :, actual_expert_idx]  # [B, N]

    # Optional detach: weights still modulate the loss, but gradient does not
    # flow back to the router. Used by the wave3b detach ablation (hmk37auo).
    if detach_weights:
        expert_weights = expert_weights.detach()

    # Apply mask to select relevant tokens
    # Dense mode: weight MASKED patches (mask==1) - head loss on masked, pixel loss on masked
    # Sparse mode: weight VISIBLE patches (mask==0) - head loss on visible
    if use_mask_tokens:
        # Dense mode: use mask as-is (mask==1 means masked patches to weight)
        expert_weights_masked = expert_weights * mask.float()  # [B, N]
    else:
        # Sparse mode: invert mask (mask==0 means visible patches to weight)
        expert_weights_masked = expert_weights * (~mask).float()  # [B, N]

    # Compute per-token loss (reduce feature dimension if needed)
    if loss_per_token.dim() == 3:
        # [B, N, D] -> [B, N]
        loss_per_token = loss_per_token.mean(dim=-1)

    # Normalize weights if requested (prevents degenerate solutions)
    if normalize_weights:
        if normalize_per_image:
            # Per-image normalization (recommended): Each image normalized independently
            # This ensures each image contributes equally regardless of batch composition

            # Count actual masked/visible positions per image
            if use_mask_tokens:
                # Dense mode: count masked patches per image [B, 1]
                mask_count = mask.float().sum(dim=1, keepdim=True)
            else:
                # Sparse mode: count visible patches per image [B, 1]
                mask_count = (~mask).float().sum(dim=1, keepdim=True)

            # Normalize weights so mean = 1.0 per image
            # This prevents the router from minimizing loss by reducing weights
            # [B, N] / [B, 1] = [B, N] (broadcasting)
            weight_mean = expert_weights_masked.sum(dim=1, keepdim=True) / (mask_count + 1e-8)

            # Check for valid normalization per image
            valid_images = (mask_count.squeeze(1) > 0) & (weight_mean.squeeze(1) > 1e-8)

            if valid_images.all():
                expert_weights_normalized = expert_weights_masked / (weight_mean + 1e-8)
            elif valid_images.any():
                # Some images have no masked/visible patches - handle them separately
                expert_weights_normalized = expert_weights_masked.clone()
                expert_weights_normalized[valid_images] = (
                    expert_weights_masked[valid_images] / weight_mean[valid_images]
                )
                # For invalid images, use uniform weighting as fallback
                invalid_images = ~valid_images
                if invalid_images.any():
                    fallback_mask = mask.float() if use_mask_tokens else (~mask).float()
                    expert_weights_normalized[invalid_images] = fallback_mask[invalid_images]
            else:
                # All images invalid - use uniform weighting
                expert_weights_normalized = mask.float() if use_mask_tokens else (~mask).float()
        else:
            # Per-batch normalization (legacy): All images share same normalization factor
            # This can cause batch-dependent loss contributions

            # Count actual masked/visible positions globally (scalar)
            if use_mask_tokens:
                # Dense mode: count masked patches
                mask_count = mask.float().sum()
            else:
                # Sparse mode: count visible patches
                mask_count = (~mask).float().sum()

            # Normalize weights so mean = 1.0 globally
            if mask_count > 0:
                weight_mean = expert_weights_masked.sum() / (mask_count + 1e-8)
                if weight_mean > 1e-8:
                    expert_weights_normalized = expert_weights_masked / weight_mean
                else:
                    # Weights are essentially zero, use uniform weighting as fallback
                    expert_weights_normalized = expert_weights_masked + (mask.float() if use_mask_tokens else (~mask).float())
            else:
                expert_weights_normalized = expert_weights_masked
    else:
        # No normalization (backward compatibility)
        expert_weights_normalized = expert_weights_masked
        mask_count = None  # Will use weight sum for loss computation

    # Apply MoE weighting with normalized weights
    weighted_loss = loss_per_token * expert_weights_normalized  # [B, N]

    # Compute loss
    if normalize_weights:
        if normalize_per_image:
            # Per-image: average over images, then mean
            # Each image contributes equally
            per_image_loss = weighted_loss.sum(dim=1) / (mask_count.squeeze(1) + 1e-8)  # [B]
            loss = per_image_loss.mean()
        else:
            # Per-batch: global average
            loss_sum = weighted_loss.sum()
            loss = loss_sum / (mask_count + 1e-8)
    else:
        # Backward compatible: weighted average using weight sum
        weight_sum = expert_weights_masked.sum() + 1e-6
        loss = weighted_loss.sum() / weight_sum

    return loss


def compute_moe_scale_regularization(
    model: nn.Module,
    regularization_type: str = 'adaptive',
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute regularization penalty for MoE scale parameters to prevent collapse.

    This addresses the issue where the last MoE block's scale parameter grows
    excessively (e.g., 15.53 vs ~9 for other blocks) when MoE loss weighting is
    enabled, causing softmax saturation and 1.0/0.0 weight collapse.

    Args:
        model: MEDiC model (may be wrapped in DDP)
        regularization_type: Type of regularization:
            - 'none': No regularization (default for backward compatibility)
            - 'l2': Simple L2 penalty on all scale parameters
            - 'adaptive': Penalize deviation from mean scale (recommended)
            - 'adaptive_ratio': Penalize large max/min ratio
        device: Device for the penalty tensor

    Returns:
        Scalar penalty tensor
    """
    if regularization_type == 'none':
        return torch.tensor(0.0, device=device)

    # Handle DDP wrapper
    model_unwrapped = model.module if hasattr(model, 'module') else model

    # Collect scale parameters from MoE blocks
    scale_params = []
    if hasattr(model_unwrapped, 'student'):
        for name, param in model_unwrapped.student.named_parameters():
            # Match patterns:
            # - blocks.X.mlp.soft_moe.scale (when SoftMoE module is used)
            # - blocks.X.mlp.scale (when scale is directly added to mlp)
            # Both patterns indicate MoE scale parameters that need regularization
            if 'blocks' in name and 'mlp' in name and name.endswith('scale'):
                # Ensure it's actually an MoE scale, not some other scale parameter
                # MoE scales are single scalar parameters
                if param.numel() == 1:
                    scale_params.append(param)

    if len(scale_params) == 0:
        return torch.tensor(0.0, device=device)

    if regularization_type == 'adaptive':
        # Penalize deviation from mean - prevents outliers like block 11
        scales = torch.stack(scale_params)
        mean_scale = scales.mean()
        # Variance penalty encourages all blocks to have similar scales
        penalty = ((scales - mean_scale) ** 2).sum()

    elif regularization_type == 'adaptive_ratio':
        # Penalize large max/min ratio (e.g., 15.53/1.0 = 15x difference)
        scales = torch.stack(scale_params)
        max_scale = scales.max()
        min_scale = scales.min()
        ratio = max_scale / (min_scale + 1e-6)
        # Only penalize if ratio exceeds threshold (e.g., 3x)
        penalty = torch.relu(ratio - 3.0)

    elif regularization_type == 'l2':
        # Simple L2 penalty - limits absolute magnitude
        penalty = sum(p.pow(2) for p in scale_params)
    else:
        raise ValueError(f"Unknown MoE scale regularization type: {regularization_type}")

    return penalty


def compute_importance_loss(
    moe_weights: Union[torch.Tensor, Dict[str, torch.Tensor]],
    weight_type: str = "combine",
    expert_indices: Optional[List[int]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute importance loss to prevent MoE expert collapse.

    Forces balanced expert usage across the batch by penalizing uneven routing distributions.
    Uses coefficient of variation squared as the penalty metric.

    Args:
        moe_weights: Routing weights, supports two formats:
                    - Dict format: {'combine': [B, N, E], 'dispatch': [B, N, E]}
                    - Tensor format: [B, N, E] (treated as combine weights)
        weight_type: Which weights to use if dict format: "combine" or "dispatch"
        expert_indices: Optional list of expert indices to regularize. If None, regularize all experts.
                       If provided, only compute loss for specified experts (e.g., [0, 1] for first two experts)

    Returns:
        importance_loss: Scalar loss (coefficient of variation squared)
        metrics: Dict with diagnostics
            - 'importance_cv': Coefficient of variation (0 = balanced, high = imbalanced)
            - 'importance_cv_squared': CV squared (the actual loss)
            - 'expert_i_importance': Sum of routing weights for expert i
    """
    # Extract weights (handle both dict and tensor formats)
    if isinstance(moe_weights, dict):
        if weight_type not in moe_weights:
            raise ValueError(
                f"weight_type='{weight_type}' not found in moe_weights dict. "
                f"Available keys: {list(moe_weights.keys())}"
            )
        weights = moe_weights[weight_type]
    else:
        weights = moe_weights

    # Compute expert importance: sum of routing weights across all tokens in batch
    # weights: [B, N, num_experts]
    importance = weights.sum(dim=[0, 1])  # [num_experts]

    # Filter to specified experts if provided
    if expert_indices is not None:
        # Only compute CV on specified experts
        importance_filtered = importance[expert_indices]  # [len(expert_indices)]
    else:
        # Use all experts (original behavior)
        importance_filtered = importance

    # Coefficient of variation: std / mean
    # CV = 0 means perfectly balanced, high CV means imbalanced
    importance_mean = importance_filtered.mean()
    importance_std = importance_filtered.std(unbiased=False)
    cv = importance_std / (importance_mean + 1e-8)
    cv_squared = cv ** 2

    # Metrics for logging
    metrics = {
        'importance_cv': cv.detach(),
        'importance_cv_squared': cv_squared.detach(),
    }

    # Per-expert importance for monitoring (always log all experts, not just filtered ones)
    num_experts = importance.size(0)
    for i in range(num_experts):
        metrics[f'expert_{i}_importance'] = importance[i].detach()

    return cv_squared, metrics


def compute_dispatch_entropy_loss(
    moe_weights: Union[torch.Tensor, Dict[str, torch.Tensor]],
    weight_type: str = "dispatch",
    aggregation: str = "mean",
    expert_indices: Optional[List[int]] = None,
    expert_weights: Optional[Union[float, List[float], torch.Tensor]] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute entropy loss to encourage uniform dispatch weight distributions.

    Maximizes entropy of dispatch weights per expert, encouraging each expert to
    attend to all tokens uniformly rather than focusing on a few tokens.

    Higher entropy = more uniform distribution = better load balancing per expert.
    Lower entropy = peaked distribution = only few tokens contribute heavily.

    This is the analogue of importance loss but for dispatch weights:
    - Importance loss: Prevents expert collapse (expert-level balance)
    - Entropy loss: Prevents token collapse (token-level balance per expert)

    Mathematical formulation:
        For each expert e: H(e) = -sum_n(p_n * log(p_n))
        where p_n = dispatch_weights[:, n, e] (token n's contribution to expert e)

        Loss aggregation depends on 'aggregation' parameter:
        - "mean": Loss = -mean(H) (original, can mask bad experts)
        - "min": Loss = -min(H) (penalize worst expert most)
        - "mean+var": Loss = -mean(H) + variance_weight * var(H) (balance experts)

    Args:
        moe_weights: Routing weights, supports two formats:
                    - Dict format: {'combine': [B, N, E], 'dispatch': [B, N, E]}
                    - Tensor format: [B, N, E] (treated as specified weight_type)
        weight_type: Which weights to use if dict format: "combine" or "dispatch"
                    Typically "dispatch" for this loss.
        aggregation: How to aggregate entropy across experts:
                    - "mean": Average entropy (original, default)
                    - "min": Minimum entropy (penalize worst expert)
                    - "mean+var": Mean with variance penalty (balance experts)
                    - "separate": Apply independently to each expert (sum, no averaging)
        expert_indices: Optional list of expert indices to regularize. If None, regularize all experts.
                       If provided, only compute loss for specified experts (e.g., [0, 1] for first two experts)
        expert_weights: Optional per-expert loss weights. Can be:
                       - None: Uniform weighting (default, backward compatible)
                       - float: Same weight for all experts (e.g., 0.1)
                       - list: Per-expert weights (e.g., [0.15, 0.05] for 2 experts)
                       - torch.Tensor: Per-expert weights as tensor
                       If aggregation != "separate", weights are applied before aggregation.
                       If aggregation == "separate", weights scale each expert's contribution.

    Returns:
        entropy_loss: Scalar loss (negative entropy with chosen aggregation)
        metrics: Dict with diagnostics
            - 'dispatch_entropy': Mean entropy across experts (higher = more uniform)
            - 'dispatch_entropy_max': Maximum possible entropy (log(N))
            - 'dispatch_uniformity': Normalized entropy in [0, 1] (1 = uniform)
            - 'expert_i_entropy': Entropy for expert i
            - 'entropy_variance': Variance of per-expert entropies (if applicable)
    """
    # Extract weights (handle both dict and tensor formats)
    if isinstance(moe_weights, dict):
        if weight_type not in moe_weights:
            raise ValueError(
                f"weight_type='{weight_type}' not found in moe_weights dict. "
                f"Available keys: {list(moe_weights.keys())}"
            )
        weights = moe_weights[weight_type]
    else:
        weights = moe_weights

    # weights: [B, N, num_experts]
    B, N, num_experts = weights.shape

    # Compute entropy per expert: H(expert_e) = -sum_n(p_n * log(p_n))
    # where p_n = weights[:, n, e] for each expert e

    # Add small epsilon to avoid log(0)
    eps = 1e-8
    weights_safe = weights + eps

    # Entropy per batch and expert: -sum over tokens
    # entropy_per_sample: [B, num_experts]
    entropy_per_sample = -(weights_safe * torch.log(weights_safe)).sum(dim=1)

    # Compute per-expert entropy (average across batch)
    entropy_per_expert = entropy_per_sample.mean(dim=0)  # [num_experts]

    # Filter to specified experts if provided
    if expert_indices is not None:
        # Only compute loss on specified experts
        entropy_per_expert_filtered = entropy_per_expert[expert_indices]  # [len(expert_indices)]
    else:
        # Use all experts (original behavior)
        entropy_per_expert_filtered = entropy_per_expert

    # Convert expert_weights to tensor if provided (for per-expert weighting)
    weights_tensor_filtered = None
    if expert_weights is not None:
        # Convert to tensor based on input type
        if isinstance(expert_weights, (int, float)):
            # Scalar weight - uniform (backward compatible)
            weights_tensor = torch.tensor([expert_weights] * num_experts,
                                        dtype=entropy_per_expert.dtype,
                                        device=entropy_per_expert.device)
        elif isinstance(expert_weights, list):
            # List of per-expert weights
            if len(expert_weights) != num_experts:
                raise ValueError(
                    f"expert_weights list length ({len(expert_weights)}) must match "
                    f"num_experts ({num_experts})"
                )
            weights_tensor = torch.tensor(expert_weights,
                                        dtype=entropy_per_expert.dtype,
                                        device=entropy_per_expert.device)
        else:
            # Already a tensor
            weights_tensor = expert_weights.to(dtype=entropy_per_expert.dtype,
                                              device=entropy_per_expert.device)

        # Filter weights to match expert_indices if provided
        if expert_indices is not None:
            weights_tensor_filtered = weights_tensor[expert_indices]
        else:
            weights_tensor_filtered = weights_tensor

    # Average entropy across filtered experts (for metrics)
    mean_entropy = entropy_per_expert_filtered.mean()

    # Maximum possible entropy (uniform distribution: p = 1/N for all tokens)
    max_entropy = torch.log(torch.tensor(N, dtype=weights.dtype, device=weights.device))

    # Normalized entropy (0 to 1, where 1 = uniform)
    uniformity = mean_entropy / (max_entropy + eps)

    # Loss computation based on aggregation strategy (operates on filtered experts)
    if weights_tensor_filtered is None:
        # Original behavior (backward compatible) - no per-expert weighting
        if aggregation == "mean":
            # Original: average entropy across experts
            # Problem: can mask bad experts with good ones
            loss = -mean_entropy
        elif aggregation == "min":
            # Penalize the worst (lowest entropy) expert most strongly
            # This ensures all experts improve, not just the average
            min_entropy = entropy_per_expert_filtered.min()
            loss = -min_entropy
        elif aggregation == "mean+var":
            # Mean entropy with variance penalty
            # Encourages all experts to have similar (high) entropy
            entropy_variance = entropy_per_expert_filtered.var()
            variance_weight = 0.1  # Could be made configurable
            loss = -mean_entropy + variance_weight * entropy_variance
        elif aggregation == "separate":
            # Apply entropy loss independently to each expert
            # Each expert gets the full weight (no averaging)
            # This is effectively 2x stronger than "mean" but with equal distribution
            # Unlike "min" which only penalizes the worst expert,
            # this maintains good entropy while fixing bad entropy
            loss = -entropy_per_expert_filtered.sum()  # Sum of negative entropies
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}. "
                            f"Choose from: 'mean', 'min', 'mean+var', 'separate'")
    else:
        # Per-expert weighted loss
        if aggregation in ["mean", "separate"]:
            # Apply weights to each expert's entropy, then sum
            # Note: "mean" vs "separate" now only affects whether we had filtered by expert_indices
            # With per-expert weights, both aggregate by weighted sum
            weighted_entropies = -entropy_per_expert_filtered * weights_tensor_filtered
            loss = weighted_entropies.sum()
        elif aggregation == "min":
            # Weight the minimum entropy expert
            min_idx = entropy_per_expert_filtered.argmin()
            loss = -entropy_per_expert_filtered[min_idx] * weights_tensor_filtered[min_idx]
        elif aggregation == "mean+var":
            # Weight each expert's entropy, then add variance penalty
            weighted_entropies = -entropy_per_expert_filtered * weights_tensor_filtered
            entropy_variance = entropy_per_expert_filtered.var()
            variance_weight = 0.1  # Could be made configurable
            loss = weighted_entropies.sum() + variance_weight * entropy_variance
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}. "
                            f"Choose from: 'mean', 'min', 'mean+var', 'separate'")

    # Metrics for logging
    metrics = {
        'dispatch_entropy': mean_entropy.detach(),
        'dispatch_entropy_max': max_entropy.detach(),
        'dispatch_uniformity': uniformity.detach(),
    }

    # Add variance metric if using variance-based aggregation
    if aggregation == "mean+var":
        metrics['entropy_variance'] = entropy_variance.detach()

    # Per-expert entropy for monitoring
    for i in range(num_experts):
        metrics[f'expert_{i}_entropy'] = entropy_per_expert[i].detach()

    # Log per-expert weighted losses if using per-expert weights
    if weights_tensor_filtered is not None:
        # Log the weights being applied
        for i in range(len(weights_tensor_filtered)):
            expert_idx = expert_indices[i] if expert_indices is not None else i
            metrics[f'expert_{expert_idx}_entropy_weight'] = weights_tensor_filtered[i].detach()

        # Log the individual weighted losses (before summing)
        if aggregation in ["mean", "separate"]:
            weighted_losses = -entropy_per_expert_filtered * weights_tensor_filtered
            for i in range(len(weighted_losses)):
                expert_idx = expert_indices[i] if expert_indices is not None else i
                metrics[f'expert_{expert_idx}_weighted_loss'] = weighted_losses[i].detach()

    return loss, metrics


def compute_three_losses(
    pred_tok: Optional[torch.Tensor],
    pred_pix: Optional[torch.Tensor],
    T: Optional[torch.Tensor],
    img: torch.Tensor,
    mask: torch.Tensor,
    cfg: Dict[str, Any],
    model: Optional[torch.nn.Module] = None,
    accum_iter: int = 1,
    current_micro_step: int = 0,
    use_mask_tokens: Optional[bool] = None,
    combine_weights: Optional[torch.Tensor] = None,
    ids_restore: Optional[torch.Tensor] = None,  # for MoE loss weighting alignment
    combine_weights_dict: Optional[Dict[int, torch.Tensor]] = None,  # all MoE blocks for per-block regularization
    head_combine_weights: Optional[torch.Tensor] = None,  # per-loss selected/aggregated weights for head loss
    pixel_combine_weights: Optional[torch.Tensor] = None,  # per-loss selected/aggregated weights for pixel loss
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Computes the MEDiC losses with full conditional support and normalization options.

    When head_combine_weights/pixel_combine_weights are provided (from multi-block
    aggregation), they are used for the respective loss. Otherwise falls back to
    combine_weights (last block) for backward compatibility.
    """
    loss_dict: Dict[str, torch.Tensor] = {}
    losses_cfg = cfg.get("losses", {})

    # Get loss weighting method
    loss_weighting_method = losses_cfg.get("loss_weighting_method", "literal")

    # Collect individual losses and their names
    individual_losses = []
    loss_names = []

    # Get normalization method
    norm_method = losses_cfg.get("normalization_method",
                                 "variance")  # but we are not using this

    # Read MoE config once for all losses (to avoid locals() usage)
    moe_normalize_weights = losses_cfg.get("moe_normalize_weights", True)
    loss_expert_indices = losses_cfg.get("moe_loss_expert_indices", None)
    weight_type = losses_cfg.get("moe_weight_type", "combine")
    normalize_per_image = losses_cfg.get("moe_normalize_per_image", True)
    moe_detach_loss_weights = losses_cfg.get("moe_detach_loss_weights", False)

    # Validate loss_expert_indices once for duplicates
    if loss_expert_indices is not None:
        if len(loss_expert_indices) != len(set(loss_expert_indices)):
            raise ValueError(
                f"losses.moe_loss_expert_indices contains duplicates: {loss_expert_indices}. "
                "Each expert should appear at most once. "
                "Duplicates would cause multiple losses to use the same expert, defeating the purpose of multi-expert isolation."
            )

    # Check if using sparse encoder mode
    # Option 1: Use explicitly passed value (most efficient)
    if use_mask_tokens is not None:
        # Use the explicitly passed value - this is the most efficient path
        pass
    # Option 2: Get from model if not explicitly passed (backward compatibility)
    elif model is not None:
        # Try to get from model (handles both regular and DDP wrapped)
        if hasattr(model, 'student') and hasattr(model.student, 'use_mask_tokens'):
            use_mask_tokens = model.student.use_mask_tokens
        elif hasattr(model, 'module') and hasattr(model.module, 'student'):
            use_mask_tokens = model.module.student.use_mask_tokens
        else:
            # Fallback to config
            use_mask_tokens = cfg.get("model", {}).get("student", {}).get("use_mask_tokens", True)
    else:
        raise ValueError("use_mask_tokens must be provided either explicitly or via the model")

    # Get MoE weighting flags
    moe_weight_head = losses_cfg.get("moe_weight_head_loss", False)
    moe_weight_pixel = losses_cfg.get("moe_weight_pixel_loss", False)

    # Validation: pixel loss MoE weighting only works in dense mode
    # In sparse mode, encoder only processes visible patches but pixel loss operates on masked patches
    # These are disjoint sets, so no MoE weights exist for masked patches
    if not use_mask_tokens and moe_weight_pixel:
        raise ValueError(
            "Sparse mode (use_mask_tokens=False) cannot use MoE weighting for pixel loss. "
            "In sparse mode, the encoder only sees visible patches but pixel loss operates on "
            "masked patches (disjoint sets). Set moe_weight_pixel_loss=False or use dense mode."
        )

    # Check if MoE weighting is requested but combine_weights not provided
    if (moe_weight_head or moe_weight_pixel) and combine_weights is None:
        raise ValueError(
            "MoE loss weighting is enabled but combine_weights not provided. "
            "Ensure the model forward pass returns combine weights when MoE weighting is enabled."
        )

    # Per-loss weight selection: use per-loss weights if provided, else fall back to
    # combine_weights (last block) for backward compatibility
    head_weights_for_loss = head_combine_weights if head_combine_weights is not None else combine_weights
    pixel_weights_for_loss = pixel_combine_weights if pixel_combine_weights is not None else combine_weights

    # 1. Head/Distillation loss (conditional)
    if losses_cfg.get("use_head_loss", True) and pred_tok is not None and T is not None:
        # Teacher targets are already normalized in train.py when normalize_targets=True
        # Apply normalization to predictions if enabled
        pred_tok_normalized = pred_tok
        if losses_cfg.get("normalize_predictions", False):
            pred_tok_normalized = apply_normalization(pred_tok, norm_method)

        # Get head loss type from nested config
        head_loss_type = losses_cfg.get("head", {}).get("type", "smooth_l1")

        # Check if MoE weighting is enabled for head loss
        if moe_weight_head and head_weights_for_loss is not None:
            # MoE-weighted head loss (both dense and sparse modes supported)
            # Compute per-token loss without reduction, then apply MoE weighting
            # Dense mode: pred/target shape [B, N, D], loss on masked patches
            # Sparse mode: pred/target shape [B, num_visible, D], loss on visible patches
            pred_patches = pred_tok_normalized[:, 1:, :]  # Skip CLS

            # Teacher is already filtered and aligned in train.py for both dense and sparse modes
            # Dense mode: T is [B, N+1, D] - all patches + CLS
            # Sparse mode: T is [B, num_visible+1, D] - visible patches + CLS (already filtered and aligned!)
            # Just remove CLS token (first position)
            target_patches = T[:, 1:, :]

            # Compute per-token per-feature loss
            if head_loss_type == "l1":
                loss_per_token = torch.abs(pred_patches - target_patches)  # [B, N, D]
            elif head_loss_type == "smooth_l1":
                beta = losses_cfg.get("head", {}).get("beta", 1.0)
                loss_fn = torch.nn.SmoothL1Loss(beta=beta, reduction="none")
                loss_per_token = loss_fn(pred_patches, target_patches)  # [B, N, D]
            elif head_loss_type == "l2":
                loss_per_token = (pred_patches - target_patches) ** 2  # [B, N, D]
            else:
                raise ValueError(f"Unknown head loss type: {head_loss_type}")

            # Compute unweighted loss for logging (before MoE weighting)
            L_head_unweighted = loss_per_token.mean()

            # Apply MoE weighting with Expert 1 (slot 0) for distillation
            # In dense mode: weight masked patches (same as standard head loss)
            # pass ids_restore for shuffling alignment
            # Use MoE config variables defined at the beginning of the function

            L_head = apply_moe_loss_weighting(
                loss_per_token, head_weights_for_loss, mask,
                expert_idx=0, use_mask_tokens=use_mask_tokens,
                ids_restore=ids_restore,
                normalize_weights=moe_normalize_weights,
                loss_expert_indices=loss_expert_indices,
                weight_type=weight_type,
                normalize_per_image=normalize_per_image,
                detach_weights=moe_detach_loss_weights
            )
        elif use_mask_tokens:
            # BEiT mode: use masked loss functions (standard path)
            if head_loss_type == "l1":
                L_head = masked_l1_loss(
                    pred_tok_normalized[:, 1:, :],
                    T[:, 1:, :],
                    mask
                )
            elif head_loss_type == "smooth_l1":
                L_head = masked_smooth_l1_loss(
                    pred_tok_normalized[:, 1:, :],
                    T[:, 1:, :],
                    mask,
                    beta=losses_cfg.get("head", {}).get("beta", 1.0)
                )
            elif head_loss_type == "l2":
                L_head = masked_mse_loss(
                    pred_tok_normalized[:, 1:, :],
                    T[:, 1:, :],
                    mask
                )
            else:
                raise ValueError(f"Unknown head loss type: {head_loss_type}. Must be 'l1', 'l2', or 'smooth_l1'")
        else:
            # MAE sparse mode: no masking needed (already sparse)
            # Skip CLS token
            pred_patches = pred_tok_normalized[:, 1:, :]
            T_patches = T[:, 1:, :]

            if head_loss_type == "l1":
                L_head = F.l1_loss(pred_patches, T_patches, reduction='mean')
            elif head_loss_type == "smooth_l1":
                beta = losses_cfg.get("head", {}).get("beta", 1.0)
                L_head = F.smooth_l1_loss(pred_patches, T_patches, beta=beta, reduction='mean')
            elif head_loss_type == "l2":
                L_head = F.mse_loss(pred_patches, T_patches, reduction='mean')
            else:
                raise ValueError(f"Unknown head loss type: {head_loss_type}. Must be 'l1', 'l2', or 'smooth_l1'")

        loss_dict["L_head"] = L_head
        # Add unweighted loss if MoE weighting was applied
        if moe_weight_head and head_weights_for_loss is not None:
            loss_dict["L_head_unweighted"] = L_head_unweighted
        individual_losses.append(L_head)
        loss_names.append("head")

    # 2. Decoder pixel loss (conditional)
    use_pixel = losses_cfg.get("use_decoder_loss", False)
    if use_pixel and pred_pix is not None:
        patch_size = cfg["model"]["student"]["patch_size"]
        with torch.no_grad():
            pixel_target = _get_patchify_img()(img, patch_size=patch_size)
            if losses_cfg.get("decoder_norm_pix_loss", True):
                mean = pixel_target.mean(dim=-1, keepdim=True)
                var = pixel_target.var(dim=-1, keepdim=True)
                pixel_target = (pixel_target - mean) / (var + 1.0e-6) ** 0.5

        # DEBUG: Log decoder output statistics for debugging
        # if torch.rand(1).item() < 0.01:  # Log 1% of batches for better debugging
        #     print(f"🔍 DECODER DEBUG:")
        #     print(f"  pred_pix: mean={pred_pix.mean().item():.4f}, std={pred_pix.std().item():.4f}, range=[{pred_pix.min().item():.4f}, {pred_pix.max().item():.4f}]")
        #     print(f"  pixel_target: mean={pixel_target.mean().item():.4f}, std={pixel_target.std().item():.4f}, range=[{pixel_target.min().item():.4f}, {pixel_target.max().item():.4f}]")
        #     print(f"  mask ratio: {mask.float().mean().item():.3f}, decoder_norm_pix_loss: {losses_cfg.get('decoder_norm_pix_loss', True)}")
        #     print(f"  shapes: pred_pix={pred_pix.shape}, pixel_target={pixel_target.shape}, mask={mask.shape}")

        # Use L1, smooth L1, or L2 loss based on config
        pixel_loss_type = losses_cfg.get("decoder", {}).get("type", "l2")

        # Check if MoE weighting is enabled for pixel loss
        # NOTE: In sparse mode, MoE weighting for pixel loss is IMPOSSIBLE because:
        # - Encoder only processes visible patches (mask==0)
        # - Pixel loss operates on masked patches (mask==1)
        # - These are disjoint sets, so no MoE weights exist for masked patches
        can_use_moe_weighting = moe_weight_pixel and pixel_weights_for_loss is not None and use_mask_tokens

        if can_use_moe_weighting:
            # MoE-weighted pixel loss (ONLY in dense mode)
            # Compute per-token loss without reduction, then apply MoE weighting

            # Compute per-token per-feature loss
            if pixel_loss_type == "l1":
                loss_per_token = torch.abs(pred_pix - pixel_target)  # [B, N, patch_dim]
            elif pixel_loss_type == "smooth_l1":
                beta = losses_cfg.get("decoder", {}).get("beta", 1.0)
                loss_fn = torch.nn.SmoothL1Loss(beta=beta, reduction="none")
                loss_per_token = loss_fn(pred_pix, pixel_target)  # [B, N, patch_dim]
            elif pixel_loss_type == "l2":
                loss_per_token = (pred_pix - pixel_target) ** 2  # [B, N, patch_dim]
            else:
                raise ValueError(f"Unknown decoder loss type: {pixel_loss_type}")

            # Compute unweighted loss for logging (before MoE weighting)
            L_pix_unweighted = loss_per_token.mean()

            # Apply MoE weighting with Expert 2 (slot 1) for reconstruction
            # Weight masked patches by their combine weights
            # pass ids_restore for shuffling alignment
            # Use MoE config variables defined at the beginning of the function

            L_pix = apply_moe_loss_weighting(
                loss_per_token, pixel_weights_for_loss, mask,
                expert_idx=1, use_mask_tokens=use_mask_tokens,
                ids_restore=ids_restore,
                normalize_weights=moe_normalize_weights,
                loss_expert_indices=loss_expert_indices,
                weight_type=weight_type,
                normalize_per_image=normalize_per_image,
                detach_weights=moe_detach_loss_weights
            )
        else:
            # Standard pixel loss (no MoE weighting)
            # Note: Sparse + pixel MoE weighting already validated at line 930 (throws error)
            if pixel_loss_type == "l1":
                L_pix = masked_l1_loss(pred_pix, pixel_target, mask)
            elif pixel_loss_type == "smooth_l1":
                L_pix = masked_smooth_l1_loss(
                    pred_pix, pixel_target, mask,
                    beta=losses_cfg.get("decoder", {}).get("beta", 1.0)
                )
            elif pixel_loss_type == "l2":
                L_pix = masked_mse_loss(pred_pix, pixel_target, mask)
            else:
                raise ValueError(f"Unknown decoder loss type: {pixel_loss_type}. Must be 'l1', 'l2', or 'smooth_l1'")

        loss_dict["L_pix"] = L_pix
        # Add unweighted loss if MoE weighting was applied
        if can_use_moe_weighting:
            loss_dict["L_pix_unweighted"] = L_pix_unweighted
        individual_losses.append(L_pix)
        loss_names.append("decoder")

    # 3. CLS token loss (conditional) - Supporting both CE and cosine similarity
    if losses_cfg.get("use_cls_loss", False) and pred_tok is not None and T is not None:
        # Extract CLS tokens (T is normalized in train.py when normalize_targets=True)
        target_cls = T[:, 0, :]  # Teacher CLS (normalized if enabled)
        pred_cls = pred_tok[:, 0, :]  # Student CLS (raw)

        # Choose loss type based on config
        cls_loss_type = losses_cfg.get("cls", {}).get("type", "cross_entropy")

        if cls_loss_type == "cosine":
            # Use cosine similarity loss for normalized embeddings
            temperature = losses_cfg.get("cls", {}).get("temperature", 0.1)
            L_cls = cls_cosine_loss(pred_cls, target_cls, temperature=temperature)
        elif cls_loss_type == "cross_entropy":
            # Original MEDiC approach: direct cross-entropy on normalized teacher vs raw student
            L_cls = cls_ce_loss(pred_cls, target_cls)
        else:
            raise ValueError(f"Unknown cls_loss_type: {cls_loss_type}. Must be 'cross_entropy' or 'cosine'")

        loss_dict["L_cls"] = L_cls
        individual_losses.append(L_cls)
        loss_names.append("cls")

    # Compute total loss based on weighting method
    if not individual_losses:
        # No losses computed - this should not happen in normal operation
        raise ValueError("At least one loss must be enabled")

    if loss_weighting_method == "literal":
        # Use fixed weights from config
        total_loss = torch.tensor(0.0, device=img.device, dtype=img.dtype)
        if "L_head" in loss_dict:
            total_loss = total_loss + losses_cfg.get("head_loss_weight", 1.0) * loss_dict["L_head"]
        if "L_pix" in loss_dict:
            # Use pixel_loss_weight for the pixel reconstruction (decoder) loss
            decoder_weight = losses_cfg.get("pixel_loss_weight", 1.0)
            total_loss = total_loss + decoder_weight * loss_dict["L_pix"]
        if "L_cls" in loss_dict:
            total_loss = total_loss + losses_cfg.get("cls_loss_weight", 1.0) * loss_dict["L_cls"]

    else:
        raise ValueError(f"Unknown loss_weighting_method: {loss_weighting_method}")

    # MoE Scale Regularization (to prevent last block collapse with loss weighting)
    # Only apply if MoE is enabled and regularization is configured
    if (cfg.get('model', {}).get('student', {}).get('use_soft_moe', False) and
        cfg.get('losses', {}).get('moe_scale_regularization', 'none') != 'none' and
        model is not None):

        scale_reg_type = cfg['losses']['moe_scale_regularization']
        scale_penalty = compute_moe_scale_regularization(
            model,
            regularization_type=scale_reg_type,
            device=total_loss.device
        )

        penalty_weight = cfg.get('losses', {}).get('moe_scale_penalty_weight', 0.01)
        total_loss = total_loss + penalty_weight * scale_penalty

        # Add to log dict for monitoring
        loss_dict['moe_scale_penalty'] = scale_penalty

    # MoE Importance Loss (to prevent expert collapse by balancing expert usage)
    # Only apply if MoE is enabled and importance loss is configured
    # Now applies to ALL MoE blocks, not just the last one
    if (cfg.get('losses', {}).get('use_importance_loss', False) and
        (combine_weights_dict is not None or combine_weights is not None)):

        # Get weight type for importance computation (should match loss weighting)
        weight_type_for_importance = cfg.get('losses', {}).get('moe_weight_type', 'combine')

        # Check if we should regularize all blocks or just the last one
        regularize_all_blocks = cfg.get('losses', {}).get('moe_regularize_all_blocks', False)

        # Check if we should regularize all experts or only selected ones
        regularize_all_experts = cfg.get('losses', {}).get('moe_regularize_all_experts', True)

        # Get expert indices for selective regularization
        if regularize_all_experts:
            # Regularize all experts (pass None to use all)
            expert_indices_for_reg = None
        else:
            # Only regularize experts specified in moe_loss_expert_indices
            expert_indices_for_reg = cfg.get('losses', {}).get('moe_loss_expert_indices', None)

        # Compute importance loss for ALL blocks if dict available and flag is set, otherwise just last block
        if regularize_all_blocks and combine_weights_dict is not None and len(combine_weights_dict) > 0:
            # Apply to all MoE blocks and average
            block_importance_losses = []
            block_importance_metrics = {}

            for block_idx in sorted(combine_weights_dict.keys()):
                block_weights = combine_weights_dict[block_idx]
                block_loss, block_metrics = compute_importance_loss(
                    block_weights,
                    weight_type=weight_type_for_importance,
                    expert_indices=expert_indices_for_reg
                )
                block_importance_losses.append(block_loss)

                # Store per-block metrics
                for key, val in block_metrics.items():
                    block_importance_metrics[f"block_{block_idx}_{key}"] = val

            # Average importance loss across all blocks
            importance_loss = torch.stack(block_importance_losses).mean()

            # Also compute overall metrics across all blocks
            # Extract the specific weight type from each block's dict before concatenating
            all_blocks_weights = torch.cat([
                combine_weights_dict[idx][weight_type_for_importance] if isinstance(combine_weights_dict[idx], dict)
                else combine_weights_dict[idx]
                for idx in sorted(combine_weights_dict.keys())
            ], dim=0)
            _, overall_metrics = compute_importance_loss(
                all_blocks_weights,
                weight_type=weight_type_for_importance,
                expert_indices=expert_indices_for_reg
            )

            # Combine metrics: overall + per-block
            importance_metrics = overall_metrics
            importance_metrics.update(block_importance_metrics)

        else:
            # Smart block selection: regularize the block(s) used for loss weighting,
            # NOT just the last block. Mirrors the entropy loss smart block selection.
            losses_cfg = cfg.get('losses', {})
            loss_blocks = set()
            head_block = losses_cfg.get('moe_head_loss_block', None)
            pixel_block = losses_cfg.get('moe_pixel_loss_block', None)
            if head_block is not None:
                if isinstance(head_block, (list, tuple)):
                    loss_blocks.update(head_block)
                else:
                    loss_blocks.add(head_block)
            if pixel_block is not None:
                if isinstance(pixel_block, (list, tuple)):
                    loss_blocks.update(pixel_block)
                else:
                    loss_blocks.add(pixel_block)

            if loss_blocks and combine_weights_dict is not None and len(combine_weights_dict) > 0:
                # Regularize the specific block(s) used for loss weighting
                block_importance_losses = []
                block_importance_metrics = {}

                for block_idx in sorted(loss_blocks):
                    if block_idx in combine_weights_dict:
                        block_weights = combine_weights_dict[block_idx]
                        block_loss, block_metrics = compute_importance_loss(
                            block_weights,
                            weight_type=weight_type_for_importance,
                            expert_indices=expert_indices_for_reg
                        )
                        block_importance_losses.append(block_loss)
                        for key, val in block_metrics.items():
                            block_importance_metrics[f"block_{block_idx}_{key}"] = val

                if block_importance_losses:
                    importance_loss = torch.stack(block_importance_losses).mean()
                    # Also compute overall metrics from the selected blocks
                    selected_weights = torch.cat([
                        combine_weights_dict[idx][weight_type_for_importance]
                        if isinstance(combine_weights_dict[idx], dict)
                        else combine_weights_dict[idx]
                        for idx in sorted(loss_blocks) if idx in combine_weights_dict
                    ], dim=0)
                    _, overall_metrics = compute_importance_loss(
                        selected_weights,
                        weight_type=weight_type_for_importance,
                        expert_indices=expert_indices_for_reg
                    )
                    importance_metrics = overall_metrics
                    importance_metrics.update(block_importance_metrics)
                    importance_metrics['importance_loss_blocks'] = torch.tensor(float(len(loss_blocks)))
                else:
                    # Blocks not found in dict, fall back to last block
                    importance_loss, importance_metrics = compute_importance_loss(
                        combine_weights,
                        weight_type=weight_type_for_importance,
                        expert_indices=expert_indices_for_reg
                    )
                    importance_metrics['importance_last_block_only'] = torch.tensor(1.0)
            else:
                # No specific blocks configured — backward compatible last-block behavior
                importance_loss, importance_metrics = compute_importance_loss(
                    combine_weights,
                    weight_type=weight_type_for_importance,
                    expert_indices=expert_indices_for_reg
                )
                importance_metrics['importance_last_block_only'] = torch.tensor(1.0)

        # Get importance loss weight from config
        importance_weight = cfg.get('losses', {}).get('importance_loss_weight', 0.1)
        total_loss = total_loss + importance_weight * importance_loss

        # Add metrics to loss dict for monitoring
        loss_dict['L_importance'] = importance_loss
        for key, val in importance_metrics.items():
            loss_dict[key] = val

    # MoE Dispatch Entropy Loss (to encourage uniform token contributions per expert)
    # Only apply if MoE is enabled and entropy loss is configured
    # Now applies to ALL MoE blocks, not just the last one
    if (cfg.get('losses', {}).get('use_dispatch_entropy_loss', False) and
        (combine_weights_dict is not None or combine_weights is not None)):

        # Get weight type for entropy computation (typically "dispatch")
        weight_type_for_entropy = cfg.get('losses', {}).get('moe_entropy_weight_type', 'dispatch')

        # Check if we should regularize all blocks or just the last one (same flag as importance loss)
        regularize_all_blocks = cfg.get('losses', {}).get('moe_regularize_all_blocks', False)

        # Check if we should regularize all experts or only selected ones (same flag as importance loss)
        regularize_all_experts = cfg.get('losses', {}).get('moe_regularize_all_experts', True)

        # Get expert indices for selective regularization
        if regularize_all_experts:
            # Regularize all experts (pass None to use all)
            expert_indices_for_reg = None
        else:
            # Only regularize experts specified in moe_loss_expert_indices
            expert_indices_for_reg = cfg.get('losses', {}).get('moe_loss_expert_indices', None)

        # Get entropy loss weight from config (can be scalar or list for per-expert weighting)
        entropy_weight = cfg.get('losses', {}).get('dispatch_entropy_loss_weight', 0.01)

        # Compute entropy loss for ALL blocks if dict available and flag is set, otherwise just last block
        if regularize_all_blocks and combine_weights_dict is not None and len(combine_weights_dict) > 0:
            # Apply to all MoE blocks and average
            block_entropy_losses = []
            block_entropy_metrics = {}

            for block_idx in sorted(combine_weights_dict.keys()):
                block_weights = combine_weights_dict[block_idx]
                # Get aggregation strategy from config (default to "mean" for backward compatibility)
                entropy_aggregation = cfg.get('losses', {}).get('moe_entropy_aggregation', 'mean')
                block_loss, block_metrics = compute_dispatch_entropy_loss(
                    block_weights,
                    weight_type=weight_type_for_entropy,
                    aggregation=entropy_aggregation,
                    expert_indices=expert_indices_for_reg,
                    expert_weights=entropy_weight  # NEW: Pass per-expert weights
                )
                block_entropy_losses.append(block_loss)

                # Store per-block metrics
                for key, val in block_metrics.items():
                    block_entropy_metrics[f"block_{block_idx}_{key}"] = val

            # Average entropy loss across all blocks
            entropy_loss = torch.stack(block_entropy_losses).mean()

            # Also compute overall metrics across all blocks
            # Extract the specific weight type from each block's dict before concatenating
            all_blocks_weights = torch.cat([
                combine_weights_dict[idx][weight_type_for_entropy] if isinstance(combine_weights_dict[idx], dict)
                else combine_weights_dict[idx]
                for idx in sorted(combine_weights_dict.keys())
            ], dim=0)
            entropy_aggregation = cfg.get('losses', {}).get('moe_entropy_aggregation', 'mean')
            _, overall_metrics = compute_dispatch_entropy_loss(
                all_blocks_weights,
                weight_type=weight_type_for_entropy,
                aggregation=entropy_aggregation,
                expert_indices=expert_indices_for_reg,
                expert_weights=entropy_weight  # NEW: Pass per-expert weights
            )

            # Combine metrics: overall + per-block
            entropy_metrics = overall_metrics
            entropy_metrics.update(block_entropy_metrics)
            # Add indicator showing number of blocks used
            entropy_metrics['entropy_num_blocks'] = torch.tensor(float(len(combine_weights_dict)))

        else:
            # Smart block selection: regularize the block(s) used for loss weighting,
            # NOT just the last block. When moe_head_loss_block or moe_pixel_loss_block
            # is set, those blocks should be entropy-regularized to prevent degenerate routing.
            entropy_aggregation = cfg.get('losses', {}).get('moe_entropy_aggregation', 'mean')

            # Collect unique loss-relevant blocks from config
            loss_blocks = set()
            head_block = losses_cfg.get('moe_head_loss_block', None)
            pixel_block = losses_cfg.get('moe_pixel_loss_block', None)
            if head_block is not None:
                if isinstance(head_block, (list, tuple)):
                    loss_blocks.update(head_block)
                else:
                    loss_blocks.add(head_block)
            if pixel_block is not None:
                if isinstance(pixel_block, (list, tuple)):
                    loss_blocks.update(pixel_block)
                else:
                    loss_blocks.add(pixel_block)

            if loss_blocks and combine_weights_dict is not None and len(combine_weights_dict) > 0:
                # Regularize the specific block(s) used for loss weighting
                block_entropy_losses = []
                block_entropy_metrics = {}

                for block_idx in sorted(loss_blocks):
                    if block_idx in combine_weights_dict:
                        block_weights = combine_weights_dict[block_idx]
                        block_loss, block_metrics = compute_dispatch_entropy_loss(
                            block_weights,
                            weight_type=weight_type_for_entropy,
                            aggregation=entropy_aggregation,
                            expert_indices=expert_indices_for_reg,
                            expert_weights=entropy_weight
                        )
                        block_entropy_losses.append(block_loss)
                        for key, val in block_metrics.items():
                            block_entropy_metrics[f"block_{block_idx}_{key}"] = val

                if block_entropy_losses:
                    entropy_loss = torch.stack(block_entropy_losses).mean()
                    # Also compute overall metrics from the selected blocks
                    selected_weights = torch.cat([
                        combine_weights_dict[idx][weight_type_for_entropy]
                        if isinstance(combine_weights_dict[idx], dict)
                        else combine_weights_dict[idx]
                        for idx in sorted(loss_blocks) if idx in combine_weights_dict
                    ], dim=0)
                    _, overall_metrics = compute_dispatch_entropy_loss(
                        selected_weights,
                        weight_type=weight_type_for_entropy,
                        aggregation=entropy_aggregation,
                        expert_indices=expert_indices_for_reg,
                        expert_weights=entropy_weight
                    )
                    entropy_metrics = overall_metrics
                    entropy_metrics.update(block_entropy_metrics)
                    entropy_metrics['entropy_loss_blocks'] = torch.tensor(float(len(loss_blocks)))
                else:
                    # Blocks not found in dict, fall back to last block
                    entropy_loss, entropy_metrics = compute_dispatch_entropy_loss(
                        combine_weights,
                        weight_type=weight_type_for_entropy,
                        aggregation=entropy_aggregation,
                        expert_indices=expert_indices_for_reg,
                        expert_weights=entropy_weight
                    )
                    entropy_metrics['entropy_last_block_only'] = torch.tensor(1.0)
            else:
                # No specific blocks configured — use last block (backward compatible)
                entropy_loss, entropy_metrics = compute_dispatch_entropy_loss(
                    combine_weights,
                    weight_type=weight_type_for_entropy,
                    aggregation=entropy_aggregation,
                    expert_indices=expert_indices_for_reg,
                    expert_weights=entropy_weight
                )
                entropy_metrics['entropy_last_block_only'] = torch.tensor(1.0)

        # Weighting is now applied inside compute_dispatch_entropy_loss
        # So we just add the loss directly (no multiplication needed)
        total_loss = total_loss + entropy_loss

        # Add metrics to loss dict for monitoring
        loss_dict['L_dispatch_entropy'] = entropy_loss
        for key, val in entropy_metrics.items():
            loss_dict[key] = val

    loss_dict["total_loss"] = total_loss


    # Convert to scalars for logging
    log_dict = {}
    for k, v in loss_dict.items():
        log_dict[k] = v.item() if hasattr(v, "item") else v

    return total_loss, log_dict


# ── Backward-compat aliases (MEDiC scaffold naming) ──────────────────────────
# Allow MEDiC-style imports (`from src.utils.losses import compute_loss, patchify`)
# to keep working against the MEDiC_torch-ported API.
from src.utils.utils import patchify_img as patchify  # noqa: E402, F401


def compute_loss(pred_tok, pred_pix, T, img, mask, cfg):  # noqa: E305
    """MEDiC-scaffold-compatible compute_loss; thin shim over compute_three_losses.

    Injects use_mask_tokens from the config so callers don't need to pass it
    explicitly (MEDiC scaffold's signature didn't).
    """
    use_mt = bool(
        cfg.get("model", {}).get("student", {}).get("use_mask_tokens", False)
    )
    return compute_three_losses(
        pred_tok, pred_pix, T, img, mask, cfg, use_mask_tokens=use_mt
    )
