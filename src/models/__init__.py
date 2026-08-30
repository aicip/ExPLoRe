# prevents 'python -OO' from stripping docstrings we rely on
__all__ = ["build_teacher"]

from typing import Dict, Any

from .vision_transformer_mim import VisionTransformerMIM
from .decoder_mae import MAEDecoder


def build_student(cfg: Dict[str, Any]) -> VisionTransformerMIM:
    """Builds the student ViT model."""
    student_cfg = cfg["model"]["student"]

    # Check if sparse mode is requested
    use_mask_tokens = student_cfg.get("use_mask_tokens", True)

    # For sparse mode, we need to ensure relative position bias is disabled
    # because sparse mode has variable sequence length
    if not use_mask_tokens:
        # Sparse mode requires absolute position embeddings (either learnable or sincos)
        default_abs_pos = True
        default_sincos_pos = False  # Default to learnable, but allow sincos override
        default_rel_pos = False
        default_shared_rel = False
    else:
        # Dense mode can use either absolute or relative position embeddings
        default_abs_pos = False
        default_sincos_pos = False
        default_rel_pos = False
        default_shared_rel = True  # BEiT-style default

    model = VisionTransformerMIM(
        img_size=student_cfg["img_size"],
        patch_size=student_cfg["patch_size"],
        embed_dim=student_cfg["embed_dim"],
        depth=student_cfg["depth"],
        num_heads=student_cfg["num_heads"],
        mlp_ratio=student_cfg.get("mlp_ratio", 4.0),  # Regular MLP ratio for non-MoE blocks
        drop_path_rate=student_cfg.get("drop_path_rate", 0.1),
        init_values=student_cfg.get("init_values", 0.1),  # layer scale
        use_abs_pos_emb=student_cfg.get("use_abs_pos_emb", default_abs_pos),
        use_sincos_pos_emb=student_cfg.get("use_sincos_pos_emb", default_sincos_pos),
        use_shared_rel_pos_bias=student_cfg.get("use_shared_rel_pos_bias", default_shared_rel),
        use_rel_pos_bias=student_cfg.get("use_rel_pos_bias", default_rel_pos),
        use_mask_tokens=use_mask_tokens,
        # Soft MoE parameters
        use_soft_moe=student_cfg.get("use_soft_moe", False),
        moe_num_experts=student_cfg.get("moe_num_experts", 2),
        moe_slots_per_expert=student_cfg.get("moe_slots_per_expert", 1),
        moe_mlp_ratio=student_cfg.get("moe_mlp_ratio", 4.0),  # MLP ratio for MoE blocks
        moe_placement=student_cfg.get("moe_placement", "alternating"),
    )
    return model


def build_pixel_decoder(cfg: Dict[str, Any]) -> MAEDecoder:
    """Builds the MAE-style pixel decoder with same positional encoding as encoder."""
    decoder_cfg = cfg["model"]["decoder"]
    student_cfg = cfg["model"]["student"]
    student_embed_dim = student_cfg["embed_dim"]
    patch_size = student_cfg["patch_size"]
    img_size = student_cfg.get("img_size", 224)

    # Decoder can have its own position embedding configuration
    # If use_sincos_pos_emb is specified, use sincos (MAE-style)
    # Otherwise fall back to encoder's configuration for compatibility
    use_sincos = decoder_cfg.get("use_sincos_pos_emb", False)

    if use_sincos:
        # MAE-style: sincos absolute position embeddings
        use_abs_pos_emb = True
        use_rel_pos_bias = False
        use_shared_rel_pos_bias = False
    else:
        # Fall back to encoder's configuration
        use_abs_pos_emb = student_cfg.get("use_abs_pos_emb", False)
        use_rel_pos_bias = student_cfg.get("use_rel_pos_bias", False)
        use_shared_rel_pos_bias = student_cfg.get("use_shared_rel_pos_bias", True)

    return MAEDecoder(
        in_dim=student_embed_dim,
        embed_dim=decoder_cfg["decoder_embed_dim"],
        depth=decoder_cfg["decoder_depth"],
        num_heads=decoder_cfg["decoder_num_heads"],
        patch_size=patch_size,
        img_size=img_size,
        replace_mask_tokens=decoder_cfg.get("replace_mask_tokens", True),  # Default to True (MAE-style)
        use_abs_pos_emb=use_abs_pos_emb,
        use_rel_pos_bias=use_rel_pos_bias,
        use_shared_rel_pos_bias=use_shared_rel_pos_bias,
        use_sincos_pos_emb=use_sincos,  # Pass this flag to decoder
    )


def build_teacher(cfg: Dict[str, Any]):
    """Build the frozen CLIP teacher model, importing CLIP lazily."""
    from .clip_teacher import ClipTeacher

    teacher_cfg = cfg["model"]["teacher"]
    return ClipTeacher(
        model_name=teacher_cfg["name"],
        layer_extraction=teacher_cfg.get("layer_extraction", "last"),
        num_layers_to_extract=teacher_cfg.get("num_layers_to_extract", 1)
    )
