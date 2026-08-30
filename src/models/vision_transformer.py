"""
Vision Transformer for Masked Image Modeling (MEDiC).

Based on the BEiT architecture with support for:
- Absolute or sincos position embeddings
- Shared relative position bias
- Sparse encoding (MAE-style: drop masked patches)
- Dense encoding (BEiT-style: replace with mask tokens)
- Finetuning with classification head

Reference: "A Unified View of Masked Image Modeling" (arXiv:2210.10615)
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import DropPath, LayerNorm, Mlp, PatchEmbed, SwiGLU, trunc_normal_, use_fused_attn
except ImportError:  # minimal CPU verification environments
    from .timm_compat import DropPath, LayerNorm, Mlp, PatchEmbed, SwiGLU, trunc_normal_, use_fused_attn


def gen_relative_position_index(window_size: Tuple[int, int]) -> torch.Tensor:
    """Generate relative position index for attention bias."""
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(
        torch.meshgrid(
            [torch.arange(window_size[0]), torch.arange(window_size[1])], indexing="ij"
        )
    )
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(
        size=(window_area + 1,) * 2, dtype=relative_coords.dtype
    )
    relative_position_index[1:, 1:] = relative_coords.sum(-1)
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class Attention(nn.Module):
    """Multi-head self-attention with optional relative position bias."""

    fused_attn: torch.jit.Final[bool]  # type: ignore

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: Optional[Tuple[int, int]] = None,
        attn_head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False if qkv_bias else True)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.register_buffer("k_bias", torch.zeros(all_head_dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )
            self.register_buffer(
                "relative_position_index",
                gen_relative_position_index(window_size),
                persistent=False,
            )
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[  # type: ignore
            self.relative_position_index.view(-1)  # type: ignore
        ].view(
            self.window_size[0] * self.window_size[1] + 1,  # type: ignore
            self.window_size[0] * self.window_size[1] + 1,  # type: ignore
            -1,
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        qkv_bias = (
            torch.cat((self.q_bias, self.k_bias, self.v_bias))  # type: ignore
            if self.q_bias is not None
            else None
        )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            rel_pos_bias = None
            if self.relative_position_bias_table is not None:
                rel_pos_bias = self._get_rel_pos_bias()
                if shared_rel_pos_bias is not None:
                    rel_pos_bias = rel_pos_bias + shared_rel_pos_bias
            elif shared_rel_pos_bias is not None:
                rel_pos_bias = shared_rel_pos_bias

            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=rel_pos_bias, dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            if self.relative_position_bias_table is not None:
                attn = attn + self._get_rel_pos_bias()
            if shared_rel_pos_bias is not None:
                attn = attn + shared_rel_pos_bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attn_graph(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None):
        """Forward pass that also returns attention maps (for visualization)."""
        B, N, C = x.shape

        qkv_bias = (
            torch.cat((self.q_bias, self.k_bias, self.v_bias))  # type: ignore
            if self.q_bias is not None
            else None
        )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn_abs = attn.clone()

        if self.relative_position_bias_table is not None:
            attn = attn + self._get_rel_pos_bias()
        if shared_rel_pos_bias is not None:
            attn = attn + shared_rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn_softmax = attn
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_abs, attn_softmax


class Block(nn.Module):
    """Transformer block with LayerScale and DropPath."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        mlp_ratio: float = 4.0,
        scale_mlp: bool = False,
        swiglu_mlp: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        window_size: Optional[Tuple[int, int]] = None,
        attn_head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        if swiglu_mlp:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,  # type: ignore
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if init_values:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(
                self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias)
            )
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(
                self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias)
            )
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    def get_attn_graph(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None):
        """Forward pass returning attention maps for visualization."""
        y, attn_abs, attn_softmax = self.attn.get_attn_graph(
            self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias
        )
        mlp_out = self.mlp(self.norm2(x))

        if self.gamma_1 is None:
            x = x + self.drop_path1(y)
            x = x + self.drop_path2(mlp_out)
        else:
            x = x + self.drop_path1(self.gamma_1 * y)
            x = x + self.drop_path2(self.gamma_2 * mlp_out)

        return x, attn_abs, attn_softmax


class RelativePositionBias(nn.Module):
    """Shared relative position bias across all transformer layers."""

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_distance, num_heads)
        )
        self.register_buffer(
            "relative_position_index", gen_relative_position_index(window_size)
        )

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)  # type: ignore
        ].view(self.window_area + 1, self.window_area + 1, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class VisionTransformerMIM(nn.Module):
    """
    Student Vision Transformer for Masked Image Modeling.

    Supports both sparse encoding (MAE-style, drop masked patches) and
    dense encoding (BEiT-style, replace with mask tokens).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias: bool = True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=None,
        init_values=0.1,
        use_abs_pos_emb=False,
        use_sincos_pos_emb=False,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True,
        init_std=0.02,
        use_mask_tokens=True,
        # Finetuning parameters
        num_classes=0,
        use_mean_pooling=False,
        init_scale=0.001,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.use_mask_tokens = use_mask_tokens
        self.use_sincos_pos_emb = use_sincos_pos_emb

        # Validate sparse mode + relative position bias incompatibility
        if not use_mask_tokens and (use_rel_pos_bias or use_shared_rel_pos_bias):
            raise ValueError(
                "Sparse mode (use_mask_tokens=False) is incompatible with relative position bias. "
                "Relative position bias requires fixed sequence length, but sparse mode has variable "
                "sequence length depending on the mask. Either enable mask tokens (use_mask_tokens=True) "
                "or disable relative position bias."
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb or use_sincos_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.grid_size,
                num_heads=num_heads,
            )
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                mlp_ratio=mlp_ratio,
                scale_mlp=False,
                swiglu_mlp=False,
                proj_drop=0.0,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=LayerNorm,
                init_values=init_values,
                window_size=(self.patch_embed.grid_size if use_rel_pos_bias else None),
            )
            for i in range(depth)
        ])

        # Classification head (for finetuning)
        self.num_classes = num_classes
        self.norm = nn.Identity() if use_mean_pooling else LayerNorm(embed_dim)
        self.fc_norm = LayerNorm(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_std = init_std
        self.init_scale = init_scale

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            if self.use_sincos_pos_emb:
                from src.utils.pos_embed import get_2d_sincos_pos_embed
                grid_size = int(num_patches**0.5)
                pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
                pos_embed_tensor = torch.tensor(pos_embed, dtype=torch.float32)
                self.pos_embed.data.copy_(pos_embed_tensor.unsqueeze(0))
                self.pos_embed.requires_grad = False
            else:
                trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=self.init_std)
            self.head.weight.data.mul_(self.init_scale)
            self.head.bias.data.mul_(self.init_scale)

        self.fix_init_weight()

        if self.use_mask_tokens:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.mask_token, std=self.init_std)
        else:
            self.mask_token = None
        self.patch_size = self.patch_embed.patch_size[0]  # type: ignore

        self.ids_restore = None

    def fix_init_weight(self):
        """Rescale attention projection and MLP fc2 weights for stable training."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def get_intermediate_layers(self, x: torch.Tensor, use_last_norm: bool = False) -> list:
        """Extract intermediate layer features for linear probing (BEiT2 protocol)."""
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        intermediate_outputs = []
        shared_rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for block in self.blocks:
            x = block(x, shared_rel_pos_bias=shared_rel_pos_bias)
            if use_last_norm:
                intermediate_outputs.append(self.norm(x))
            else:
                intermediate_outputs.append(x)

        return intermediate_outputs

    @torch.jit.ignore  # type: ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed|rel_pos_bias",
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange patches back into an image. x: (N, L, patch_size**2 * 3) -> (N, 3, H, W)"""
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_generator=None,
        return_all_tokens: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            img: Input images [B, 3, H, W]
            mask: Boolean mask [B, N] where True = masked
            mask_generator: Masking generator (for shuffle indices in sparse mode)
            return_all_tokens: If True, return all tokens; if False, return CLS only

        Returns:
            x: Encoded features [B, N', D] where N' depends on masking mode
            ids_restore: Indices to restore patch order (sparse mode) or mask (dense mode)
        """
        # Finetuning mode: no masking, just classification
        if mask is None and self.num_classes > 0:
            x = self.patch_embed(img)
            batch_size = x.size(0)

            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            for blk in self.blocks:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias)

            if self.fc_norm is not None:
                x = self.fc_norm(x[:, 1:, :].mean(dim=1))
            else:
                x = self.norm(x)
                x = x[:, 0]

            return self.head(x)  # type: ignore

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        x = self.patch_embed(img)
        batch_size, num_patches, embed_dim = x.size()

        if mask is None:
            raise ValueError(
                "Mask must be provided to VisionTransformerMIM.forward(). "
                "For automatic mask generation, use MEDiCModel."
            )

        if self.use_mask_tokens:
            # Dense mode (BEiT-style): replace masked patches with mask tokens
            mask_token = self.mask_token.expand(batch_size, num_patches, -1)
            w = mask.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

            if self.pos_embed is not None:
                x = x + self.pos_embed[:, 1:, :]

            # Handle patch shuffling
            if hasattr(mask_generator, 'shuffle_patches') and mask_generator.shuffle_patches:
                if hasattr(mask_generator, 'ids_shuffle') and mask_generator.ids_shuffle is not None:
                    if mask_generator.ids_shuffle.dim() == 1:
                        x = x[:, mask_generator.ids_shuffle, :]
                    else:
                        ids_expanded = mask_generator.ids_shuffle.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        x = torch.gather(x, dim=1, index=ids_expanded)
                    self.ids_restore = mask_generator.ids_restore
                else:
                    self.ids_restore = None
            else:
                self.ids_restore = None

            self.visible_positions_shuffled = None

            cls_token = self.cls_token
            if self.pos_embed is not None:
                cls_token = cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        else:
            # Sparse mode (MAE-style): drop masked patches
            if self.pos_embed is not None:
                x = x + self.pos_embed[:, 1:, :]

            # Handle shuffling before removing masked patches
            if hasattr(mask_generator, 'shuffle_patches') and mask_generator.shuffle_patches:
                if hasattr(mask_generator, 'ids_shuffle') and mask_generator.ids_shuffle is not None:
                    if mask_generator.ids_shuffle.dim() == 1:
                        x = x[:, mask_generator.ids_shuffle, :]
                        mask = mask[:, mask_generator.ids_shuffle]
                    else:
                        ids_expanded = mask_generator.ids_shuffle.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        x = torch.gather(x, dim=1, index=ids_expanded)
                        mask = torch.gather(mask.float(), dim=1, index=mask_generator.ids_shuffle).bool()
                    self.ids_restore = mask_generator.ids_restore
                    self.ids_shuffle = mask_generator.ids_shuffle
                else:
                    self.ids_restore = None
                    self.ids_shuffle = None
            else:
                self.ids_restore = None
                self.ids_shuffle = None
                self.visible_positions_shuffled = None

            # Keep only visible (unmasked) patches
            visible_mask = ~mask
            batch_size_actual = x.shape[0]

            if batch_size_actual == 0:
                embed_dim = x.shape[-1] if x.shape[-1] > 0 else self.embed_dim
                x = torch.zeros(0, 1, embed_dim, device=x.device, dtype=x.dtype)
                x = self.norm(x)
                if not return_all_tokens:
                    return x[:, 0] if x.shape[1] > 0 else x.squeeze(1), None
                return x, mask

            num_visible_per_sample = visible_mask.sum(dim=1)

            if torch.all(num_visible_per_sample == num_visible_per_sample[0]):
                num_visible = num_visible_per_sample[0].item()

                if num_visible > 0:
                    visible_indices = visible_mask.nonzero(as_tuple=False)
                    patch_indices = visible_indices[:, 1].reshape(batch_size_actual, num_visible)
                    self.visible_positions_shuffled = patch_indices
                    x = torch.gather(
                        x, dim=1,
                        index=patch_indices.unsqueeze(-1).expand(-1, -1, embed_dim),
                    )
                else:
                    x = torch.zeros(batch_size_actual, 1, embed_dim, device=x.device, dtype=x.dtype)
                    self.visible_positions_shuffled = None
            else:
                raise RuntimeError(
                    f"Variable number of visible patches per sample: {num_visible_per_sample.tolist()}. "
                    f"All samples must have the same number of visible patches."
                )

            cls_token = self.cls_token
            if self.pos_embed is not None:
                cls_token = cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(batch_size_actual, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, shared_rel_pos_bias=rel_pos_bias)

        x = self.norm(x)

        if not return_all_tokens:
            return x[:, 0], None

        if self.use_mask_tokens:
            return x, mask
        else:
            return x, self.ids_restore
