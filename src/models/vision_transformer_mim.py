import math
from functools import partial
import warnings
from typing import Dict, Optional, Tuple, Union, Callable, Type
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import PatchEmbed, Mlp, SwiGLU, LayerNorm, DropPath, trunc_normal_, use_fused_attn
except ImportError:  # minimal CPU verification environments
    from .timm_compat import PatchEmbed, Mlp, SwiGLU, LayerNorm, DropPath, trunc_normal_, use_fused_attn


def gen_relative_position_index(window_size: Tuple[int, int]) -> torch.Tensor:
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(
        torch.meshgrid(
            [torch.arange(window_size[0]), torch.arange(window_size[1])], indexing="ij"
        )
    )  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = (
        coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(
        size=(window_area + 1,) * 2, dtype=relative_coords.dtype
    )
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class Attention(nn.Module):
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
            )  # 2*Wh-1 * 2*Ww-1, nH
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
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
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
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim

        if self.fused_attn:
            rel_pos_bias = None
            if self.relative_position_bias_table is not None:
                rel_pos_bias = self._get_rel_pos_bias()
                if shared_rel_pos_bias is not None:
                    rel_pos_bias = rel_pos_bias + shared_rel_pos_bias
            elif shared_rel_pos_bias is not None:
                rel_pos_bias = shared_rel_pos_bias

            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=rel_pos_bias,
                dropout_p=self.attn_drop.p,
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
        B, N, C = x.shape

        qkv_bias = (
            torch.cat((self.q_bias, self.k_bias, self.v_bias))  # type: ignore
            if self.q_bias is not None
            else None
        )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim

        # No fused attention by default
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
        use_soft_moe: bool = False,
        moe_num_experts: int = 2,
        moe_slots_per_expert: int = 1,
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
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.use_soft_moe = use_soft_moe

        if use_soft_moe:
            # Use Soft MoE layer instead of standard MLP
            from .soft_moe import SoftMoELayer
            self.mlp = SoftMoELayer(
                dim=dim,
                num_experts=moe_num_experts,
                slots_per_expert=moe_slots_per_expert,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=proj_drop,
            )
        elif swiglu_mlp:
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

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None, return_combine_weights: bool = False):
        """
        Forward pass through the block.

        Args:
            x: Input tokens [B, N, D]
            shared_rel_pos_bias: Optional relative position bias
            return_combine_weights: If True and using Soft MoE, return combine weights

        Returns:
            If return_combine_weights=True and use_soft_moe=True:
                (x, combine_weights): Output tokens and combine weights [B, N, num_slots]
            Otherwise:
                x: Output tokens only
        """
        # Attention block
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))

        # MLP block (Soft MoE or standard)
        combine_weights = None
        if self.use_soft_moe:
            # Soft MoE block - pass return_weights flag to control weight computation
            mlp_out, combine_weights = self.mlp(self.norm2(x), return_weights=return_combine_weights)
            if self.gamma_2 is None:
                x = x + self.drop_path2(mlp_out)
            else:
                x = x + self.drop_path2(self.gamma_2 * mlp_out)
        else:
            # Regular MLP block - no return_weights parameter needed
            if self.gamma_2 is None:
                x = x + self.drop_path2(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
            # combine_weights remains None for non-MoE blocks

        if return_combine_weights:
            return x, combine_weights
        return x

    def get_attn_graph(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None):
        # Compute attention graph using the get_attn_graph method of self.attn
        y, attn_abs, attn_softmax = self.attn.get_attn_graph(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias)

        # Handle MLP (which may be MoE returning tuple or standard MLP)
        if self.use_soft_moe:
            mlp_out, _ = self.mlp(self.norm2(x))  # MoE returns (output, combine_weights)
        else:
            mlp_out = self.mlp(self.norm2(x))  # Standard MLP returns output only

        if self.gamma_1 is None:
            x = x + self.drop_path1(y)
            x = x + self.drop_path2(mlp_out)
        else:
            x = x + self.drop_path1(self.gamma_1 * y)
            x = x + self.drop_path2(self.gamma_2 * mlp_out)

        return x, attn_abs, attn_softmax


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_distance, num_heads)
        )
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.register_buffer(
            "relative_position_index", gen_relative_position_index(window_size)
        )

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)  # type: ignore
        ].view(
            self.window_area + 1, self.window_area + 1, -1
        )  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class VisionTransformerMIM(nn.Module):
    """
    Student Vision Transformer for Masked Image Modeling.
    Based on the BEiT architecture.
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
        use_soft_moe=False,
        moe_num_experts=2,
        moe_slots_per_expert=1,
        moe_mlp_ratio=None,  # If specified, use different mlp_ratio for MoE blocks
        moe_placement="alternating",
        # Finetuning parameters
        num_classes=0,
        use_mean_pooling=False,
        init_scale=0.001,
    ):
        """Initializes the VisionTransformerMIM model."""
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.use_mask_tokens = use_mask_tokens
        self.use_sincos_pos_emb = use_sincos_pos_emb
        self.use_soft_moe = use_soft_moe
        self.moe_placement = moe_placement
        self.moe_mlp_ratio = moe_mlp_ratio if moe_mlp_ratio is not None else mlp_ratio

        # Validate that sparse mode and relative position bias are not used together
        if not use_mask_tokens and (use_rel_pos_bias or use_shared_rel_pos_bias):
            raise ValueError(
                "Sparse mode (use_mask_tokens=False) is incompatible with relative position bias. "
                "Relative position bias requires fixed sequence length, but sparse mode has variable "
                "sequence length depending on the mask. Either enable mask tokens (use_mask_tokens=True) "
                "or disable relative position bias (use_rel_pos_bias=False, use_shared_rel_pos_bias=False)."
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Create position embeddings if using absolute OR sincos
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

        # Determine MoE block placement pattern
        def should_use_moe(block_idx: int, depth: int, placement: str) -> bool:
            """Determine if a block should use MoE based on placement strategy."""
            if not use_soft_moe:
                return False

            if placement == "alternating":
                # Every other block (odd indices): 1,3,5,7,9,11
                return block_idx % 2 == 1
            elif placement == "second_half":
                # Second half of blocks (works for any depth, including odd)
                # For even depths (e.g., 12): blocks 6-11 (6+6 split)
                # For odd depths (e.g., 11): blocks 6-10 (6+5 split, first half gets extra)
                import math
                return block_idx >= math.ceil(depth / 2)
            elif placement == "all":
                # All blocks use MoE: 0,1,2,...,depth-1
                return True
            else:
                raise ValueError(
                    f"Invalid moe_placement: {placement}. "
                    f"Must be one of: 'alternating', 'second_half', 'all'"
                )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      qkv_bias=qkv_bias,
                      mlp_ratio=self.moe_mlp_ratio if should_use_moe(i, depth, moe_placement) else mlp_ratio,
                      scale_mlp=False,
                      swiglu_mlp=False,
                      proj_drop=0.0,
                      attn_drop=attn_drop_rate,
                      drop_path=dpr[i],
                      norm_layer=LayerNorm,
                      init_values=init_values,
                      window_size=(
                            self.patch_embed.grid_size if use_rel_pos_bias else None
                      ),
                      use_soft_moe=should_use_moe(i, depth, moe_placement),
                      moe_num_experts=moe_num_experts,
                      moe_slots_per_expert=moe_slots_per_expert)
                for i in range(depth)
            ]
        )

        # Track MoE block indices for dispatch weight extraction
        if use_soft_moe:
            self.moe_blocks = [i for i in range(depth) if should_use_moe(i, depth, moe_placement)]
            self.last_moe_block_idx = max(self.moe_blocks) if self.moe_blocks else None
        else:
            self.moe_blocks = []
            self.last_moe_block_idx = None

        # Classification head for finetuning
        self.num_classes = num_classes
        self.norm = nn.Identity() if use_mean_pooling else LayerNorm(embed_dim)
        self.fc_norm = LayerNorm(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_std = init_std
        self.init_scale = init_scale

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            if self.use_sincos_pos_emb:
                # Use fixed sincos position embeddings (MAE-style)
                from src.utils.pos_embed import get_2d_sincos_pos_embed
                grid_size = int(num_patches**0.5)
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim, grid_size, cls_token=True
                )
                # Convert numpy array to torch tensor and set as fixed (no gradients)
                pos_embed_tensor = torch.tensor(pos_embed, dtype=torch.float32)
                self.pos_embed.data.copy_(pos_embed_tensor.unsqueeze(0))
                self.pos_embed.requires_grad = False  # Fixed, not learnable
            else:
                # Use learnable position embeddings (default)
                trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)

        # Initialize classification head
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=self.init_std)
            self.head.weight.data.mul_(self.init_scale)
            self.head.bias.data.mul_(self.init_scale)

        self.fix_init_weight()

        # Only create mask token if using BEiT-style masking
        if self.use_mask_tokens:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # Initialize mask token based on mode:
            # - Dense mode (BEiT-style): use trunc_normal_ like BEiT
            # - Note: In sparse mode (MAE-style), encoder doesn't use mask tokens
            trunc_normal_(self.mask_token, std=self.init_std)
        else:
            # Sparse mode (MAE-style): no mask tokens in encoder
            self.mask_token = None
        self.patch_size = self.patch_embed.patch_size[0]  # type:ignore
        
        # For MAE-style sparse encoding, we need to track indices
        self.ids_restore = None

    def fix_init_weight(self):
        """Rescales the weights of the attention and MLP layers."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)

            # Handle both standard MLP and Soft MoE layers
            if layer.use_soft_moe:
                # For MoE, rescale fc2 of each expert
                from .soft_moe import SoftMoELayer
                assert isinstance(layer.mlp, SoftMoELayer)
                for expert in layer.mlp.experts:
                    rescale(expert.fc2.weight.data, layer_id + 1)
            else:
                # Standard MLP
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module):
        """Initializes weights for linear and layernorm layers."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def no_weight_decay(self):
        """Return parameter names that should not have weight decay applied."""
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        """
        Returns the number of transformer blocks in the model.
        """
        return len(self.blocks)

    def get_intermediate_layers(self, x: torch.Tensor, use_last_norm: bool = False) -> list:
        """
        Extract intermediate layer features for linear probing (following BEiT2 protocol).

        Args:
            x: Input tensor of shape (B, C, H, W)
            use_last_norm: If True, apply LayerNorm to features. If False, return raw features.
                          Default False matches BEiT2 default behavior.

        Returns:
            List of intermediate layer outputs, each of shape (B, N+1, D) where:
            - B: batch size
            - N: number of patches (H*W/patch_size^2)
            - D: embedding dimension
            - +1: for CLS token
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Collect outputs from all transformer blocks
        intermediate_outputs = []

        shared_rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for block in self.blocks:
            x = block(x, shared_rel_pos_bias=shared_rel_pos_bias)
            # Apply normalization only if requested (BEiT2 compatibility)
            if use_last_norm:
                intermediate_outputs.append(self.norm(x))
            else:
                intermediate_outputs.append(x)  # Raw features by default

        return intermediate_outputs

    @torch.jit.ignore  # type: ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r"^cls_token|pos_embed|patch_embed|rel_pos_bias",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )
        return matcher

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Rearranges a sequence of patches back into an image.
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward(self, img: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mask_generator = None,
                return_all_tokens: bool = True,
                return_combine_weights: bool = False
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[int, torch.Tensor]]]:

        # Finetuning mode: no masking, just classification
        if mask is None and self.num_classes > 0:
            x = self.patch_embed(img)
            batch_size, num_patches, embed_dim = x.size()

            # Add CLS token
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

            # Add position embeddings
            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            # Apply transformer blocks, collecting MoE weights if requested
            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            all_moe_weights = [] if return_combine_weights else None

            for i, blk in enumerate(self.blocks):
                is_moe_block = self.use_soft_moe and i in self.moe_blocks

                if return_combine_weights and is_moe_block:
                    x, block_weights = blk(x, shared_rel_pos_bias=rel_pos_bias, return_combine_weights=True)
                    if block_weights is not None:
                        all_moe_weights.append(block_weights)
                else:
                    x = blk(x, shared_rel_pos_bias=rel_pos_bias, return_combine_weights=False)

            # Classification head
            if self.fc_norm is not None:
                # Mean pooling
                x = self.fc_norm(x[:, 1:, :].mean(dim=1))
            else:
                # CLS token
                x = self.norm(x)
                x = x[:, 0]

            logits = self.head(x)

            if return_combine_weights:
                return logits, all_moe_weights
            return logits  # type: ignore

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        x = self.patch_embed(img)
        batch_size, num_patches, embed_dim = x.size()

        if mask is None:
            # mask_generator is passed through for shuffle indices but doesn't generate masks here
            # MEDiCModel is responsible for mask generation before calling this forward method
            raise ValueError("Mask must be provided to VisionTransformerMIM.forward(). "
                           "For automatic mask generation, use MEDiCModel.")

        if self.use_mask_tokens:
            # BEiT-style: use mask tokens
            # Apply mask tokens FIRST
            mask_token = self.mask_token.expand(batch_size, num_patches, -1)
            w = mask.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

            # Add position embeddings AFTER masking so mask tokens get positions too
            # This ensures consistency between pretraining and evaluation
            if self.pos_embed is not None:
                x = x + self.pos_embed[:, 1:, :]

            # Check if we should shuffle patches (for better reconstruction)
            if hasattr(mask_generator, 'shuffle_patches') and mask_generator.shuffle_patches:
                if hasattr(mask_generator, 'ids_shuffle') and mask_generator.ids_shuffle is not None:
                    # Apply shuffling to all patches
                    # Handle batched shuffling - ids_shuffle might be [L] or [B, L]
                    if mask_generator.ids_shuffle.dim() == 1:
                        # Single shuffle for all batch elements
                        x = x[:, mask_generator.ids_shuffle, :]
                    else:
                        # Per-batch shuffling - OPTIMIZED with gather (52x speedup)
                        # Expand indices for gathering x (which has embed_dim)
                        ids_expanded = mask_generator.ids_shuffle.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        x = torch.gather(x, dim=1, index=ids_expanded)
                    # Store restore indices for decoder
                    self.ids_restore = mask_generator.ids_restore
                else:
                    self.ids_restore = None
            else:
                self.ids_restore = None

            # Dense mode: all patches present (with mask tokens), no visible positions needed
            self.visible_positions_shuffled = None

            # Add CLS token with position
            cls_token = self.cls_token
            if self.pos_embed is not None:
                cls_token = cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        else:
            # MAE-style: drop masked tokens
            # Add position embeddings FIRST (critical for position information)
            if self.pos_embed is not None:
                x = x + self.pos_embed[:, 1:, :]

            # Check if we should shuffle BEFORE removing masked patches
            if hasattr(mask_generator, 'shuffle_patches') and mask_generator.shuffle_patches:
                if hasattr(mask_generator, 'ids_shuffle') and mask_generator.ids_shuffle is not None:
                    # Shuffle all patches first, then extract visible
                    if mask_generator.ids_shuffle.dim() == 1:
                        # Single shuffle for all batch elements
                        x = x[:, mask_generator.ids_shuffle, :]
                        mask = mask[:, mask_generator.ids_shuffle]
                    else:
                        # Per-batch shuffling - OPTIMIZED with gather (52x speedup)
                        # Expand indices for gathering x (which has embed_dim)
                        ids_expanded = mask_generator.ids_shuffle.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        x = torch.gather(x, dim=1, index=ids_expanded)

                        # For mask (2D tensor), we don't need the last dimension expansion
                        mask = torch.gather(mask.float(), dim=1, index=mask_generator.ids_shuffle).bool()
                    # Store both shuffle and restore indices
                    self.ids_restore = mask_generator.ids_restore
                    self.ids_shuffle = mask_generator.ids_shuffle
                else:
                    self.ids_restore = None
                    self.ids_shuffle = None
            else:
                # No shuffling - no ids_restore needed (weights already in spatial order)
                # Dense mode does the same at line 680
                self.ids_restore = None
                self.ids_shuffle = None
                # No shuffling: no visible positions tracking needed
                self.visible_positions_shuffled = None

            # Keep only visible patches using efficient batch processing
            # mask is True for masked, False for visible
            visible_mask = ~mask  # [B, N]
            batch_size_actual = x.shape[0]

            # Handle empty batch edge case
            if batch_size_actual == 0:
                # Return empty tensor with correct shape [0, 1, embed_dim]
                embed_dim = x.shape[-1] if x.shape[-1] > 0 else self.embed_dim
                x = torch.zeros(0, 1, embed_dim, device=x.device, dtype=x.dtype)
                x = self.norm(x)
                if not return_all_tokens:
                    return x[:, 0] if x.shape[1] > 0 else x.squeeze(1), None
                return x, mask

            # Count visible patches per sample
            num_visible_per_sample = visible_mask.sum(dim=1)  # [B]

            # Check if all samples have the same number of visible patches
            if torch.all(num_visible_per_sample == num_visible_per_sample[0]):
                # FAST PATH: All samples have same number of visible patches
                # This is the common case with block masking at fixed ratio
                num_visible = num_visible_per_sample[0].item()

                if num_visible > 0:
                    # Convert boolean mask to gather indices efficiently
                    # Get all visible positions as (batch_idx, patch_idx) pairs
                    visible_indices = visible_mask.nonzero(as_tuple=False)  # [total_visible, 2]

                    # Extract patch indices and reshape to [B, num_visible]
                    # This works because each batch has exactly num_visible patches
                    patch_indices = visible_indices[:, 1].reshape(batch_size_actual, num_visible)

                    # Store visible positions for decoder (needed for correct unshuffling with block masking)
                    # These are positions in the SHUFFLED array where visible patches are located
                    self.visible_positions_shuffled = patch_indices

                    # Use gather for efficient batch-parallel extraction
                    x = torch.gather(x, dim=1,
                                   index=patch_indices.unsqueeze(-1).expand(-1, -1, embed_dim))
                else:
                    # Edge case: all patches are masked (shouldn't happen normally)
                    x = torch.zeros(batch_size_actual, 1, embed_dim, device=x.device, dtype=x.dtype)
                    self.visible_positions_shuffled = None

            else:
                # This should NEVER happen with properly implemented masking generators
                # All samples in a batch should have the same number of visible patches
                # If this occurs, it indicates a bug in the masking generator
                raise RuntimeError(
                    f"Unexpected variable number of visible patches per sample: {num_visible_per_sample.tolist()}. "
                    f"All samples in a batch must have the same number of visible patches. "
                    f"This indicates a bug in the masking generator or incorrect mask creation. "
                    f"If you need variable masking, please implement it explicitly with proper padding."
                )

            # Add CLS token with position
            cls_token = self.cls_token
            if self.pos_embed is not None:
                cls_token = cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(batch_size_actual, -1, -1)  # FIX: Use batch_size_actual, not batch_size
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = self.pos_drop(x)

        # Forward through blocks, capturing combine weights from MoE blocks if requested
        combine_weights = None  # Last block's weights (for backward compatibility)
        combine_weights_dict = {} if (return_combine_weights and self.use_soft_moe) else None

        for i, blk in enumerate(self.blocks):
            is_moe_block = self.use_soft_moe and i in self.moe_blocks

            # Request combine weights from ALL MoE blocks if visualization requested
            if return_combine_weights and is_moe_block:
                x, block_combine_weights = blk(x, shared_rel_pos_bias=rel_pos_bias, return_combine_weights=True)
                combine_weights_dict[i] = block_combine_weights
                # Also keep last block's weights for backward compatibility
                if i == self.last_moe_block_idx:
                    combine_weights = block_combine_weights
            else:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias, return_combine_weights=False)

        x = self.norm(x)

        if not return_all_tokens:
            return x[:, 0], None, combine_weights, combine_weights_dict  # Return only the CLS token

        # Return ids_restore for sparse mode (needed by decoder), mask for dense mode
        # Also return combine_weights (last block) and combine_weights_dict (all blocks)
        if self.use_mask_tokens:
            return x, mask, combine_weights, combine_weights_dict
        else:
            return x, self.ids_restore, combine_weights, combine_weights_dict


    def get_features_for_evolved_masking(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention features for evolved masking (EPM integration).

        This method processes patch embeddings through the transformer blocks
        and accumulates attention maps for semantic part discovery.

        Args:
            x: Patch embeddings [B, N, D] (without CLS token)

        Returns:
            Aggregated attention maps [B, N, N] summed across heads and layers
        """
        with torch.no_grad():
            batch_size, num_patches, embed_dim = x.shape

            # Add position embeddings
            if self.pos_embed is not None:
                x = x + self.pos_embed[:, 1:, :]

            # Add CLS token
            cls_token = self.cls_token
            if self.pos_embed is not None:
                cls_token = cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # Apply dropout
            x = self.pos_drop(x)

            # Get shared relative position bias
            rel_pos_bias = None
            if self.rel_pos_bias is not None:
                rel_pos_bias = self.rel_pos_bias()

            # Accumulate attention across all blocks
            attn_maps = None
            for blk in self.blocks:
                x, attn_abs, attn_softmax = blk.get_attn_graph(x, shared_rel_pos_bias=rel_pos_bias)

                if attn_maps is None:
                    attn_maps = attn_softmax
                else:
                    attn_maps = attn_maps + attn_softmax

            # Remove CLS token interactions: [B, H, N+1, N+1] -> [B, H, N, N]
            attn_maps = attn_maps[:, :, 1:, 1:]

            # Sum across attention heads: [B, H, N, N] -> [B, N, N]
            attn_maps = attn_maps.sum(dim=1)

            return attn_maps
