# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# mae: https://github.com/facebookresearch/mae
# slowfast: https://github.com/facebookresearch/SlowFast
# --------------------------------------------------------


from functools import partial
from typing import Tuple, Optional, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hiera import Hiera, HieraBlock
from .hiera_utils import pretrained_model, undo_windowing, conv_nd


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Identity):
        return x

    B, num_mask_units = x.shape[0:2]
    # Apply head, e.g [B, #MUs, My, Mx, C] -> head([B * #MUs, C, My, Mx])
    permute = [0] + [len(x.shape) - 2] + list(range(1, len(x.shape) - 2))
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(permute))

    # Restore original layout, e.g. [B * #MUs, C', My', Mx'] -> [B, #MUs, My', Mx', C']
    permute = [0] + list(range(2, len(x.shape))) + [1]
    x = x.permute(permute).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])
    return x


class MaskedAutoencoderHiera(Hiera):
    """Masked Autoencoder with Hiera backbone"""

    def __init__(
        self,
        in_chans: int = 3,
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        **kwdargs,
    ):
        super().__init__(
            in_chans=in_chans,
            patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            **kwdargs,
        )

        del self.norm, self.head
        encoder_dim_out = self.blocks[-1].dim_out
        self.encoder_norm = norm_layer(encoder_dim_out)
        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]
        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for i in self.stage_ends[: self.q_pool]:  # resolution constant after q_pool
            kernel = [
                i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)
            ]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(self.q_stride))(
                    self.blocks[i].dim_out,
                    encoder_dim_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim
            )
        )

        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (
            self.q_stride[-1] ** self.q_pool
        )  # patch stride of prediction

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * in_chans,
        )  # predictor
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pixel_label_2d(
        self, input_img: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*
        input_img = input_img.permute(0, 2, 3, 1)

        size = self.pred_stride
        label = input_img.unfold(1, size, size).unfold(2, size, size)
        label = label.flatten(1, 2).flatten(2)
        label = label[mask]
        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def get_pixel_label_3d(
        self, input_vid: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*

        # We use time strided loss, only take the first frame from each token
        input_vid = input_vid[:, :, ::self.patch_stride[0], :, :]

        size = self.pred_stride
        label = input_vid.unfold(3, size, size).unfold(4, size, size)
        label = label.permute(0, 2, 3, 4, 5, 6, 1)  # Different from 2d, mistake during training lol
        label = label.flatten(1, 3).flatten(2)
        label = label[mask]

        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label


    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if mask is None:
            mask = self.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super().forward(x, mask, return_intermediates=True)
        # Resolution unchanged after q_pool stages, so skip those features
        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        # Multi-scale fusion
        x = 0.0
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            x += apply_fusion_head(head, interm_x)

        x = self.encoder_norm(x)

        return x, mask

    def forward_decoder(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.decoder_embed(x)

        # Combine visible and mask tokens

        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, #MUs_all]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return x, mask

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """
        if len(self.q_stride) == 2:
            label = self.get_pixel_label_2d(x, mask)
        elif len(self.q_stride) == 3:
            label = self.get_pixel_label_3d(x, mask)
        else:
            raise NotImplementedError

        pred = pred[mask]
        loss = (pred - label) ** 2

        return loss.mean(), pred, label

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.6,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        latent, mask = self.forward_encoder(x, mask_ratio, mask=mask)
        pred, pred_mask = self.forward_decoder(
            latent, mask
        )  # pred_mask is mask at resolution of *prediction*

        # Toggle mask, to generate labels for *masked* tokens
        return *self.forward_loss(x, pred, ~pred_mask), mask



class MaskedAutoencoderHieraUnet(Hiera):
    """Unet Masked Autoencoder with Hiera backbone"""

    def __init__(
        self,

        # general configs
        input_size: Tuple[int, ...] = (224, 224),
        in_chans: int = 3,
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,

        # encoder configs
        # input_size: Tuple[int, ...] = (224, 224),
        # encoder_in_chans: int = 3,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        # encoder_patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        # encoder_mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        encoder_norm_layer: Union[str, nn.Module] = "LayerNorm",
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_pos_embed: bool = False,

        # decoder configs
        decoder_embed_dim: int = 512,
        # decoder_depth: int = 8,
        decoder_stages: Tuple[int, ...] = (2, 2, 2, 2),
        decoder_num_heads: int = 16,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            input_size = input_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            num_heads = num_heads,
            num_classes = num_classes,
            stages = stages,
            q_pool = q_pool,
            q_stride = q_stride,
            mask_unit_size = mask_unit_size,
            mask_unit_attn = mask_unit_attn,
            dim_mul = dim_mul,
            head_mul = head_mul,
            patch_kernel = patch_kernel,
            patch_stride = patch_stride,
            patch_padding = patch_padding,
            mlp_ratio = mlp_ratio,
            drop_path_rate = drop_path_rate,
            norm_layer = encoder_norm_layer,
            head_dropout = head_dropout,
            head_init_scale = head_init_scale,
            sep_pos_embed = sep_pos_embed,
        )

        del self.norm, self.head
        encoder_dim_out = self.blocks[-1].dim_out
        self.encoder_norm = norm_layer(encoder_dim_out)
        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]
        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for i in self.stage_ends[: self.q_pool]:  # resolution constant after q_pool
            kernel = [
                i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)
            ]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(self.q_stride))(
                    self.blocks[i].dim_out,
                    encoder_dim_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim
            )
        )

        decoder_depth = sum(decoder_stages)
        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (
            self.q_stride[-1] ** self.q_pool
        )  # patch stride of prediction

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * in_chans,
        )  # predictor
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pixel_label_2d(
        self, input_img: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*
        input_img = input_img.permute(0, 2, 3, 1)

        size = self.pred_stride
        label = input_img.unfold(1, size, size).unfold(2, size, size)
        label = label.flatten(1, 2).flatten(2)
        label = label[mask]
        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def get_pixel_label_3d(
        self, input_vid: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*

        # We use time strided loss, only take the first frame from each token
        input_vid = input_vid[:, :, ::self.patch_stride[0], :, :]

        size = self.pred_stride
        label = input_vid.unfold(3, size, size).unfold(4, size, size)
        label = label.permute(0, 2, 3, 4, 5, 6, 1)  # Different from 2d, mistake during training lol
        label = label.flatten(1, 3).flatten(2)
        label = label[mask]

        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label


    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if mask is None:
            mask = self.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super().forward(x, mask, return_intermediates=True)
        # Resolution unchanged after q_pool stages, so skip those features
        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        # Multi-scale fusion
        x = 0.0
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            x += apply_fusion_head(head, interm_x)

        x = self.encoder_norm(x)

        return x, mask

    def forward_decoder(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.decoder_embed(x)

        # Combine visible and mask tokens

        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, #MUs_all]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return x, mask

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """
        if len(self.q_stride) == 2:
            label = self.get_pixel_label_2d(x, mask)
        elif len(self.q_stride) == 3:
            label = self.get_pixel_label_3d(x, mask)
        else:
            raise NotImplementedError

        pred = pred[mask]
        loss = (pred - label) ** 2

        return loss.mean(), pred, label

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.6,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        ############################## forward_encoder
        if mask is None:
            mask = self.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super().forward(x, mask, return_intermediates=True)
        # Resolution unchanged after q_pool stages, so skip those features
        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        # Multi-scale fusion
        encoder_outputs = intermediates
        x = 0.0
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            x += apply_fusion_head(head, interm_x)

        x = self.encoder_norm(x)

        latent, mask = x, mask
        
        ############################## forward_decoder
        x, mask = latent, mask
        # Embed tokens
        x = self.decoder_embed(x)

        # Combine visible and mask tokens

        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, #MUs_all]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Reverse encoder outputs for skip connections
        print([i.shape for i in encoder_outputs])
        skip_connections = encoder_outputs[:-1][::-1]
        skip_shapes = []  # To store spatial shapes of skip connections
        # for i, skip in enumerate(skip_connections):
        #     skip = self.decoder_embed(skip)
        #     skip = undo_windowing(
        #         skip,
        #         [s // (2 ** (i + 1)) for s in self.tokens_spatial_shape_final],
        #         [mu // (2 ** (i + 1)) for mu in self.mask_unit_spatial_shape_final],
        #     )
        #     skip_connections[i] = skip
        #     skip_shapes.append(skip.shape[2:])

        # Apply decoder blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)

        # Apply decoder with skip connections
        for i, blk in enumerate(self.decoder_blocks):
            print(i,blk)
            print(x.shape)
            if i%2==0:
                print(skip_connections[i//2].shape)
            # Upsample x if necessary
            # if i < len(skip_connections):
            #     x = F.interpolate(x, size=skip_connections[i].shape[2:], mode='nearest')
            #     x = x + skip_connections[i]  # Skip connection
            # Flatten x for transformer block
            # B, C, *spatial_dims = x.shape
            # x = x.view(B, -1, C)
            x = blk(x)
            # Reshape back to spatial dimensions
            # x = x.view(B, C, *spatial_dims)


        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        pred, pred_mask =  x, mask
        # pred_mask is mask at resolution of *prediction*

        # Toggle mask, to generate labels for *masked* tokens
        return *self.forward_loss(x, pred, ~pred_mask), mask


# Image Models

@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
}, default="mae_in1k")
def mae_hiera_tiny_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), q_pool=2, **kwargs,
    )


@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
}, default="mae_in1k")
def mae_hiera_small_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), q_pool=2, **kwargs,
    )


@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
}, default="mae_in1k")
def mae_hiera_base_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), q_pool=2, **kwargs,
    )


@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
}, default="mae_in1k")
def mae_hiera_base_plus_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), q_pool=2, **kwargs,
    )


@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
}, default="mae_in1k")
def mae_hiera_large_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), q_pool=2, **kwargs,
    )


@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
}, default="mae_in1k")
def mae_hiera_huge_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), q_pool=2, **kwargs,
    )



# Video Models

@pretrained_model({
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
}, default="mae_k400")
def mae_hiera_base_16x224(num_classes: int = 400, **kwdargs):
    return MaskedAutoencoderHiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        q_pool=2,
        **kwdargs
    )


@pretrained_model({
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth",
}, default="mae_k400")
@pretrained_model(None)
def mae_hiera_base_plus_16x224(**kwdargs):
    return mae_hiera_base_16x224(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs
    )


@pretrained_model({
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth",
}, default="mae_k400")
@pretrained_model(None)
def mae_hiera_large_16x224(**kwdargs):
    return mae_hiera_base_16x224(
        embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs
    )


@pretrained_model({
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth",
}, default="mae_k400")
def mae_hiera_huge_16x224(**kwdargs):
    return mae_hiera_base_16x224(
        embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs
    )
