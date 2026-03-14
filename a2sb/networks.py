# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


# Implemented based on Guided Diffusion https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
#     Licensed under the MIT license.


import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Union, List
from .utils import SequenceLength
from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
    )
from abc import abstractmethod

try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

_SAGE_WARNING_PRINTED = {} # Track warnings by reason
_SAGE_GLOBALLY_DISABLED = False # Permanent fallback if any kernel fails


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.group_norm(x.float(), self.num_groups, weight, bias, self.eps).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class EmbeddingConditionalBlock(nn.Module):
    """
    Any module where forward() takes an arbitrary embedding as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class EmbeddingConditionalSequential(nn.Sequential, EmbeddingConditionalBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbeddingConditionalBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(EmbeddingConditionalBlock):
    def __init__(self, in_channels: int, out_channels: int, emb_in_channels: int, use_scale_shift_norm=True, use_skip=True, p_dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_skip = use_skip
        if self.use_skip:
            assert(in_channels == out_channels)

        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
            )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            torch.nn.Conv2d(emb_in_channels, self.out_channels * 2 if use_scale_shift_norm else self.out_channels, kernel_size=1)
            )
        self.out_norm = GroupNorm32(32, out_channels)
        self.out_rest = nn.Sequential(nn.SiLU(),
                                      nn.Dropout(p=p_dropout),
                                      zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)))


    def forward(self, x, emb):
        """
        x: torch.tensor B x in_channels x H x W
        emb: B x emb_in_channels x 1 x 1 (or B x emb_in_channels x H x W)
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            # Broadcasting works automatically if emb_out is [B, 2C, 1, 1]
            h = self.out_norm(h) * (1 + scale) + shift
            h = self.out_rest(h)
        else:
            h = h + emb_out
            h = self.out_rest(self.out_norm(h))

        if self.use_skip:
            return x + h
        else:
            return h


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        if x.shape[1] != self.channels:
             # Handle cases where concat/skip might have changed channel count unexpectedly
             pass 
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class AttnUNetF(nn.Module):
    def __init__(self, n_updown_levels: int, in_channels: int, hidden_channels: Union[int, List[int]], out_channels: int, emb_channels: int,
                 rotary_dims=16, band_embedding_dim=0, attention_levels=None, n_attn_heads=4, num_res_blocks=2, use_attn_input_norm=True, attention_type="sdpa"):
        """
        Final architecture with sane parameterization
        inputs:
            attention_levels: 0-indexed levels specifying which ones should have attention layers
            n_updown_levels: total number of levels
            use_attn_input_norm: whether to use gradnorm as input to attention layers. Defaults to false, but will default to True in the future
            num_res_blocks: number of computational blocks per level
        """
        super().__init__()
        self.band_embedding_dim = band_embedding_dim
        assert(band_embedding_dim%2 == 0)
        self.enc_blocks = nn.ModuleList()
        self.ds_layers = nn.ModuleList()
        self.us_layers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.attention_levels = torch.tensor(attention_levels)
        decoder_attention_levels = n_updown_levels - 1 - self.attention_levels # index for decoder half

        self.n_updown_levels = n_updown_levels
        if type(hidden_channels) is int:
            self.hidden_channels_levels = [hidden_channels] * (n_updown_levels + 1)

        else:
            self.hidden_channels_levels = hidden_channels
        hidden_channels_levels = self.hidden_channels_levels
        self.input_projection = nn.Conv2d(in_channels, hidden_channels_levels[0], 3, padding=1)
        self.emb_channels = emb_channels + band_embedding_dim

        for level in range(n_updown_levels):
            # construct encoder
            # level 0 is the input layer
            ds_in_channels = hidden_channels_levels[level]
            ds_out_channels = hidden_channels_levels[level+1]
            layers = []
            for _ in range(num_res_blocks):
                layers.append(ResBlock(ds_in_channels, ds_in_channels, self.emb_channels))
                if level in attention_levels:  # downsampling level
                    layers.append(RotaryAttentionPool2d(embed_dim=ds_in_channels,
                                                        rotary_dim=32,
                                                        attn_dim=ds_in_channels,
                                                        num_heads=n_attn_heads,
                                                        output_dim=ds_in_channels,
                                                        use_input_norm=use_attn_input_norm,
                                                        attention_type=attention_type
                                                        ))
            self.enc_blocks.append(EmbeddingConditionalSequential(*layers))
            self.ds_layers.append(Downsample(ds_in_channels, True, out_channels=ds_out_channels))

            # construct decoder, level 0 is the first level after the middle block here
            us_in_channels = hidden_channels_levels[n_updown_levels - level]
            us_out_channels = hidden_channels_levels[n_updown_levels - level - 1]
            self.us_layers.append(Upsample(us_in_channels, True, out_channels=us_out_channels))
            layers = []
            for _ in range(num_res_blocks):
                layers.append(ResBlock(us_in_channels, us_in_channels, self.emb_channels))
                if level in decoder_attention_levels:
                    layers.append(RotaryAttentionPool2d(embed_dim=us_in_channels,
                                                        rotary_dim=32,
                                                        attn_dim=us_in_channels,
                                                        num_heads=n_attn_heads,
                                                        output_dim=us_in_channels,
                                                        use_input_norm=use_attn_input_norm,
                                                        attention_type=attention_type
                                                        ))
            self.dec_blocks.append(EmbeddingConditionalSequential(*layers))
        # construct middle block
        self.middle_block = EmbeddingConditionalSequential(ResBlock(hidden_channels_levels[-1],
                                                                     hidden_channels_levels[-1],
                                                                     self.emb_channels),
                                                           RotaryAttentionPool2d(embed_dim=hidden_channels_levels[-1],
                                                                                 rotary_dim=32,
                                                                                 attn_dim=hidden_channels_levels[-1],
                                                                                 num_heads=n_attn_heads,
                                                                                 output_dim=hidden_channels_levels[-1],
                                                                                 use_input_norm=use_attn_input_norm,
                                                                                 attention_type=attention_type
                                                                                 ),
                                                           ResBlock(hidden_channels_levels[-1],
                                                                    hidden_channels_levels[-1],
                                                                    self.emb_channels))

        self.output_projection = nn.Sequential(
            GroupNorm32(32, hidden_channels_levels[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_channels_levels[0], out_channels, 3, padding=1)
            )

    def get_band_embeddings(self, n_bands, device, dtype=torch.float32):
        if not hasattr(self, '_band_emb_cache'):
            self._band_emb_cache = {}
            
        cache_key = (n_bands, str(device), str(dtype))
        if cache_key in self._band_emb_cache:
            return self._band_emb_cache[cache_key]
            
        n_freqs = self.band_embedding_dim // 2
        coords = torch.arange(0, n_bands).to(device).to(dtype)
        coords_exp = coords.unsqueeze(0).repeat(n_freqs, 1)
        freqs = torch.arange(0, n_freqs).unsqueeze(-1) + 1
        freqs = freqs.to(device).to(dtype)
        coords_exp = freqs * (coords_exp) * 2 * 3.14159 / (3*n_bands)
        cos_embs = torch.cos(coords_exp)
        sin_embs = torch.sin(coords_exp)
        band_embs = torch.cat((cos_embs, sin_embs), 0)
        band_embs_cat = band_embs.unsqueeze(0).unsqueeze(-1)
        
        self._band_emb_cache[cache_key] = band_embs_cat
        return band_embs_cat

    def forward(self, x, emb):
        """
        x: torch.tensor B x C x H x W
        emb: torch.tensor B x D_emb
        """

        hs = []

        h = self.input_projection(x)
        emb = emb.to(h.dtype).view(h.shape[0], -1, 1, 1) # [B, D, 1, 1]
        
        # Band embeddings are constant per frequency bin, we can use them via broadcasting
        # or concatenation if we must. Since ResBlock expects one 'emb' tensor, 
        # we concatenate the band_emb to the time_emb once at the highest dimension.
        
        # Optimization: Don't repeat spatially. Let torch broadcasting/pooling handle it.
        # However, band_emb is [1, D_band, H, 1]. Time_emb is [B, D_time, 1, 1].
        # We need a unified [B, D_total, H, 1] that broadcast across W.
        
        for level in range(self.n_updown_levels):
            # Time embedding doesn't need spatial repeat, but band_emb does (vertical only)
            if self.band_embedding_dim > 0:
                band_emb = self.get_band_embeddings(h.shape[2], h.device, dtype=h.dtype) # [1, D_band, H, 1]
                curr_emb = torch.cat([band_emb.expand(h.shape[0], -1, -1, -1), emb.expand(-1, -1, h.shape[2], -1)], dim=1)
            else:
                curr_emb = emb
                
            h = self.enc_blocks[level](h, curr_emb)
            h = self.ds_layers[level](h)
            hs.append(h)
            
        # middle block
        if self.band_embedding_dim > 0:
            band_emb = self.get_band_embeddings(h.shape[2], h.device, dtype=h.dtype)
            curr_emb = torch.cat([band_emb.expand(h.shape[0], -1, -1, -1), emb.expand(-1, -1, h.shape[2], -1)], dim=1)
        else:
            curr_emb = emb
        h = self.middle_block(h, curr_emb)

        for level in range(self.n_updown_levels):
            h = h + hs.pop()
            # Corresponding embedding for this level (spatially adjusted)
            if self.band_embedding_dim > 0:
                band_emb = self.get_band_embeddings(h.shape[2], h.device, dtype=h.dtype)
                curr_emb = torch.cat([band_emb.expand(h.shape[0], -1, -1, -1), emb.expand(-1, -1, h.shape[2], -1)], dim=1)
            else:
                curr_emb = emb
            h = self.dec_blocks[level](h, curr_emb)
            h = self.us_layers[level](h)

        h = self.output_projection(h)
        return h


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / torch.sqrt(torch.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
        rotary_dims = 16
    ):
        super().__init__()

        #self.positional_embedding = nn.Parameter(
        #    torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        #)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        #x = x # + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class RotaryAttentionPool2d(nn.Module):
    def __init__(
            self,
            rotary_dim=32,
            attn_dim: int = None,
            embed_dim: int = None,
            num_heads: int = None,
            output_dim: int = None,
            use_input_norm: bool = False,
            attention_type: str = "sdpa"
    ):
        super().__init__()
        self.attn_dim = attn_dim
        self.output_dim = output_dim
        self.use_input_norm = use_input_norm
        self.num_heads = num_heads
        
        # Pre-calculate best attention engine
        self.actual_attention_type = "sdpa"
        if attention_type == "sage" and SAGE_ATTENTION_AVAILABLE:
            head_dim = attn_dim // num_heads
            # SageAttention 2 supports head_dim 64, 96, and 128
            if head_dim not in [64, 96, 128]:
                reason = f"Unsupported head_dim: {head_dim} (Sage 2 supports 64, 96, 128)"
                if reason not in _SAGE_WARNING_PRINTED:
                    print(f"[A2SB] SageAttention skipped for layers with head_dim {head_dim}. Using SDPA.")
                    _SAGE_WARNING_PRINTED[reason] = True
            else:
                self.actual_attention_type = "sage"

        if use_input_norm:
            self.gnorm = GroupNorm32(32, embed_dim)
        self.q_proj = nn.Conv2d(embed_dim, attn_dim, 1)
        self.k_proj= nn.Conv2d(embed_dim, attn_dim, 1)
        self.v_proj = nn.Conv2d(embed_dim, output_dim, 1)
        self.num_heads = num_heads
        self.pos_emb = RotaryEmbedding(
            dim=rotary_dim,
            freqs_for='pixel',
            max_freq=64)

    def forward(self, x):
        """
        x: tensor of shape b x c x h x w
        """
        if self.use_input_norm:
            x = self.gnorm(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # split heads:
        _b, _dims, _height, _width = q.shape
        attn_head_dim = self.attn_dim // self.num_heads
        out_head_dim = self.output_dim // self.num_heads
        q = q.view(_b, self.num_heads, attn_head_dim, _height, _width).permute(0, 1, 3, 4, 2).contiguous()
        k = k.view(_b, self.num_heads, attn_head_dim, _height, _width).permute(0, 1, 3, 4, 2).contiguous()
        v = v.view(_b, self.num_heads, out_head_dim, _height, _width).permute(0, 1, 3, 4, 2).contiguous()

        # apply rotary (freqs matches H and W dimensions)
        freqs = self.pos_emb.get_axial_freqs(_height, _width)
        
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # Flatten for high-performance attention kernels: [B, Heads, HW, Dim]
        # Transpose to NHD layout [B, Seq, Heads, Dim]
        q = q.reshape(_b, self.num_heads, _height * _width, attn_head_dim).transpose(1, 2).contiguous()
        k = k.reshape(_b, self.num_heads, _height * _width, attn_head_dim).transpose(1, 2).contiguous()
        v = v.reshape(_b, self.num_heads, _height * _width, out_head_dim).transpose(1, 2).contiguous()

        global _SAGE_GLOBALLY_DISABLED
        if self.actual_attention_type == "sage" and not _SAGE_GLOBALLY_DISABLED:
            try:
                # tensor_layout="NHD" is required — tensors are [B, Seq, Heads, Dim].
                # Without it SA2 assumes HND layout and dispatches the wrong kernel,
                # causing cudaErrorNoKernelImageForDevice on Blackwell (SM_120) GPUs.
                attn_out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
            except Exception as e:
                error_str = str(e)
                if "no kernel image" in error_str or "cudaErrorNoKernelImageForDevice" in error_str:
                    # Per-layer sticky fallback — other head_dim layers may still work.
                    reason = f"No SA2 kernel for head_dim={q.shape[-1]} on this GPU arch"
                    if reason not in _SAGE_WARNING_PRINTED:
                        print(f"[A2SB] SageAttention kernel unavailable for head_dim={q.shape[-1]}. "
                              f"Falling back to SDPA for this layer only.")
                        _SAGE_WARNING_PRINTED[reason] = True
                    self.actual_attention_type = "sdpa"
                else:
                    _SAGE_GLOBALLY_DISABLED = True
                    print(f"[A2SB] SageAttention unexpected failure: {e}. Switching all layers to SDPA.")
                attn_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        else:
            attn_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

        # attn_out: [B, Seq, Heads, out_head_dim]
        # Restore spatial layout: [B, output_dim, H, W]
        attn_out = attn_out.transpose(1, 2) # [B, Heads, Seq, out_head_dim]
        attn_out = attn_out.reshape(_b, self.num_heads, _height, _width, out_head_dim)
        attn_out = attn_out.permute(0, 1, 4, 2, 3).reshape(_b, self.output_dim, _height, _width)
        return attn_out


class SinusoidalTemporalEmbedding(nn.Module):
    def __init__(self, n_bands, min_freq=1, max_freq=16):
        super().__init__()
        self.n_bands = n_bands
        multipliers = torch.linspace(min_freq, max_freq, n_bands).unsqueeze(0)
        self.register_buffer('multipliers', multipliers)

    def forward(self, t):
        """
        input:
            t: torch.tensor of dims B (batch) [0,1]
        output:
            t_emb: torch.tensor of dims B x 2*n_bands
        """
        sin_vals = torch.sin(t[:, None] @ self.multipliers)
        cos_vals = torch.cos(t[:, None] @ self.multipliers)
        return torch.cat((sin_vals, cos_vals), -1)
