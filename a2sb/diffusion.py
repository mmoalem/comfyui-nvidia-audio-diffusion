# ---------------------------------------------------------------
# Adapted from https://github.com/NVlabs/I2SB/blob/master/i2sb/diffusion.py
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from math import ceil
from einops import rearrange


def get_multidiffusion_vf(vf_model, x_t, t_emb, win_length=256, hop_length=128, batch_size=1):
    """
    Compute vector fields using multidiffusion windowing for long sequences.
    Vectorized version using unfold for much higher performance.
    """
    b_size, num_channels, win_height, seq_len = x_t.shape
    device = x_t.device
    
    # 1. Extract windows efficiently using unfold
    # x_t: [B, C, H, L] -> windows: [B, C, H, num_hops, win_length]
    windows = x_t.unfold(-1, win_length, hop_length)
    num_hops = windows.shape[3]
    
    # Rearrange to [num_hops * B, C, H, win_length]
    # We want each hop for all channels to be contiguous
    windows = windows.permute(3, 0, 1, 2, 4).reshape(-1, num_channels, win_height, win_length)
    
    # 2. Process windows in batches
    v_out_list = []
    total_samples = windows.shape[0]
    
    for i in range(0, total_samples, batch_size):
        curr_batch_x = windows[i : i + batch_size].contiguous().to(memory_format=torch.channels_last)
        # Repeat time embedding for each window in the batch
        # t_emb is [B, D]. We need it to match curr_batch_x[i].
        # In the bandwidth node, t_emb passed is [B, D].
        # In our reshaped windows, the batch dimension follows (hop0_B0, hop0_B1, hop1_B0, hop1_B1...)
        # So we repeat the same B-size block of embeddings.
        curr_batch_t = t_emb.repeat(ceil(curr_batch_x.shape[0] / b_size), 1)[:curr_batch_x.shape[0]]
        
        v_out_list.append(vf_model(curr_batch_x, curr_batch_t))
        
    v_out = torch.cat(v_out_list, dim=0) # [num_hops * B, C, H, win_length]
    
    # 3. Apply Hann window for smooth blending
    hann = torch.hann_window(win_length, periodic=True).to(device).to(v_out.dtype)
    hann = hann.view(1, 1, 1, win_length)
    v_out = v_out * hann
    
    # 4. Accumulate back into sequence
    # Reshape back to [num_hops, B, C, H, win_length]
    v_out = v_out.view(num_hops, b_size, num_channels, win_height, win_length)
    
    # Reconstruction via summation
    vf_t = torch.zeros_like(x_t)
    counts = torch.zeros_like(x_t)
    
    for hop_idx in range(num_hops):
        l_idx = hop_idx * hop_length
        r_idx = l_idx + win_length
        vf_t[..., l_idx:r_idx] += v_out[hop_idx]
        counts[..., l_idx:r_idx] += hann
            
    return vf_t / (counts + 1e-8)


def multidiffusion_pad_inputs(input, win_length, hop_length, padding_constant=None):
    _b, _c, _h, width = input.shape
    if width <= win_length:  # no hops
        to_pad = win_length - width
    else:
        pad_to = ceil((width - win_length) / hop_length) * hop_length + win_length
        to_pad = pad_to - width

    if to_pad > 0:
        padding = input[..., :to_pad]
        if padding_constant is not None:
            padding = padding * 0 + padding_constant
        input_padded = torch.cat([input, padding], dim=-1)
    else:
        input_padded = input.clone()
    return input_padded


def multidiffusion_unpad_outputs(output, original_width: int):
    return output[..., :original_width]


def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


class Diffusion(nn.Module):
    def __init__(self, beta_min=1e-4, beta_max=0.3):
        super().__init__()
        # t = 0 (clean data), t=1 (corrupted posterior)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta_t(self, t):
        if t <= 0.5:
            return t**2 * self.beta_max
        else:
            return (1 - t)**2 * self.beta_max

    def get_int_beta_0_t(self, t):
        """t: torch.tensor [0,1]"""
        beta_int = t.clone()
        full_integral = 2 * self.beta_max * (0.5**3) / 3
        half_inds = t > 0.5
        beta_int[half_inds] = full_integral - 1 / 3 * self.beta_max * ((1 - t[half_inds])**3)
        beta_int[~half_inds] = 1 / 3 * self.beta_max * (t[~half_inds]**3)
        return beta_int

    def get_std_fwd(self, t):
        return torch.sqrt(self.get_int_beta_0_t(t))

    def get_std_rev(self, t):
        return torch.sqrt(self.get_int_beta_0_t(1 - t))

    def get_std_t(self, t):
        sigma_fwd = self.get_std_fwd(t)
        sigma_rev = self.get_std_rev(t)
        coef1, coef2, var = compute_gaussian_product_coef(sigma_fwd, sigma_rev)
        return torch.sqrt(var)

    def q_sample(self, t, x_0, x_1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """
        sigma_fwd = self.get_std_fwd(t)
        sigma_rev = self.get_std_rev(t)

        coef1, coef2, var = compute_gaussian_product_coef(sigma_fwd, sigma_rev)
        while len(coef1.shape) < len(x_0.shape):
            coef1 = coef1[:, None]
            coef2 = coef2[:, None]
            var = var[:, None]
        x_t = coef1 * x_0 + coef2 * x_1
        std_sb_t = torch.sqrt(var)
        if not ot_ode:
            x_t += std_sb_t * torch.randn_like(x_t)
        return x_t.detach()

    def p_posterior(self, t_prev, t, x_t, x_0, ot_ode=False):
        assert t_prev < t
        std_t = self.get_std_fwd(t)
        std_t_prev = self.get_std_fwd(t_prev)
        std_delta = (std_t**2 - std_t_prev**2).sqrt()
        mu_x0, mu_xt, var = compute_gaussian_product_coef(std_t_prev, std_delta)
        x_t_prev = mu_x0 * x_0 + mu_xt * x_t

        if not ot_ode and t_prev > 0:
            x_t_prev = x_t_prev + var.sqrt() * torch.randn_like(x_t_prev)
        return x_t_prev

    def get_pred_x0(self, t, x_t, net_out):
        std_fwd_t = self.get_std_fwd(t)
        pred_x0 = x_t - std_fwd_t * net_out
        return pred_x0
