# ---------------------------------------------------------------
# Adapted from https://github.com/NVlabs/I2SB/
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from typing import List, Optional
from einops import rearrange
import inspect

class ComplexSpectrogram:
    def __init__(self, n_fft=2048, win_length=2048, hop_length=512, eps=1e-9):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.eps = eps

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # stft/view_as_real often need float32
        orig_dtype = waveform.dtype
        spec = torch.stft(waveform.float(), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=torch.hann_window(self.win_length).to(waveform.device), return_complex=True)
        spec = torch.view_as_real(spec).permute(2, 0, 1).to(orig_dtype)
        return spec

class ComplexToMagInstPhase:
    def __call__(self, complex_spec: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(complex_spec[0:1] ** 2 + complex_spec[1:2] ** 2)
        phase = torch.atan2(complex_spec[1:2], complex_spec[0:1])
        return torch.cat([mag, torch.cos(phase), torch.sin(phase)], 0)

class MagInstPhaseToComplex:
    def __call__(self, msp_spec: torch.Tensor) -> torch.Tensor:
        mag = msp_spec[:1]
        cos_theta = msp_spec[1:2]
        sin_theta = msp_spec[2:3]
        return torch.cat([mag * cos_theta, mag * sin_theta], 0)

class SVDFixMagInstPhase:
    def __call__(self, msp_spec: torch.Tensor) -> torch.Tensor:
        mag = msp_spec[:1]
        cos_theta = msp_spec[1:2]
        sin_theta = msp_spec[2:3]
        top = torch.cat([cos_theta, -sin_theta], 0)
        bottom = torch.cat([sin_theta, cos_theta], 0)
        rot = torch.stack([top, bottom], 0) 
        rot = rearrange(rot, "r c n t -> n t r c")
        
        # SVD not implemented for BF16 on CUDA
        orig_dtype = rot.dtype
        U, S, Vh = torch.linalg.svd(rot.float())
        
        new_S = S.clone()
        new_S[..., 0] = 1
        new_S[..., 1] = torch.det(U @ Vh)
        new_rot = U @ torch.diag_embed(new_S) @ Vh
        new_cos_sin_theta = new_rot[..., :, 0]
        new_cos_sin_theta = rearrange(new_cos_sin_theta, "n t r -> r n t").to(orig_dtype)
        new_msp_spec = torch.cat([mag, new_cos_sin_theta], 0)
        return new_msp_spec

class InverseComplexSpectrogram:
    def __init__(self, n_fft=2048, win_length=2048, hop_length=512, eps=1e-9):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.eps = eps

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        # view_as_complex/istft do not support BF16
        spec_f = spec.float()
        spec_c = torch.view_as_complex(spec_f.permute(1, 2, 0).contiguous().to(memory_format=torch.contiguous_format))
        return torch.istft(spec_c, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                           window=torch.hann_window(self.win_length).to(spec.device))

class PowerScaleSpectrogram:
    def __init__(self, power=0.5, channels=None, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.power = power
        self.channels = channels

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        spec_abs = spec.abs()
        scale = spec_abs ** self.power / (spec_abs + self.eps)
        if self.channels is None:
            spec = spec * scale
        else:
            inds_to_scale = torch.tensor(self.channels).to(spec.device)
            spec = spec.clone()
            spec[inds_to_scale] = spec[inds_to_scale] * scale[inds_to_scale]
        return spec

class SpectrogramDropDCTerm:
    def __init__(self):
        super().__init__()

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        return spec[..., 1:, :]

class SpectrogramAddDCTerm:
    def __init__(self):
        super().__init__()

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        dc_channel = torch.zeros_like(spec[..., :1, :])
        return torch.cat((dc_channel, spec), -2)

def apply_audio_transforms(audio: torch.Tensor, transforms: List):
    for tx_fn in transforms:
        output = tx_fn(audio)
        if type(output) is tuple:
            audio = output[0]
        else:
            audio = output
    return audio
