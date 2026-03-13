import torch
import numpy as np
import comfy
import comfy.model_management as mm
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2sb.audio_transforms.transforms import (
    ComplexSpectrogram, ComplexToMagInstPhase, SpectrogramDropDCTerm, PowerScaleSpectrogram,
    SpectrogramAddDCTerm, SVDFixMagInstPhase, MagInstPhaseToComplex, InverseComplexSpectrogram
)
from a2sb.corruption.corruptions import UpsampleMask, mask_with_noise
from a2sb.diffusion import multidiffusion_pad_inputs, multidiffusion_unpad_outputs, get_multidiffusion_vf
from a2sb.networks import SinusoidalTemporalEmbedding

class A2SB_BandwidthExtension:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a2sb_model": ("A2SB_MODEL",),
                "audio": ("AUDIO",),
                "steps": ("INT", {"default": 50, "min": 10, "max": 200}),
                "cutoff_freq": ("INT", {"default": 0, "min": 0, "max": 22050}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 64}),
                "unload_model": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "extend_bandwidth"
    CATEGORY = "audio/A2SB"

    def compute_rolloff_freq(self, audio_waveform, sr, roll_percent=0.99):
        # audio_waveform: [samples] -> expect Torch tensor on CPU/GPU
        device = audio_waveform.device
        y = audio_waveform
        if y.dim() > 1:
            y = y[0]
        
        if len(y) == 0:
            print("[A2SB] Warning: empty audio for rolloff detection")
            return 8000
            
        # Manual spectral rolloff in Torch
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft).to(device)
        
        # Compute STFT
        spec = torch.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = spec.abs() # [freq_bins, frames]
        power = mag**2
        
        # Aggregate power across bins
        total_energy = power.sum(dim=0)
        cum_energy = torch.cumsum(power, dim=0)
        
        # Find bin where cum_energy exceeds roll_percent of total
        threshold = roll_percent * total_energy
        rolloff_bins = (cum_energy >= threshold).long().argmax(dim=0)
        
        # Convert bins to frequency
        # max_freq = sr / 2
        # freq_per_bin = sr / n_fft
        rolloff_freqs = rolloff_bins * (sr / n_fft)
        
        rolloff_freq = int(torch.mean(rolloff_freqs.float()).item())
        print(f"[A2SB] Auto-detected 99% rolloff frequency: {rolloff_freq} Hz")
        return rolloff_freq

    def torch_resample(self, wav, orig_sr, target_sr):
        if orig_sr == target_sr:
            return wav
        
        #wav: [C, S] or [S]
        was_1d = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0) # [1, 1, S]
            was_1d = True
        else:
            wav = wav.unsqueeze(0) # [1, C, S]
            
        new_len = int(wav.shape[-1] * target_sr / orig_sr)
        # mode='linear' for 1D (treating it as 'bilinear' but on 1D it uses 'linear')
        # Wait, for 3D input (B, C, L), mode='linear' is correct.
        resampled = torch.nn.functional.interpolate(wav, size=new_len, mode='linear', align_corners=False)
        
        if was_1d:
            return resampled.squeeze(0).squeeze(0)
        else:
            return resampled.squeeze(0)

    def extend_bandwidth(self, a2sb_model, audio, steps, cutoff_freq, batch_size, unload_model):
        mm.throw_exception_if_processing_interrupted()
        device = mm.get_torch_device()
        
        # Unpack model bundle
        models = a2sb_model["models"]
        t_cutoffs = a2sb_model["t_cutoffs"]
        diffusion = a2sb_model["diffusion"]
        use_ot_ode = a2sb_model["use_ot_ode"]
        dtype = a2sb_model.get("dtype", torch.float32)
        
        n_timestep_channels = 128
        t_to_emb = SinusoidalTemporalEmbedding(n_bands=int(n_timestep_channels//2), min_freq=0.5).to(device).to(dtype)

        # 1. Audio formatting
        audio_waveform = audio["waveform"]
        sr = audio["sample_rate"]
        
        # Handle Batch dimension if present
        if audio_waveform.dim() == 3:
            audio_waveform = audio_waveform[0] # Take first batch [C, S]
            
        # A2SB requires 44100Hz
        if sr != 44100:
            print(f"[A2SB] Resampling input from {sr}Hz to 44100Hz...")
            audio_waveform = self.torch_resample(audio_waveform, sr, 44100)
            sr = 44100

        # A2SB is mono-trained, but we can batch channels
        if audio_waveform.ndim == 1:
            audio_waveform = audio_waveform.unsqueeze(0) # [1, S]
            
        n_channels = audio_waveform.shape[0]
        original_audio = audio_waveform.to(device) # shape [C, samples]

        # Determine cutoff frequency (using first channel for detection)
        if cutoff_freq == 0:
            cutoff_freq = self.compute_rolloff_freq(audio_waveform[0], sr)
            
        # 2. Forward Transforms (waveform -> STFT)
        print(f"[A2SB] Applying STFT transforms to {n_channels} channel(s)...")
        transform_gt = [
            ComplexSpectrogram(n_fft=2048, win_length=2048, hop_length=512, eps=1e-9),
            ComplexToMagInstPhase(),
            SpectrogramDropDCTerm(),
            PowerScaleSpectrogram(power=0.25, channels=[0], eps=1e-9)
        ]
        
        # Process channels
        stft_channels = []
        for c in range(n_channels):
            res = original_audio[c]
            for tx in transform_gt:
                res = tx(res)
            stft_channels.append(res)
            
        # Stack channels into batch dim: [C, 3, H, W]
        stft_target = torch.stack(stft_channels, dim=0)
        
        # 3. Create Up-sampling mask (corruption)
        print(f"[A2SB] Applying corruption mask at {cutoff_freq}Hz...")
        # Get mask for one channel and repeat
        mask_single = UpsampleMask.get_upsample_mask(
            stft_target[0], 
            cutoff_freq, cutoff_freq, sampling_rate=44100, dc_dropped=True
        ).to(device)
        mask = mask_single.unsqueeze(0).repeat(n_channels, 1, 1, 1)
        
        # Add noise to masked regions
        noise_level = 0.5
        stft_corrupted = mask_with_noise(stft_target, mask, noise_level)
        
        def get_vf_model(t_val):
            model_idx = 0
            for idx, thresh in enumerate(t_cutoffs):
                if t_val >= thresh:
                    model_idx = idx + 1
            return models[model_idx]
            
        # Load all needed models to GPU
        mm.load_models_gpu(models)

        # 4. Sampling Loop (Diffusion)
        print(f"[A2SB] Starting sampling over {steps} steps (Batch: {n_channels})...")
        t_steps = torch.linspace(1, 0.05, int(steps)).to(device).to(dtype)
        n_steps = len(t_steps) - 1
        
        win_length = 256
        hop_length = 128
        
        original_width = stft_corrupted.shape[-1]
        x_1 = multidiffusion_pad_inputs(stft_corrupted, win_length, hop_length).to(dtype)
        mask_padded = multidiffusion_pad_inputs(mask, win_length, hop_length).to(dtype)
        
        x_t = x_1.clone()
        pbar = comfy.utils.ProgressBar(n_steps)
        
        try:
            with torch.no_grad():
                for t_idx in range(n_steps):
                    mm.throw_exception_if_processing_interrupted()
                    
                    t = t_steps[t_idx:t_idx+1]
                    t_prev = t_steps[t_idx+1:t_idx+2]
                    
                    t_emb = t_to_emb(t).repeat(x_1.shape[0], 1)
                    
                    patcher = get_vf_model(t[0].item())
                    model = patcher.model
                    
                    # Get vector field prediction using multidiffusion windows
                    vf_output = get_multidiffusion_vf(
                        model, x_t, t_emb, 
                        win_length=win_length, hop_length=hop_length, batch_size=batch_size
                    )
                    
                    pred_x0 = diffusion.get_pred_x0(t, x_t, vf_output)
                    
                    # Data consistency step (mask projection)
                    pred_x0 = pred_x0 * mask_padded + (1-mask_padded) * x_1
                    
                    x_t_prev = diffusion.p_posterior(t_prev, t, x_t, pred_x0, ot_ode=use_ot_ode)
                    x_t = x_t_prev
                    
                    # Add noise back matching mask
                    xt_true = x_1
                    if not use_ot_ode:
                        std_sb = diffusion.get_std_t(t_prev)
                        xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                    x_t = (1. - mask_padded) * xt_true + mask_padded * x_t
                    
                    pbar.update(1)
        except Exception as e:
            if unload_model:
                mm.soft_empty_cache()
            raise e
            
        # Clean up
        pred_x0 = multidiffusion_unpad_outputs(pred_x0, original_width)
        
        if unload_model:
            print("[A2SB] Unloading models from VRAM...")
            # Unload patches
            for p in models:
                p.unpatch_model(mm.get_torch_device())
            mm.soft_empty_cache()

        # 5. Inverse Transforms (STFT -> waveform)
        print("[A2SB] Applying inverse STFT transforms...")
        transform_inv = [
            PowerScaleSpectrogram(power=4, channels=[0], eps=1e-9),
            SpectrogramAddDCTerm(),
            SVDFixMagInstPhase(),
            MagInstPhaseToComplex(),
            InverseComplexSpectrogram(n_fft=2048, win_length=2048, hop_length=512, eps=1e-9)
        ]
        
        # Process each reconstructed channel
        out_channels = []
        for c in range(n_channels):
            spec_c = pred_x0[c]
            for tx in transform_inv:
                spec_c = tx(spec_c)
            out_channels.append(spec_c)
            
        wav_out = torch.stack(out_channels, dim=0).unsqueeze(0).cpu() # [1, C, samples]
        
        return ({"waveform": wav_out, "sample_rate": 44100},)
