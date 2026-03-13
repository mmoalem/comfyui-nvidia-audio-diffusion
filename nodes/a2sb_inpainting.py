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
from a2sb.corruption.corruptions import mask_with_noise
from a2sb.diffusion import multidiffusion_pad_inputs, multidiffusion_unpad_outputs, get_multidiffusion_vf
from a2sb.networks import SinusoidalTemporalEmbedding
from a2sb.utils import find_middle_of_zero_segments

class A2SB_Inpainting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a2sb_model": ("A2SB_MODEL",),
                "audio": ("AUDIO",),
                "steps": ("INT", {"default": 50, "min": 10, "max": 200}),
                "segments": ("STRING", {"default": "1.0-1.5, 3.0-4.0", "multiline": False}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 64}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "inpaint_audio"
    CATEGORY = "audio/A2SB"
    
    def torch_resample(self, wav, orig_sr, target_sr):
        if orig_sr == target_sr:
            return wav
        
        was_1d = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0) # [1, 1, S]
            was_1d = True
        else:
            wav = wav.unsqueeze(0) # [1, C, S]
            
        new_len = int(wav.shape[-1] * target_sr / orig_sr)
        resampled = torch.nn.functional.interpolate(wav, size=new_len, mode='linear', align_corners=False)
        
        if was_1d:
            return resampled.squeeze(0).squeeze(0)
        else:
            return resampled.squeeze(0)

    def parse_segments(self, segments_str):
        # Parses "1.0-1.5, 3.0-4.0" into [(1.0, 1.5), (3.0, 4.0)]
        segments = []
        if not segments_str.strip():
            return segments
        parts = segments_str.split(',')
        for p in parts:
            if '-' in p:
                start, end = p.split('-')
                segments.append((float(start.strip()), float(end.strip())))
        return segments

    def inpaint_audio(self, a2sb_model, audio, steps, segments, batch_size, fast_mode, unload_model):
        mm.throw_exception_if_processing_interrupted()
        device = mm.get_torch_device()
        
        models = a2sb_model["models"]
        t_cutoffs = a2sb_model["t_cutoffs"]
        diffusion = a2sb_model["diffusion"]
        use_ot_ode = a2sb_model["use_ot_ode"]
        dtype = a2sb_model.get("dtype", torch.float32)
        
        t_to_emb = SinusoidalTemporalEmbedding(n_bands=64, min_freq=0.5).to(device).to(dtype)

        # 1. Audio formatting
        audio_waveform = audio["waveform"]
        sr = audio["sample_rate"]
        
        # Handle Batch dimension if present
        if audio_waveform.dim() == 3:
            audio_waveform = audio_waveform[0] # Take first batch [C, S]
            
        if sr != 44100:
            print(f"[A2SB] Resampling input from {sr}Hz to 44100Hz...")
            audio_waveform = self.torch_resample(audio_waveform, sr, 44100)
            sr = 44100

        # A2SB is mono-trained, but we can batch channels
        if audio_waveform.ndim == 1:
            audio_waveform = audio_waveform.unsqueeze(0) # [1, S]
            
        n_channels = audio_waveform.shape[0]
        original_audio = audio_waveform.to(device)

        # 2. Forward Transforms
        print(f"[A2SB] Applying STFT transforms to {n_channels} channel(s)...")
        transform_gt = [
            ComplexSpectrogram(n_fft=2048, win_length=2048, hop_length=512, eps=1e-9),
            ComplexToMagInstPhase(),
            SpectrogramDropDCTerm(),
            PowerScaleSpectrogram(power=0.25, channels=[0], eps=1e-9)
        ]
        
        stft_channels = []
        for c in range(n_channels):
            res = original_audio[c]
            for tx in transform_gt:
                res = tx(res)
            stft_channels.append(res)
            
        # Stack channels: [C, 3, H, W]
        stft_target = torch.stack(stft_channels, dim=0)
        
        # 3. Create Inpainting mask
        parsed_segments = self.parse_segments(segments)
        print(f"[A2SB] Applying inpainting masks for segments: {parsed_segments}")
        
        mask = torch.zeros_like(stft_target)
        hop_length_audio = 512
        
        for start_time, end_time in parsed_segments:
            start_idx = int(44100 / hop_length_audio * start_time)
            end_idx = int(44100 / hop_length_audio * end_time)
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(stft_target.shape[-1], end_idx)
            if start_idx < end_idx:
                mask[:, :, :, start_idx:end_idx] = 1
            
        noise_level = 0.5
        stft_corrupted = mask_with_noise(stft_target, mask, noise_level)
        
        def get_vf_model(t_val):
            model_idx = 0
            for idx, thresh in enumerate(t_cutoffs):
                if t_val >= thresh:
                    model_idx = idx + 1
            return models[model_idx]
            
        mm.load_models_gpu(models)

        # 4. Sampling Loop
        print(f"[A2SB] Starting inpainting over {steps} steps (Batch: {n_channels})...")
        t_steps = torch.linspace(1, 0.05, int(steps)).to(device).to(dtype)
        n_steps = len(t_steps) - 1
        
        win_length = 256
        hop_length = 128
        
        original_width = stft_corrupted.shape[-1]
        x_1 = multidiffusion_pad_inputs(stft_corrupted, win_length, hop_length).to(dtype)
        mask_padded = multidiffusion_pad_inputs(mask, win_length, hop_length, padding_constant=0).to(dtype)
        
        pbar = comfy.utils.ProgressBar(n_steps)
        
        try:
            with torch.no_grad():
                if fast_mode:
                    # In fast mode, we iterate over masked windows
                    print("[A2SB] Using fast inpaint mode (processing masked segments only)")
                    # Process only masked regions (use first channel to find segments)
                    middle_indices = find_middle_of_zero_segments(1 - mask_padded[0, 0, 0])
                    if middle_indices.numel() == 0:
                        print("[A2SB] Warning: No masked segments found for inpainting. Returning original contaminated input.")
                        pred_x0 = x_1
                    else:
                        for center_idx in middle_indices:
                            l_idx = int(center_idx - win_length / 2)
                            r_idx = int(center_idx + win_length / 2)
                            if l_idx < 0:
                                r_idx -= l_idx
                                l_idx = 0
                            if r_idx > x_1.shape[-1]:
                                l_idx -= (r_idx - x_1.shape[-1])
                                r_idx = x_1.shape[-1]
                                
                            curr_x_1 = x_1[:, :, :, l_idx:r_idx].clone()
                            curr_mask = mask_padded[:, :, :, l_idx:r_idx]
                            
                            x_t = curr_x_1.clone()
                            for t_idx in range(n_steps):
                                t = t_steps[t_idx:t_idx+1]
                                t_prev = t_steps[t_idx+1:t_idx+2]
                                t_emb = t_to_emb(t).repeat(curr_x_1.shape[0], 1)
                                
                                patcher = get_vf_model(t[0].item())
                                
                                vf_output = get_multidiffusion_vf(
                                    patcher.model, x_t, t_emb, 
                                    win_length=win_length, hop_length=hop_length, batch_size=batch_size
                                )
                                
                                pred_x0_part = diffusion.get_pred_x0(t, x_t, vf_output)
                                pred_x0_part = pred_x0_part * curr_mask + (1-curr_mask) * curr_x_1
                                
                                x_t_prev = diffusion.p_posterior(t_prev, t, x_t, pred_x0_part, ot_ode=use_ot_ode)
                                x_t = x_t_prev
                                
                                xt_true = curr_x_1
                                if not use_ot_ode:
                                    std_sb = diffusion.get_std_t(t_prev)
                                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                                x_t = (1. - curr_mask) * xt_true + curr_mask * x_t
                                    
                            x_1[:, :, :, l_idx:r_idx] = pred_x0_part
                        pred_x0 = x_1
                else:
                    # Full multidiffusion pass
                    x_t = x_1.clone()
                    for t_idx in range(n_steps):
                        mm.throw_exception_if_processing_interrupted()
                        
                        t = t_steps[t_idx:t_idx+1]
                        t_prev = t_steps[t_idx+1:t_idx+2]
                        t_emb = t_to_emb(t).repeat(x_1.shape[0], 1)
                        
                        patcher = get_vf_model(t[0].item())
                        
                        vf_output = get_multidiffusion_vf(
                            patcher.model, x_t, t_emb, 
                            win_length=win_length, hop_length=hop_length, batch_size=batch_size
                        )
                        
                        pred_x0 = diffusion.get_pred_x0(t, x_t, vf_output)
                        pred_x0 = pred_x0 * mask_padded + (1-mask_padded) * x_1
                        
                        x_t_prev = diffusion.p_posterior(t_prev, t, x_t, pred_x0, ot_ode=use_ot_ode)
                        x_t = x_t_prev
                        
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

        pred_x0 = multidiffusion_unpad_outputs(pred_x0, original_width)
        
        if unload_model:
            print("[A2SB] Unloading models from VRAM...")
            for p in models:
                p.unpatch_model(mm.get_torch_device())
            mm.soft_empty_cache()

        # 5. Inverse Transforms
        print("[A2SB] Applying inverse STFT transforms...")
        transform_inv = [
            PowerScaleSpectrogram(power=4, channels=[0], eps=1e-9),
            SpectrogramAddDCTerm(),
            SVDFixMagInstPhase(),
            MagInstPhaseToComplex(),
            InverseComplexSpectrogram(n_fft=2048, win_length=2048, hop_length=512, eps=1e-9)
        ]
        
        out_channels = []
        for c in range(n_channels):
            spec_c = pred_x0[c]
            for tx in transform_inv:
                spec_c = tx(spec_c)
            out_channels.append(spec_c)
            
        wav_out = torch.stack(out_channels, dim=0).unsqueeze(0).cpu() # [1, C, samples]
        
        return ({"waveform": wav_out, "sample_rate": 44100},)
        
        return ({"waveform": wav_out, "sample_rate": 44100},)
