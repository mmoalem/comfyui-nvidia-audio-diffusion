import torch
import numpy as np

def mask_with_noise(x, mask, noise_level):
    # For stereo coherence, we use the same noise pattern across channels if batch size > 1
    if x.shape[0] > 1:
        # Generate noise for one channel and repeat
        noise_single = torch.randn_like(x[0:1]) * noise_level
        noise = noise_single.repeat(x.shape[0], 1, 1, 1)
    else:
        noise = torch.randn_like(x) * noise_level
    return x * (1 - mask) + mask * noise

class UpsampleMask:
    def __init__(self, min_cutoff_freq: int, max_cutoff_freq: int, sampling_rate: int, dc_dropped: bool=True):
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.sampling_rate = sampling_rate
        self.dc_dropped = dc_dropped

    @staticmethod
    def get_upsample_mask(spec: torch.Tensor, min_cutoff_freq: int, max_cutoff_freq: int, sampling_rate: int, dc_dropped=True):
        c, h, l = spec.shape
        if dc_dropped:
            n_fft = h * 2
        else:
            n_fft = (h - 1) * 2
        inpaint_mask = torch.zeros(c, h, l).to(spec.device)
        low = int(n_fft * min_cutoff_freq / float(sampling_rate))
        high = min(int(n_fft * max_cutoff_freq / float(sampling_rate)), h)
        high = max(high, low + 1)
        
        cutoff = torch.randint(low=low, high=high, size=[1])
        inpaint_mask[:, cutoff[0]:, :] = 1
        return inpaint_mask

    def __call__(self, spec: torch.Tensor):
        return self.get_upsample_mask(spec, self.min_cutoff_freq, self.max_cutoff_freq, self.sampling_rate, self.dc_dropped)

class TimestampedSegmentInpaintMaskTransform:
    def __init__(self, start_time=0.5, end_time=1.0, hop_length=512, sampling_rate=44100, fill_noise_level=0.5):
        self.start_idx = int(sampling_rate / hop_length * start_time)
        self.end_idx = int(sampling_rate / hop_length * end_time)
        self.fill_noise_level = fill_noise_level

    def __call__(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros_like(spec)
        mask[:, :, self.start_idx:self.end_idx] = 1
        masked_and_noised_spec = mask_with_noise(spec, mask, self.fill_noise_level)
        return masked_and_noised_spec, mask

class AutoDeclipMask:
    def __init__(self, threshold=0.99, padding_ms=5, sampling_rate=44100, hop_length=512):
        self.threshold = threshold
        self.padding_bins = int(sampling_rate / hop_length * (padding_ms / 1000.0))

    def __call__(self, waveform: torch.Tensor, spec_height: int, spec_len: int) -> torch.Tensor:
        """
        waveform: [C, S]
        Returns: [C, 3, spec_height, spec_len] mask
        """
        c, s = waveform.shape
        # 1. Detect clipping in time domain
        # waveform is on device
        is_clipped = (waveform.abs() >= self.threshold).float() # [C, S]
        
        # 2. Map to STFT bins (rough approximation)
        # Each STFT frame represents 512 samples
        # We use max-pooling to catch any clipping within the frame
        # [C, 1, S] -> [C, 1, L_stft]
        clipped_frames = torch.nn.functional.max_pool1d(
            is_clipped.unsqueeze(1), 
            kernel_size=512, 
            stride=512
        ).squeeze(1) # [C, L_stft]
        
        # 3. Add padding to masks (dilate)
        if self.padding_bins > 0:
            kernel = torch.ones(1, 1, self.padding_bins * 2 + 1).to(waveform.device)
            clipped_frames = torch.nn.functional.conv1d(
                clipped_frames.unsqueeze(1), 
                kernel, 
                padding=self.padding_bins
            ).squeeze(1) > 0
            clipped_frames = clipped_frames.float()

        # 4. Expand to full 4D mask [C, 3, H, L]
        # [C, 1, 1, L_stft]
        mask = clipped_frames.view(c, 1, 1, -1)
        mask = mask.expand(-1, 3, spec_height, -1).contiguous()
        
        # Ensure length matches spec_len
        if mask.shape[-1] > spec_len:
            mask = mask[..., :spec_len]
        elif mask.shape[-1] < spec_len:
            pad = spec_len - mask.shape[-1]
            mask = torch.cat([mask, mask[..., -1:].repeat(1, 1, 1, pad)], dim=-1)
            
        return mask
