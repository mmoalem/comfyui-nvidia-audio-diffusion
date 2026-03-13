import torch
import sys
import os

sys.path.insert(0, 'e:/AI/NvidiaAudioDiffusion/ComfyUI-A2SB')
from a2sb.networks import AttnUNetF

network = AttnUNetF(
            n_updown_levels=5,
            in_channels=3,
            hidden_channels=[128, 256, 512, 768, 1024, 2048],
            out_channels=3,
            emb_channels=128,
            band_embedding_dim=16,
            n_attn_heads=8,
            attention_levels=[3, 4],
            use_attn_input_norm=True,
            num_res_blocks=2
        )

ckpt = torch.load(r'e:/AI/ComfyUI_windows_portable/ComfyUI/models/A2SB/A2SB_twosplit_0.0_0.5_release.ckpt', map_location='cpu')

new_state_dict = {}
for key, value in ckpt['state_dict'].items():
    if "vf_model." in key:
        new_key = key.replace("vf_model.", "")
        new_state_dict[new_key] = value

network.load_state_dict(new_state_dict)
print('Successfully loaded the weights into AttnUNetF!')
