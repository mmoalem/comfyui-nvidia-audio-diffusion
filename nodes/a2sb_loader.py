import os
import torch
import folder_paths
import comfy.model_management as mm
import comfy.model_patcher as mp
from huggingface_hub import hf_hub_download

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2sb.networks import AttnUNetF
from a2sb.diffusion import Diffusion

class A2SB_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["2-split (recommended)", "1-split"], {"default": "2-split (recommended)"}),
                "precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
                "use_ot_ode": ("BOOLEAN", {"default": False}),
                "use_compile": ("BOOLEAN", {"default": False}),
                "attention_type": (["sdpa", "sage"], {"default": "sdpa"}),
            }
        }

    RETURN_TYPES = ("A2SB_MODEL",)
    RETURN_NAMES = ("a2sb_model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/A2SB"
    
    def __init__(self):
        self.model_dir = os.path.join(folder_paths.models_dir, "A2SB")
        os.makedirs(self.model_dir, exist_ok=True)
        self.repo_id = "nvidia/audio_to_audio_schrodinger_bridge"

    def download_model(self, filename):
        file_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(file_path):
            print(f"[A2SB] Downloading {filename} from HuggingFace (~3.4GB)...")
            hf_hub_download(repo_id=self.repo_id, filename=f"ckpt/{filename}", local_dir=self.model_dir, local_dir_use_symlinks=False)
            # HF Hub downloads to 'ckpt/filename' locally because of the path in the repo
            actual_path = os.path.join(self.model_dir, "ckpt", filename)
            if os.path.exists(actual_path):
                import shutil
                shutil.move(actual_path, file_path)
        return file_path

    def instantiate_network(self, attention_type):
        return AttnUNetF(
            n_updown_levels=5,
            in_channels=3,
            hidden_channels=[128, 256, 512, 768, 1024, 2048],
            out_channels=3,
            emb_channels=128,
            band_embedding_dim=16,
            n_attn_heads=8,
            attention_levels=[3, 4],
            use_attn_input_norm=True,
            num_res_blocks=2,
            attention_type=attention_type
        )

    def load_weights(self, network, checkpoint_path, dtype):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        new_state_dict = {}
        for key, value in state_dict.items():
            if "vf_model." in key:
                new_key = key.replace("vf_model.", "")
                # Only apply channels_last to 4D tensors (convolutions)
                v = value.to(dtype)
                if v.ndim == 4:
                    v = v.to(memory_format=torch.channels_last)
                new_state_dict[new_key] = v
        network.load_state_dict(new_state_dict, strict=True)
        return network

    def wrap_model(self, network):
        # Calculate size for ComfyUI memory management
        # Assume FP32 weights initially
        param_count = sum(p.numel() for p in network.parameters())
        model_size_bytes = param_count * 4
        
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        patcher = mp.ModelPatcher(
            network,
            load_device=load_device,
            offload_device=offload_device,
            size=model_size_bytes
        )
        return patcher

    def load_model(self, model_type, precision, use_ot_ode, use_compile, attention_type):
        import logging
        logging.info(f"[A2SB] Loading model type: {model_type}, precision: {precision}, attention: {attention_type}")
        
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        
        models = []
        t_cutoffs = []
        
        if model_type == "2-split (recommended)":
            file1 = "A2SB_twosplit_0.0_0.5_release.ckpt"
            file2 = "A2SB_twosplit_0.5_1.0_release.ckpt"
            
            p1 = self.download_model(file1)
            p2 = self.download_model(file2)
            
            net1 = self.instantiate_network(attention_type)
            net1.to(dtype).to(memory_format=torch.channels_last)
            net1 = self.load_weights(net1, p1, dtype)
            if use_compile:
                print("[A2SB] Compiling model 1...")
                net1 = torch.compile(net1)
            models.append(self.wrap_model(net1))
            
            net2 = self.instantiate_network(attention_type)
            net2.to(dtype).to(memory_format=torch.channels_last)
            net2 = self.load_weights(net2, p2, dtype)
            if use_compile:
                print("[A2SB] Compiling model 2...")
                net2 = torch.compile(net2)
            models.append(self.wrap_model(net2))
            
            t_cutoffs = [0.5]
            
        else:
            file1 = "A2SB_onesplit_0.0_1.0_release.ckpt"
            p1 = self.download_model(file1)
            
            net1 = self.instantiate_network(attention_type)
            net1.to(dtype).to(memory_format=torch.channels_last)
            net1 = self.load_weights(net1, p1, dtype)
            if use_compile:
                print("[A2SB] Compiling model...")
                net1 = torch.compile(net1)
            models.append(self.wrap_model(net1))
            t_cutoffs = []
            
        diffusion = Diffusion(beta_max=1.0) # Config says beta_max: 1.0
        
        return ({
            "models": models,
            "t_cutoffs": t_cutoffs,
            "diffusion": diffusion,
            "use_ot_ode": use_ot_ode,
            "dtype": dtype
        },)
