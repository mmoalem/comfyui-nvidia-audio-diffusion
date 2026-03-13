from .a2sb_loader import A2SB_ModelLoader
from .a2sb_bandwidth_extension import A2SB_BandwidthExtension
from .a2sb_inpainting import A2SB_Inpainting

NODE_CLASS_MAPPINGS = {
    "A2SB_ModelLoader": A2SB_ModelLoader,
    "A2SB_BandwidthExtension": A2SB_BandwidthExtension,
    "A2SB_Inpainting": A2SB_Inpainting
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "A2SB_ModelLoader": "Load A2SB Audio Model",
    "A2SB_BandwidthExtension": "A2SB Bandwidth Extension",
    "A2SB_Inpainting": "A2SB Audio Inpainting"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
