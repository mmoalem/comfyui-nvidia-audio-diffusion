# ComfyUI-A2SB: High-Fidelity Audio Restoration

A ComfyUI custom node suite for **Audio-to-Audio Schrödinger Bridges (A2SB)**. This pack brings state-of-the-art audio restoration, bandwidth extension, and inpainting to the ComfyUI ecosystem, with deep optimizations for modern NVIDIA GPUs (Blackwell/Ada).

![Screenshot of A2SB Workflow](https://github.com/user-attachments/assets/placeholder)

## 🌟 Key Features

- **Bandwidth Extension (BWE)**: Restore high frequencies up to 44.1kHz from low-quality recordings.
- **Advanced Inpainting**: Seamlessly "heal" audio segments using diffusion.
- **Auto-Declipper**: Automatically detects and repairs clipped/distorted audio peaks.
- **Refiner Mode**: Polishes existing audio textures to remove "metallic" artifacts or digital crunchiness.
- **Blackwell-Ready Design**:
  - **SageAttention Support**: Native integration for lightning-fast attention on RTX 40/50 series.
  - **Broadcasting Optimization**: Minimum memory traffic during multidiffusion.
  - **Channels Last Format**: Optimized for NVIDIA Tensor Cores.
  - **Torch Compile Support**: Maximum throughput via kernel fusion.

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   cd custom_nodes
   git clone https://github.com/mmoalem/comfyui-nvidia-audio-diffusion
   ```

2. **Install dependencies**:
   Run the `install.py` script included in the folder or manually install:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For maximum performance on Blackwell/Ada GPUs, it is highly recommended to install [SageAttention](https://github.com/thu-ml/SageAttention).*

3. **Models**:
   The nodes will automatically download the required checkpoints (~6.8GB) from HuggingFace on first use. Models are saved to `ComfyUI/models/A2SB`.

## 🛠️ Usage

### A2SB Model Loader
- **model_type**: Choose between `1-split` (fast) or `2-split` (higher quality).
- **precision**: `bf16` is recommended for modern GPUs.
- **attention_type**: Select `sage` if you have SageAttention installed, otherwise use `sdpa`.
- **use_compile**: Enable for a permanent speed boost after a short initial compilation delay.

### A2SB Bandwidth Extension
- Use this to fix muffled audio.
- Set `cutoff_freq` to `0` for auto-detection, or manually (e.g., `8000`) to force regeneration of the high-end (excellent for fixing metallic ACE-Step audio).

### A2SB Inpainting
- **auto_declip**: Turn this ON to automatically fix distortion in loud recordings.
- **refiner_strength**: Set between `0.1` and `0.5` to "smooth out" crunchy audio without changing the content.

## 📜 License & Credits

### Credits
- **Original Research**: Developed by **NVIDIA CORPORATION**. Based on the paper *"Audio-to-Audio Schrödinger Bridges"* ([Research Paper](https://arxiv.org/abs/2403.07634)).
- **Architecture**: U-Net implementation inspired by OpenAI's **Guided Diffusion**.

### License
- **A2SB Source Code**: Licensed under the **NVIDIA Source Code License for A2SB**. See the accompanying `LICENSE` file for full terms (Non-commercial, Research only).
- **Custom Node Wrapper**: Licensed under the MIT License.

---
*Developed for the ComfyUI community to enable next-generation audio processing.*
