# 2D to 3D Stereoscopic Video Converter

A high-performance pipeline for converting 2D media into immersive stereoscopic 3D video. Supports Side-by-Side (Full / Half SBS) and Top-and-Bottom (Up-Down / Over-Under) formats compatible with VR headsets (Meta Quest, Apple Vision Pro), 3D TVs, and spatial displays.

## Features

- **Depth-Image-Based Rendering (DIBR)**: Real-time dual-eye horizontal parallax generation from relative depth maps.
- **Customizable 3D Immersion**:
  - `strength`: Dynamic control over maximum pixel disparity shift.
  - `convergence`: Adjustable zero-parallax plane to balance pop-out effects vs. screen depth and minimize visual fatigue.
- **Multiple Stereoscopic Formats**:
  - Side-by-Side (SBS)
  - Top-and-Bottom (Up-Down / Over-Under)
  - Red-Cyan Anaglyph for rapid previewing
- **Hardware Acceleration**:
  - Apple Silicon Metal (`mps`) on macOS
  - NVIDIA CUDA (`cuda`) on Linux and Windows
  - Automatic fallback to multi-threaded CPU processing

## Pipeline Overview

1. **Depth Estimation**: Generates a per-pixel grayscale relative depth representation of each frame.
2. **Disparity Warping**: Computes eye-specific pixel offsets:
   $$\text{Disparity} = \text{Strength} \times (\text{Depth} - \text{Convergence})$$
3. **Format Assembly**: Packs Left and Right eye views into target stereoscopic layouts.
4. **Encoding & Audio Synchronization**: Assembles processed frames into video while preserving audio tracks.

## Quick Start

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Step 1 Core Prototype

```bash
python3 step1_stereo_core.py
```

This generates sample 2D and 3D preview files in the working directory:
- `output_sbs_3d.png` (Side-by-Side)
- `output_updown_3d.png` (Top-and-Bottom)
- `output_anaglyph_3d.png` (Red-Cyan 3D)

## Author

Mrunal Bhatt ([@bhattmrunal](https://github.com/bhattmrunal))
