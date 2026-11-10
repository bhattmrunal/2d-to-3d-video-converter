# 2D to 3D Stereoscopic Video Converter

A high-performance pipeline for converting 2D media into immersive stereoscopic 3D video. Supports Side-by-Side (Full / Half SBS) and Top-and-Bottom (Up-Down / Over-Under) formats compatible with VR headsets (Meta Quest, Apple Vision Pro), 3D TVs, and spatial displays.

## Features

- **Depth-Image-Based Rendering (DIBR)**: Real-time dual-eye horizontal parallax generation from relative depth maps.
- **Customizable 3D Immersion**:
  - `strength`: Dynamic control over maximum pixel disparity shift.
  - `convergence`: Adjustable zero-parallax plane to balance pop-out effects vs. screen depth and minimize visual fatigue.
- **Multiple Stereoscopic Formats**:
  - Side-by-Side (Full SBS and Half SBS)
  - Top-and-Bottom (Full Up-Down and Half Up-Down)
  - Red-Cyan Anaglyph for rapid previewing on standard screens
- **Hardware Acceleration**:
  - Apple Silicon Metal (`mps`) on macOS
  - NVIDIA CUDA (`cuda`) on Linux and Windows
  - Automatic fallback to multi-threaded CPU processing
- **Video & Audio Pipeline**:
  - Frame-by-frame streaming with temporal depth smoothing (flicker reduction)
  - Audio stream preservation and FFmpeg H.264 muxing
- **Interactive Interfaces**:
  - Web Studio (`app.py`) powered by Streamlit with real-time sliders and video conversion
  - Real-Time Desktop Tuner (`interactive_preview.py`) with instant 60 FPS slider response

## Pipeline Overview

1. **Depth Estimation**: Generates a per-pixel grayscale relative depth representation using monocular depth models (`MiDaS_small` or `DPT_Hybrid`).
2. **Disparity Warping**: Computes eye-specific pixel offsets:
   $$\text{Disparity} = \text{Strength} \times (\text{Depth} - \text{Convergence})$$
3. **Format Assembly**: Packs Left and Right eye views into target stereoscopic layouts.
4. **Encoding & Audio Synchronization**: Assembles processed frames into video while preserving audio tracks via FFmpeg.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Web Studio (Recommended)

Launch the interactive web interface:

```bash
streamlit run app.py
```

* Adjust 3D strength and convergence plane in real-time.
* Preview SBS, Up-Down, and Red-Cyan 3D formats instantly.
* Convert and download full 3D video files.

### 2. High-Speed Desktop Tuner

Preview stereoscopic adjustments at 60 FPS using native desktop trackbars:

```bash
python3 interactive_preview.py --input /path/to/image.jpg
```

* `Strength (px)`: Controls depth separation.
* `Convergence %`: Controls screen plane (pop-out vs. push-back).
* `Mode`: Switches between Anaglyph, Full SBS, Half SBS, and Up-Down.
* Press `s` to save a snapshot; press `q` or `ESC` to exit.

### 3. Video Batch Converter (CLI)

Convert an entire 2D video directly from the terminal:

```bash
python3 step3_convert_video.py \
  --input /path/to/video.mp4 \
  --output video_3d_sbs.mp4 \
  --format sbs \
  --strength 25.0 \
  --convergence 0.5
```

## Project Structure

```text
2d-to-3d-video-converter/
├── converter/
│   ├── __init__.py
│   ├── depth_estimator.py    # Metal (MPS), CUDA, and CPU depth inference
│   ├── stereo_renderer.py    # DIBR parallax engine and layout assemblers
│   └── video_processor.py    # Video streaming, smoothing, and audio muxing
├── app.py                    # Streamlit interactive Web Studio
├── interactive_preview.py    # Real-time desktop trackbar tuner
├── step1_stereo_core.py      # Standalone stereo warper prototype
├── step2_ai_stereo.py        # Image-level AI stereo converter CLI
├── step3_convert_video.py    # Video batch conversion CLI
├── requirements.txt
└── README.md
```

## Author

Mrunal Bhatt ([@bhattmrunal](https://github.com/bhattmrunal))
