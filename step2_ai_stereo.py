#!/usr/bin/env python3
"""
Step 2: AI-Powered Stereoscopic Conversion
Takes any 2D image, predicts a per-pixel depth map using an AI model with
hardware acceleration (Metal on Apple Silicon, CUDA on Windows/Linux),
and converts it to 3D SBS, Up-Down, and Anaglyph.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from PIL import Image

from converter.depth_estimator import DepthEstimator
from converter.stereo_renderer import StereoRenderer


def create_demo_image():
    """Generates a high-contrast test image if no input image is provided."""
    w, h = 800, 450
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background
    for y in range(h):
        ratio = y / h
        img[y, :] = [int(180 * (1 - ratio)), int(140 * (1 - ratio)), int(220 * ratio)]

    # Distant circle
    cv2.circle(img, (220, 160), 60, (255, 215, 0), -1)

    # Midground rectangle
    cv2.rectangle(img, (100, 260), (380, 420), (50, 180, 50), -1)

    # Foreground pillar
    cv2.rectangle(img, (480, 120), (620, 430), (220, 60, 60), -1)
    cv2.putText(img, "Foreground", (490, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Midground", (150, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Background", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return img


def main():
    parser = argparse.ArgumentParser(description="Convert any 2D image into 3D SBS or Up-Down using AI depth estimation.")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input 2D image (JPEG/PNG).")
    parser.add_argument("--strength", "-s", type=float, default=25.0, help="3D disparity strength in pixels (default: 25.0).")
    parser.add_argument("--convergence", "-c", type=float, default=0.5, help="Zero-parallax plane [0.0..1.0] (default: 0.5).")
    parser.add_argument("--format", "-f", type=str, default="all", choices=["sbs", "half-sbs", "updown", "half-updown", "anaglyph", "all"], help="Stereo output format.")
    parser.add_argument("--model", "-m", type=str, default="MiDaS_small", choices=["MiDaS_small", "DPT_Hybrid"], help="Depth model architecture.")
    parser.add_argument("--device", "-d", type=str, default=None, choices=["mps", "cuda", "cpu"], help="Hardware acceleration device.")
    args = parser.parse_args()

    print("=" * 65)
    print("2D-to-3D Stereo Converter - Step 2: AI Depth & Stereo Synthesis")
    print("=" * 65)

    # 1. Load Input Image
    if args.input and os.path.exists(args.input):
        print(f"[1/4] Loading input image: {args.input}")
        bgr = cv2.imread(args.input)
        if bgr is None:
            print(f"Error: Could not open {args.input}")
            sys.exit(1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        print("[1/4] No input image provided. Generating demo test image...")
        rgb = create_demo_image()
        cv2.imwrite("step2_input_demo.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print("      Saved demo frame to: step2_input_demo.png")

    # 2. Run AI Depth Estimation
    estimator = DepthEstimator(model_type=args.model, device=args.device)

    print(f"[2/4] Estimating depth with {args.model} on {estimator.device}...")
    t0 = time.time()
    depth_map = estimator.estimate_depth(rgb)
    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"      Depth estimation completed in {elapsed_ms:.1f} ms")

    # Save depth visualization
    depth_vis = (depth_map * 255.0).astype(np.uint8)
    cv2.imwrite("step2_estimated_depth.png", depth_vis)
    print("      Saved depth map to: step2_estimated_depth.png")

    # 3. Render 3D Stereoscopic Views
    print(f"[3/4] Synthesizing stereoscopic views...")
    print(f"      - 3D Strength: {args.strength} px")
    print(f"      - Convergence Plane: {args.convergence}")

    renderer = StereoRenderer(strength=args.strength, convergence=args.convergence)
    left_eye, right_eye = renderer.render_left_right(rgb, depth_map)

    # 4. Save Selected Formats
    print(f"[4/4] Assembling stereoscopic format: '{args.format}'...")

    def save_rgb(filename, img_array):
        cv2.imwrite(filename, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        print(f"      Saved: {filename}")

    if args.format in ["sbs", "all"]:
        save_rgb("step2_output_sbs.png", renderer.render_sbs(left_eye, right_eye, half=False))
    if args.format in ["half-sbs", "all"]:
        save_rgb("step2_output_half_sbs.png", renderer.render_sbs(left_eye, right_eye, half=True))
    if args.format in ["updown", "all"]:
        save_rgb("step2_output_updown.png", renderer.render_updown(left_eye, right_eye, half=False))
    if args.format in ["half-updown", "all"]:
        save_rgb("step2_output_half_updown.png", renderer.render_updown(left_eye, right_eye, half=True))
    if args.format in ["anaglyph", "all"]:
        save_rgb("step2_output_anaglyph.png", renderer.render_anaglyph(left_eye, right_eye))

    print("=" * 65)
    print("Step 2 completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
