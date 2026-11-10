#!/usr/bin/env python3
"""
Step 3: End-to-End 2D to 3D Stereoscopic Video Conversion
Converts any 2D video into Side-by-Side (Full or Half SBS), Top-and-Bottom (Up-Down),
or Anaglyph 3D video with audio preservation and hardware acceleration.
"""

import os
import sys
import argparse
import cv2
import numpy as np

from converter.video_processor import VideoProcessor


def generate_sample_video(filename="sample_input_video.mp4", duration_sec=2.5, fps=24):
    """
    Creates a sample 2D video clip featuring depth motion:
    - A colorful sphere moving forward from the background toward the camera
    - A stationary midground barrier
    - A distant background with landscape gradients
    """
    width, height = 640, 360
    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    print(f"[SampleGenerator] Creating demo video: {filename} ({width}x{height}, {total_frames} frames)...")

    for i in range(total_frames):
        t = i / float(total_frames)  # 0.0 to 1.0
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Background sky and ground gradient
        for y in range(height):
            ratio = y / height
            if y < height // 2:
                # Sky
                frame[y, :] = [int(235 * (1 - ratio)), int(206 * (1 - ratio)), int(135)]
            else:
                # Ground
                frame[y, :] = [34, int(139 * (ratio)), 34]

        # 2. Distant mountains (stationary)
        cv2.fillPoly(frame, [np.array([[50, 180], [180, 70], [310, 180]])], (100, 110, 120))
        cv2.fillPoly(frame, [np.array([[260, 180], [380, 90], [500, 180]])], (110, 120, 130))

        # 3. Midground pillar (stationary)
        cv2.rectangle(frame, (80, 120), (160, 320), (160, 82, 45), -1)
        cv2.putText(frame, "Midground", (85, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 4. Foreground Moving Object: Sphere traveling from distance towards viewer
        # Position and size scale as it gets closer
        sphere_x = int(250 + t * 240)
        sphere_y = int(140 + t * 140)
        sphere_radius = int(20 + t * 50)  # Starts small (distant), gets large (close)

        cv2.circle(frame, (sphere_x, sphere_y), sphere_radius, (0, 165, 255), -1)
        cv2.circle(frame, (sphere_x - sphere_radius // 3, sphere_y - sphere_radius // 3), sphere_radius // 4, (255, 255, 255), -1)

        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    print(f"[SampleGenerator] Created {filename} successfully.")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Convert any 2D video to 3D SBS or Up-Down video.")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input 2D video path (.mp4, .mov, .mkv).")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output 3D video path.")
    parser.add_argument("--format", "-f", type=str, default="sbs", choices=["sbs", "half-sbs", "updown", "half-updown", "anaglyph"], help="Stereoscopic output format.")
    parser.add_argument("--strength", "-s", type=float, default=24.0, help="3D disparity strength in pixels (default: 24.0).")
    parser.add_argument("--convergence", "-c", type=float, default=0.5, help="Zero-parallax plane [0.0..1.0] (default: 0.5).")
    parser.add_argument("--model", "-m", type=str, default="MiDaS_small", choices=["MiDaS_small", "DPT_Hybrid"], help="Depth model architecture.")
    parser.add_argument("--device", "-d", type=str, default=None, choices=["mps", "cuda", "cpu"], help="Hardware acceleration device.")
    parser.add_argument("--scale", type=float, default=1.0, help="Resolution scale factor (e.g. 0.5 for fast testing).")
    parser.add_argument("--smoothing", type=float, default=0.3, help="Temporal depth smoothing factor [0.0..1.0] (default: 0.3).")
    args = parser.parse_args()

    print("=" * 65)
    print("2D-to-3D Stereo Converter - Step 3: Video Processing Pipeline")
    print("=" * 65)

    # 1. Resolve Input Video
    input_video = args.input
    if not input_video or not os.path.exists(input_video):
        print("[Notice] No input video specified. Creating sample demo video...")
        input_video = generate_sample_video("sample_input_video.mp4", duration_sec=2.0, fps=24)

    # 2. Resolve Output Video Name
    if args.output:
        output_video = args.output
    else:
        base, ext = os.path.splitext(input_video)
        output_video = f"{base}_3d_{args.format}.mp4"

    # 3. Run Pipeline
    processor = VideoProcessor(
        strength=args.strength,
        convergence=args.convergence,
        format_type=args.format,
        model_type=args.model,
        device=args.device,
        temporal_smoothing=args.smoothing,
        scale=args.scale,
    )

    result_file = processor.convert_video(input_video, output_video)

    print("=" * 65)
    print("Conversion Complete!")
    print(f"3D Video Deliverable: {result_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
