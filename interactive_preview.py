#!/usr/bin/env python3
"""
Step 4: Real-Time Interactive 3D Tuner (OpenCV Desktop GUI)
Provides instant desktop trackbars to dial in 3D Depth Strength,
Convergence Plane, and View Modes in real time.
"""

import os
import sys
import argparse
import cv2
import numpy as np

from converter.depth_estimator import DepthEstimator
from converter.stereo_renderer import StereoRenderer


def main():
    parser = argparse.ArgumentParser(description="Real-time interactive 3D tuner.")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input image path.")
    parser.add_argument("--model", "-m", type=str, default="MiDaS_small", help="Depth model architecture.")
    args = parser.parse_args()

    # Load or generate frame
    if args.input and os.path.exists(args.input):
        bgr = cv2.imread(args.input)
    elif os.path.exists("step2_input_demo.png"):
        bgr = cv2.imread("step2_input_demo.png")
    else:
        print("Please provide an input image via --input or run step2_ai_stereo.py first.")
        sys.exit(1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Pre-compute Depth Map once so dragging sliders is instant (60+ FPS)
    print("Pre-computing AI depth map for instant 60 FPS interactive slider response...")
    estimator = DepthEstimator(model_type=args.model)
    depth_map = estimator.estimate_depth(rgb)
    print("Depth map ready! Opening interactive window...")

    window_name = "2D to 3D Immersion Studio - Press 'q' or ESC to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(w * 2, 1280), min(h, 720))

    # Create Trackbars
    # Trackbar 1: 3D Strength (0 to 60 px)
    cv2.createTrackbar("Strength (px)", window_name, 25, 60, lambda x: None)

    # Trackbar 2: Convergence Plane (0 to 100 -> mapped to 0.0 to 1.0)
    cv2.createTrackbar("Convergence %", window_name, 50, 100, lambda x: None)

    # Trackbar 3: Format (0 = Anaglyph, 1 = SBS Full, 2 = SBS Half, 3 = Up-Down)
    cv2.createTrackbar("Mode: 0=Ana 1=SBS 2=HSBS 3=UD", window_name, 0, 3, lambda x: None)

    renderer = StereoRenderer(strength=25.0, convergence=0.5)

    print("\nControls:")
    print("  - Drag 'Strength (px)' to increase/decrease depth separation")
    print("  - Drag 'Convergence %' to adjust screen plane (pop-out vs. recede)")
    print("  - Drag 'Mode' to switch between Anaglyph, Full SBS, Half SBS, and Up-Down")
    print("  - Press 's' to save current view to disk")
    print("  - Press 'q' or ESC to exit\n")

    current_output = None

    while True:
        # Read trackbar values
        strength_val = float(cv2.getTrackbarPos("Strength (px)", window_name))
        conv_val = float(cv2.getTrackbarPos("Convergence %", window_name)) / 100.0
        mode_val = cv2.getTrackbarPos("Mode: 0=Ana 1=SBS 2=HSBS 3=UD", window_name)

        renderer.strength = strength_val
        renderer.convergence = conv_val

        # Render Left & Right views
        left_eye, right_eye = renderer.render_left_right(rgb, depth_map)

        # Select layout
        if mode_val == 0:
            out_rgb = renderer.render_anaglyph(left_eye, right_eye)
        elif mode_val == 1:
            out_rgb = renderer.render_sbs(left_eye, right_eye, half=False)
        elif mode_val == 2:
            out_rgb = renderer.render_sbs(left_eye, right_eye, half=True)
        else:
            out_rgb = renderer.render_updown(left_eye, right_eye, half=True)

        current_output = out_rgb
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, out_bgr)
        key = cv2.waitKey(20) & 0xFF
        if key in [ord("q"), 27]:
            break
        elif key == ord("s"):
            save_name = f"interactive_snapshot_{int(strength_val)}px_mode{mode_val}.png"
            cv2.imwrite(save_name, out_bgr)
            print(f"Saved snapshot to: {save_name}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
