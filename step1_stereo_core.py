#!/usr/bin/env python3
"""
Step 1: Foundational 2D to 3D Stereoscopic Warper
Demonstrates the core math used by systems like Owl3D and VisionDepth3D:
- Depth-based pixel disparity shift
- Adjustable 3D strength and convergence plane
- Side-by-Side (SBS), Top-and-Bottom (Up-Down), and Anaglyph rendering
"""

import sys
import numpy as np
from PIL import Image, ImageDraw


def create_sample_scene(width=640, height=360):
    """
    Creates a sample 2D frame and a matching depth map.
    Depth values: 1.0 (very close / foreground) to 0.0 (far away / background).
    """
    # 1. Background image (gradient sky and ground)
    img = Image.new("RGB", (width, height), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    # Ground
    draw.rectangle([0, height // 2, width, height], fill=(34, 139, 34))
    # Distant mountain (midground)
    draw.polygon([(80, height // 2), (240, 80), (400, height // 2)], fill=(90, 105, 120))
    # Close tree / pillar (foreground)
    draw.rectangle([440, 100, 520, height - 30], fill=(139, 69, 19))
    draw.ellipse([400, 60, 560, 180], fill=(46, 139, 87))

    # 2. Matching Depth Map
    # Background is dark (far, 0.1), ground gradients forward (0.1 to 0.6)
    depth = np.full((height, width), 0.1, dtype=np.float32)
    # Ground depth gradient
    for y in range(height // 2, height):
        depth[y, :] = 0.2 + 0.5 * ((y - height // 2) / (height // 2))
    # Mountain depth (midground ~0.35)
    # Tree depth (foreground ~0.9)
    depth_img = Image.new("L", (width, height), color=25)
    d_draw = ImageDraw.Draw(depth_img)
    # Ground gradient
    for y in range(height // 2, height):
        val = int(255 * (0.2 + 0.5 * ((y - height // 2) / (height // 2))))
        d_draw.line([(0, y), (width, y)], fill=val)
    # Mountain
    d_draw.polygon([(80, height // 2), (240, 80), (400, height // 2)], fill=90)
    # Foreground tree
    d_draw.rectangle([440, 100, 520, height - 30], fill=230)
    d_draw.ellipse([400, 60, 560, 180], fill=230)

    depth_array = np.array(depth_img, dtype=np.float32) / 255.0
    return img, depth_array, depth_img


def render_stereoscopic_views(image_np, depth_map, strength=20.0, convergence=0.5):
    """
    Renders Left and Right eye views using Depth-Image-Based Rendering (DIBR).
    
    Parameters:
    - image_np: RGB image array (H, W, 3), values 0..255
    - depth_map: 2D array (H, W), normalized 0.0 (far) to 1.0 (near)
    - strength: Maximum pixel disparity shift in pixels (higher = deeper 3D)
    - convergence: Depth value where objects appear at screen depth (zero parallax)
                   Values > convergence pop OUT of the screen.
                   Values < convergence recede INTO the screen.
    """
    height, width, channels = image_np.shape

    # Calculate disparity: near objects shift opposite to far objects
    # disparity > 0 means pop-out, disparity < 0 means push-back
    disparity = strength * (depth_map - convergence)

    # Output eye buffers
    left_eye = np.zeros_like(image_np)
    right_eye = np.zeros_like(image_np)

    # Base grid of x coordinates
    x_indices = np.arange(width)

    for y in range(height):
        disp_row = disparity[y]

        # Left eye shifts by -disp / 2
        # Right eye shifts by +disp / 2
        src_x_left = np.clip(np.round(x_indices + (disp_row * 0.5)).astype(int), 0, width - 1)
        src_x_right = np.clip(np.round(x_indices - (disp_row * 0.5)).astype(int), 0, width - 1)

        left_eye[y] = image_np[y, src_x_left]
        right_eye[y] = image_np[y, src_x_right]

    return left_eye, right_eye


def create_3d_formats(left_eye, right_eye):
    """
    Assembles Left and Right views into SBS, Up-Down, and Red-Cyan Anaglyph.
    """
    # 1. Side-by-Side (SBS) - horizontally stacked
    sbs = np.concatenate([left_eye, right_eye], axis=1)

    # 2. Up-Down (Top-and-Bottom / Over-Under) - vertically stacked
    up_down = np.concatenate([left_eye, right_eye], axis=0)

    # 3. Red-Cyan Anaglyph (Left = Red channel, Right = Green & Blue channels)
    anaglyph = np.zeros_like(left_eye)
    anaglyph[:, :, 0] = left_eye[:, :, 0]    # Red from Left
    anaglyph[:, :, 1] = right_eye[:, :, 1]   # Green from Right
    anaglyph[:, :, 2] = right_eye[:, :, 2]   # Blue from Right

    return sbs, up_down, anaglyph


def main():
    print("=" * 60)
    print("2D-to-3D Stereo Converter - Step 1: Foundational DIBR Core")
    print("=" * 60)

    # Settings
    strength = 24.0       # 3D immersion strength (pixel disparity)
    convergence = 0.5    # Zero-parallax plane (0.5 = midground at screen surface)

    print(f"[1/4] Generating synthetic scene with depth layers...")
    scene_img, depth_map, depth_vis = create_sample_scene()
    image_np = np.array(scene_img)

    print(f"[2/4] Applying Depth-Image-Based Rendering (DIBR)...")
    print(f"      - 3D Strength (disparity): {strength} px")
    print(f"      - Convergence Plane: {convergence} (midground)")
    left_view, right_view = render_stereoscopic_views(image_np, depth_map, strength=strength, convergence=convergence)

    print(f"[3/4] Assembling 3D formats (SBS, Up-Down, Red-Cyan Anaglyph)...")
    sbs_np, up_down_np, anaglyph_np = create_3d_formats(left_view, right_view)

    # Save images
    scene_img.save("sample_scene_2d.png")
    depth_vis.save("sample_depth_map.png")
    Image.fromarray(sbs_np).save("output_sbs_3d.png")
    Image.fromarray(up_down_np).save("output_updown_3d.png")
    Image.fromarray(anaglyph_np).save("output_anaglyph_3d.png")

    print("[4/4] Done! Generated the following files in your folder:")
    print("      * sample_scene_2d.png      (Original 2D frame)")
    print("      * sample_depth_map.png     (Relative depth map)")
    print("      * output_sbs_3d.png        (Side-by-Side 3D format)")
    print("      * output_updown_3d.png     (Top-and-Bottom / Up-Down 3D format)")
    print("      * output_anaglyph_3d.png   (Red-Cyan 3D format for direct viewing)")
    print("=" * 60)


if __name__ == "__main__":
    main()
