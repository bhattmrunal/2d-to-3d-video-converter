import cv2
import numpy as np


class StereoRenderer:
    """
    Depth-Image-Based Rendering (DIBR) processor that synthesizes stereoscopic
    views (Left/Right eyes) from a 2D source frame and a depth map.
    """

    def __init__(self, strength=25.0, convergence=0.5):
        """
        Parameters:
        - strength: Maximum pixel disparity shift in pixels (higher = stronger 3D).
        - convergence: Zero-parallax plane (0.0 to 1.0).
                       Objects closer than convergence appear to pop out.
                       Objects further than convergence recede into the display.
        """
        self.strength = float(strength)
        self.convergence = float(convergence)

    def render_left_right(self, image_rgb: np.ndarray, depth_map: np.ndarray):
        """
        Synthesizes Left and Right views using horizontal disparity shifting
        with occlusion hole filling.
        """
        h, w = image_rgb.shape[:2]

        # Calculate per-pixel horizontal disparity shift
        # Positive disparity -> foreground (pop-out)
        # Negative disparity -> background (recede)
        disparity = self.strength * (depth_map - self.convergence)

        # Coordinate grid
        x_grid = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        y_grid = np.tile(np.arange(h)[:, np.newaxis], (1, w)).astype(np.float32)

        # Left eye shifts by -disparity / 2
        # Right eye shifts by +disparity / 2
        map_x_left = (x_grid + (disparity * 0.5)).astype(np.float32)
        map_x_right = (x_grid - (disparity * 0.5)).astype(np.float32)

        # Remap using linear interpolation
        left_eye = cv2.remap(image_rgb, map_x_left, y_grid, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        right_eye = cv2.remap(image_rgb, map_x_right, y_grid, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        return left_eye, right_eye

    def render_sbs(self, left: np.ndarray, right: np.ndarray, half: bool = False) -> np.ndarray:
        """
        Side-by-Side (SBS) stereoscopic layout.
        - Full SBS: Concatenates horizontally (W * 2, H).
        - Half SBS: Squashes each view horizontally by 50% so total resolution is (W, H).
        """
        if half:
            h, w = left.shape[:2]
            left_half = cv2.resize(left, (w // 2, h), interpolation=cv2.INTER_AREA)
            right_half = cv2.resize(right, (w // 2, h), interpolation=cv2.INTER_AREA)
            return np.concatenate([left_half, right_half], axis=1)
        return np.concatenate([left, right], axis=1)

    def render_updown(self, left: np.ndarray, right: np.ndarray, half: bool = False) -> np.ndarray:
        """
        Top-and-Bottom (Up-Down / Over-Under) stereoscopic layout.
        - Full Up-Down: Concatenates vertically (W, H * 2).
        - Half Up-Down: Squashes each view vertically by 50% so total resolution is (W, H).
        """
        if half:
            h, w = left.shape[:2]
            left_half = cv2.resize(left, (w, h // 2), interpolation=cv2.INTER_AREA)
            right_half = cv2.resize(right, (w, h // 2), interpolation=cv2.INTER_AREA)
            return np.concatenate([left_half, right_half], axis=0)
        return np.concatenate([left, right], axis=0)

    def render_anaglyph(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Red-Cyan stereoscopic anaglyph (Red channel from Left eye, Green and Blue from Right eye).
        """
        anaglyph = np.zeros_like(left)
        anaglyph[:, :, 0] = left[:, :, 0]    # Red from Left
        anaglyph[:, :, 1] = right[:, :, 1]   # Green from Right
        anaglyph[:, :, 2] = right[:, :, 2]   # Blue from Right
        return anaglyph
