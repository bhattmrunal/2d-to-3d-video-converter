import torch
import cv2
import numpy as np


class DepthEstimator:
    """
    Monocular depth estimation module with automatic hardware acceleration:
    - Apple Silicon Metal (MPS) on macOS
    - NVIDIA CUDA on Windows and Linux
    - CPU fallback
    """

    def __init__(self, model_type="MiDaS_small", device=None):
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_type = model_type
        print(f"[DepthEstimator] Loading '{self.model_type}' on device: {self.device}...")

        # Load model and transforms from Torch Hub
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            self.model_type,
            pretrained=True,
            trust_repo=True
        ).to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        print(f"[DepthEstimator] Ready.")

    def estimate_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Estimates relative depth from an RGB image array (H, W, 3).
        Returns a normalized float32 array (H, W) where:
        - 1.0 represents the nearest foreground objects
        - 0.0 represents the farthest background objects
        """
        orig_h, orig_w = image_rgb.shape[:2]

        # Apply MiDaS input transform and move to target hardware accelerator
        input_batch = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize depth prediction back to original image dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth = prediction.cpu().numpy()

        # Min-max normalization into [0.0, 1.0]
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 1e-6:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)

        return depth_normalized.astype(np.float32)
