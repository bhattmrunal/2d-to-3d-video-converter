import os
import shutil
import subprocess
import time
import cv2
import numpy as np
from tqdm import tqdm

from .depth_estimator import DepthEstimator
from .stereo_renderer import StereoRenderer


def get_ffmpeg_binary():
    """Locates the ffmpeg executable on system PATH or standard Homebrew paths."""
    candidates = [
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


class VideoProcessor:
    """
    End-to-end 2D-to-3D video conversion pipeline.
    Handles frame extraction, depth estimation, DIBR stereo synthesis,
    temporal depth smoothing, and audio preservation via FFmpeg.
    """

    def __init__(
        self,
        strength: float = 25.0,
        convergence: float = 0.5,
        format_type: str = "sbs",
        model_type: str = "MiDaS_small",
        device: str = None,
        temporal_smoothing: float = 0.3,
        max_fps: float = None,
        scale: float = 1.0,
    ):
        """
        Parameters:
        - strength: 3D disparity strength in pixels.
        - convergence: Zero-parallax screen plane [0.0..1.0].
        - format_type: 'sbs', 'half-sbs', 'updown', 'half-updown', 'anaglyph'.
        - model_type: 'MiDaS_small' (fastest) or 'DPT_Hybrid'.
        - device: 'mps', 'cuda', or 'cpu' (None = auto-detect).
        - temporal_smoothing: Weight for exponential moving average on depth maps [0.0..1.0].
                              Reduces frame-to-frame depth flickering.
        - max_fps: Limit processing FPS (optional).
        - scale: Scaling factor for frame resolution to speed up conversion (e.g. 0.5 for 50%).
        """
        self.strength = float(strength)
        self.convergence = float(convergence)
        self.format_type = format_type.lower()
        self.temporal_smoothing = float(temporal_smoothing)
        self.scale = float(scale)
        self.ffmpeg_bin = get_ffmpeg_binary()

        # Initialize core components
        self.estimator = DepthEstimator(model_type=model_type, device=device)
        self.renderer = StereoRenderer(strength=self.strength, convergence=self.convergence)

    def _has_audio(self, video_path: str) -> bool:
        """Checks if input video contains an audio stream."""
        if not self.ffmpeg_bin:
            return False
        cmd = [
            self.ffmpeg_bin,
            "-i", video_path,
            "-show_streams",
            "-select_streams", "a",
            "-loglevel", "error"
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return len(result.stdout) > 0
        except Exception:
            return False

    def convert_video(self, input_path: str, output_path: str):
        """Processes an input video file and writes the converted 3D stereoscopic video."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resolution adjustments
        proc_w = int(orig_w * self.scale)
        proc_h = int(orig_h * self.scale)

        # Determine output frame dimensions based on format
        if self.format_type == "sbs":
            out_w, out_h = proc_w * 2, proc_h
        elif self.format_type == "half-sbs":
            out_w, out_h = proc_w, proc_h
        elif self.format_type == "updown":
            out_w, out_h = proc_w, proc_h * 2
        elif self.format_type == "half-updown":
            out_w, out_h = proc_w, proc_h
        elif self.format_type == "anaglyph":
            out_w, out_h = proc_w, proc_h
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")

        # Intermediate video file (without audio)
        temp_video = "temp_stereo_no_audio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video, fourcc, orig_fps, (out_w, out_h))

        print(f"\n[VideoProcessor] Input: {input_path}")
        print(f"[VideoProcessor] Video info: {orig_w}x{orig_h} @ {orig_fps:.2f} FPS ({total_frames} frames)")
        print(f"[VideoProcessor] Processing at: {proc_w}x{proc_h} -> Output: {out_w}x{out_h} ({self.format_type.upper()})")
        print(f"[VideoProcessor] Hardware: {self.estimator.device} | Depth model: {self.estimator.model_type}")
        print(f"[VideoProcessor] 3D Strength: {self.strength} px | Convergence: {self.convergence}\n")

        prev_depth = None
        start_time = time.time()

        pbar = tqdm(total=total_frames, desc="Converting 2D to 3D", unit="frame")

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if self.scale != 1.0:
                    frame_bgr = cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # 1. Depth Estimation
                raw_depth = self.estimator.estimate_depth(frame_rgb)

                # 2. Temporal Smoothing (Exponential Moving Average)
                if prev_depth is not None and self.temporal_smoothing > 0:
                    depth = (1.0 - self.temporal_smoothing) * raw_depth + self.temporal_smoothing * prev_depth
                else:
                    depth = raw_depth
                prev_depth = depth

                # 3. Stereoscopic Rendering
                left_eye, right_eye = self.renderer.render_left_right(frame_rgb, depth)

                # 4. Format Assembly
                if self.format_type == "sbs":
                    stereo_rgb = self.renderer.render_sbs(left_eye, right_eye, half=False)
                elif self.format_type == "half-sbs":
                    stereo_rgb = self.renderer.render_sbs(left_eye, right_eye, half=True)
                elif self.format_type == "updown":
                    stereo_rgb = self.renderer.render_updown(left_eye, right_eye, half=False)
                elif self.format_type == "half-updown":
                    stereo_rgb = self.renderer.render_updown(left_eye, right_eye, half=True)
                elif self.format_type == "anaglyph":
                    stereo_rgb = self.renderer.render_anaglyph(left_eye, right_eye)

                # Write to video
                stereo_bgr = cv2.cvtColor(stereo_rgb, cv2.COLOR_RGB2BGR)
                writer.write(stereo_bgr)

                pbar.update(1)

        finally:
            cap.release()
            writer.release()
            pbar.close()

        elapsed = time.time() - start_time
        fps_proc = total_frames / elapsed if elapsed > 0 else 0
        print(f"\n[VideoProcessor] Rendered {total_frames} frames in {elapsed:.1f}s ({fps_proc:.2f} FPS average).")

        # Re-attach audio track using FFmpeg if available
        if self.ffmpeg_bin:
            print("[VideoProcessor] Finalizing video encoding and muxing audio with FFmpeg...")
            # Command to copy audio stream directly while re-encoding video to standard H.264
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", temp_video,
                "-i", input_path,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-shortest",
                output_path,
                "-loglevel", "warning"
            ]
            res = subprocess.run(cmd)
            if res.returncode == 0 and os.path.exists(output_path):
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                print(f"[VideoProcessor] Successfully created 3D video: {output_path}")
                return output_path
            else:
                print("[VideoProcessor] FFmpeg re-muxing notice. Using direct OpenCV video output.")

        # If FFmpeg not available or failed, move temporary video to output_path
        if os.path.exists(temp_video):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_video, output_path)

        print(f"[VideoProcessor] Video saved to: {output_path}")
        return output_path
