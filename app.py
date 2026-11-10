import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from converter.depth_estimator import DepthEstimator
from converter.stereo_renderer import StereoRenderer
from converter.video_processor import VideoProcessor, get_ffmpeg_binary


st.set_page_config(
    page_title="2D to 3D Stereoscopic Studio",
    page_icon="👓",
    layout="wide",
)


@st.cache_resource
def load_estimator(model_type, device=None):
    return DepthEstimator(model_type=model_type, device=device)


def main():
    st.title("👓 2D to 3D Stereoscopic Studio")
    st.caption("AI-powered monocular depth estimation and Depth-Image-Based Rendering (DIBR) by [Mrunal Bhatt](https://github.com/bhattmrunal)")

    # Sidebar: Controls & Settings
    st.sidebar.header("⚙️ Configuration")

    # Hardware Detection
    import torch
    if torch.backends.mps.is_available():
        hw_label = "Apple Silicon Metal (MPS) ⚡"
    elif torch.cuda.is_available():
        hw_label = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)}) ⚡"
    else:
        hw_label = "CPU (Multi-threaded)"

    st.sidebar.info(f"**Hardware Acceleration:**\n{hw_label}")

    model_type = st.sidebar.selectbox(
        "Depth Model",
        ["MiDaS_small", "DPT_Hybrid"],
        help="MiDaS_small is optimized for speed and video playback. DPT_Hybrid provides finer boundary sharpness."
    )

    st.sidebar.subheader("Stereoscopic Tuning")
    strength = st.sidebar.slider(
        "3D Immersion Strength (px)",
        min_value=0.0,
        max_value=60.0,
        value=25.0,
        step=1.0,
        help="Maximum horizontal disparity shift. Higher values give deeper 3D depth separation."
    )

    convergence = st.sidebar.slider(
        "Convergence Plane",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Zero-parallax plane. Depth values above this plane pop out toward you; values below recede into the display."
    )

    output_format = st.sidebar.selectbox(
        "3D Output Format",
        ["sbs", "half-sbs", "updown", "half-updown", "anaglyph"],
        format_func=lambda x: {
            "sbs": "Side-by-Side (Full SBS)",
            "half-sbs": "Half Side-by-Side (VR Standard)",
            "updown": "Top-and-Bottom (Full Up-Down)",
            "half-updown": "Half Top-and-Bottom",
            "anaglyph": "Red-Cyan Anaglyph (Direct 3D Glasses)",
        }[x]
    )

    tabs = st.tabs(["🖼️ Image & Frame Tuner", "🎬 Video Converter"])

    # -------------------------------------------------------------
    # TAB 1: Image & Frame Tuner
    # -------------------------------------------------------------
    with tabs[0]:
        st.markdown("### Real-Time 3D Immersion Tuner")
        st.write("Upload a frame or photo to visually dial in your 3D strength and convergence plane before converting full video.")

        img_file = st.file_uploader("Upload 2D Image", type=["jpg", "jpeg", "png", "webp"])
        use_sample = st.button("Load Built-in Demo Frame")

        img_to_process = None
        if img_file is not None:
            img_to_process = np.array(Image.open(img_file).convert("RGB"))
        elif use_sample or os.path.exists("step2_input_demo.png"):
            if os.path.exists("step2_input_demo.png"):
                img_to_process = cv2.cvtColor(cv2.imread("step2_input_demo.png"), cv2.COLOR_BGR2RGB)

        if img_to_process is not None:
            with st.spinner("Estimating depth and rendering stereoscopic views..."):
                estimator = load_estimator(model_type)
                depth_map = estimator.estimate_depth(img_to_process)

                renderer = StereoRenderer(strength=strength, convergence=convergence)
                left_eye, right_eye = renderer.render_left_right(img_to_process, depth_map)

                if output_format == "sbs":
                    stereo_out = renderer.render_sbs(left_eye, right_eye, half=False)
                elif output_format == "half-sbs":
                    stereo_out = renderer.render_sbs(left_eye, right_eye, half=True)
                elif output_format == "updown":
                    stereo_out = renderer.render_updown(left_eye, right_eye, half=False)
                elif output_format == "half-updown":
                    stereo_out = renderer.render_updown(left_eye, right_eye, half=True)
                else:
                    stereo_out = renderer.render_anaglyph(left_eye, right_eye)

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.write("**Source 2D Frame**")
                st.image(img_to_process, use_container_width=True)
            with col2:
                st.write("**AI Depth Map**")
                st.image((depth_map * 255).astype(np.uint8), use_container_width=True)
            with col3:
                st.write(f"**Synthesized 3D Output ({output_format.upper()})**")
                st.image(stereo_out, use_container_width=True)

                # Download button
                out_pil = Image.fromarray(stereo_out)
                temp_out = f"converted_3d_{output_format}.png"
                out_pil.save(temp_out)
                with open(temp_out, "rb") as f:
                    st.download_button(
                        label="⬇️ Download 3D Image",
                        data=f,
                        file_name=f"stereo_3d_{output_format}.png",
                        mime="image/png",
                    )

    # -------------------------------------------------------------
    # TAB 2: Video Converter
    # -------------------------------------------------------------
    with tabs[1]:
        st.markdown("### Full 2D-to-3D Video Converter")
        st.write("Convert entire video files with frame-by-frame depth estimation, temporal smoothing, and synchronized audio.")

        vid_col1, vid_col2 = st.columns(2)
        with vid_col1:
            smoothing = st.slider(
                "Temporal Depth Smoothing",
                min_value=0.0,
                max_value=0.8,
                value=0.3,
                step=0.05,
                help="Reduces depth flickering between video frames."
            )
        with vid_col2:
            scale = st.select_slider(
                "Resolution Scale",
                options=[0.5, 0.75, 1.0],
                value=1.0,
                format_func=lambda x: f"{int(x*100)}% resolution ({'Fast preview' if x < 1.0 else 'Full quality'})"
            )

        video_upload = st.file_uploader("Upload 2D Video", type=["mp4", "mov", "mkv"])
        has_sample = os.path.exists("sample_input_video.mp4")

        input_video_path = None
        if video_upload is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_upload.read())
            input_video_path = tfile.name
        elif has_sample:
            st.info("💡 A sample animated demo video is available. You can test immediately or upload your own video above.")
            if st.checkbox("Use built-in sample demo video", value=True):
                input_video_path = "sample_input_video.mp4"

        if input_video_path and st.button("🚀 Start 3D Video Conversion", type="primary"):
            output_video_path = f"converted_video_{output_format}.mp4"
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            processor = VideoProcessor(
                strength=strength,
                convergence=convergence,
                format_type=output_format,
                model_type=model_type,
                temporal_smoothing=smoothing,
                scale=scale,
            )

            status_text.text("Processing video frames... Please wait.")
            try:
                res = processor.convert_video(input_video_path, output_video_path)
                status_text.success("🎉 Conversion Complete!")
                st.video(output_video_path)

                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Converted 3D Video",
                        data=f,
                        file_name=f"converted_3d_{output_format}.mp4",
                        mime="video/mp4",
                    )
            except Exception as e:
                st.error(f"Error during video conversion: {e}")


if __name__ == "__main__":
    main()
