"""
Interactive Optical Flow Registration App
Using scikit-image for registration and Streamlit for UI
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.metrics import normalized_root_mse, structural_similarity
import tempfile
import urllib.request
import cv2

try:
    from streamlit_image_comparison import image_comparison
    HAS_IMAGE_COMPARISON = True
except ImportError:
    HAS_IMAGE_COMPARISON = False

st.set_page_config(page_title="Optical Flow Registration", layout="wide")
st.title("🔄 Optical Flow Registration")

tab_app, tab_edu = st.tabs(["Registration Tool", "What is Optical Flow Registration?"])

# ── Educational tab ────────────────────────────────────────────────────────────
with tab_edu:
    st.header("Optical Flow for Image Registration")
    st.write(
        "Image registration is the process of aligning two images of the same scene so that "
        "corresponding structures overlap. Optical flow is one way to compute the dense "
        "pixel-by-pixel displacement field needed to do that alignment."
    )

    st.subheader("The core problem")
    st.write(
        "Suppose you have two frames from a video. Between them, something moved — "
        "a cell under a microscope, a patient breathing during an MRI, a camera shifting slightly. "
        "Registration asks: *for each pixel in frame A, where did it end up in frame B?* "
        "Once you know that, you can warp frame B back onto frame A so the two line up."
    )

    st.subheader("What optical flow computes")
    st.write(
        "Optical flow estimates a vector field **u(x,y), v(x,y)** — one 2D displacement vector "
        "per pixel. The vector at each location says: the content here moved Δx pixels "
        "horizontally and Δy pixels vertically. Stacked together, these vectors form the "
        "*displacement field* or *warp field*."
    )
    st.code(
        "# The warp: for every pixel (r, c) in the reference,\n"
        "# sample frame B at (r + v[r,c], c + u[r,c])\n"
        "image_registered = warp(frame_B, [rows + v, cols + u])",
        language="python"
    )

    st.subheader("The two algorithms in this app")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**TVL1 (Total Variation L1)**")
        st.write(
            "Minimises a combined energy: an L1 data term that measures brightness "
            "consistency, plus a Total Variation regulariser that keeps the field "
            "spatially smooth while still allowing sharp motion boundaries. "
            "Fast and robust to noise — good default choice."
        )
    with col2:
        st.markdown("**iLK (Iterative Lucas-Kanade)**")
        st.write(
            "Assumes brightness is locally constant within a window of radius *r* "
            "around each pixel, then solves a least-squares system for the local "
            "velocity. Iterating and coarsening the pyramid extends its range. "
            "More detail in smooth regions; struggles with large displacements."
        )

    st.subheader("How to read the outputs")

    with st.expander("Registered output frame"):
        st.write(
            "The registered frame is the result of warping the moving frame (t+1) onto the "
            "reference frame (t) using the estimated displacement field. It should look like "
            "the reference — structures that were shifted, rotated, or deformed are moved back "
            "into alignment. Compare it side-by-side with the raw moving frame to see how much "
            "the warp corrected."
        )

    with st.expander("Color-bleed overlay"):
        st.write(
            "The reference frame is placed in the **green and blue channels** (cyan), "
            "the other frame in the **red channel**. Where the two frames disagree spatially "
            "you see red or cyan fringing — like misaligned colour printing. "
            "After registration the warp brings the frames into alignment and the fringing collapses to grey."
        )

    with st.expander("Residual error map"):
        st.write(
            "Shows the pixel-wise absolute difference |A − B| before and after registration, "
            "on the same colour scale. Bright regions are where the frames still disagree. "
            "A successful registration turns a structured, high-contrast error map into a "
            "flat, dim one. Residual error that follows object boundaries usually means the "
            "flow model couldn't capture that motion (e.g. occlusion or very large displacement)."
        )

    with st.expander("Displacement field / quiver plot"):
        st.write(
            "Each arrow shows the estimated motion vector at that location. "
            "Arrow **colour encodes magnitude** — how many pixels that location moved — "
            "using a cool→warm colormap (blue = small displacement, red = large). "
            "Length encodes direction and relative speed. "
            "A uniform field of parallel arrows means rigid translation; "
            "a swirling or diverging field means rotation or zoom."
        )

    with st.expander("NRMSE and SSIM metrics"):
        st.write(
            "**NRMSE** (Normalised Root Mean Square Error) measures average pixel-level "
            "disagreement — lower is better. "
            "**SSIM** (Structural Similarity Index) measures perceived similarity in "
            "luminance, contrast, and structure — higher is better, maximum 1.0. "
            "Together they give a before/after score for how much registration helped. "
            "Neither is perfect: NRMSE is sensitive to global brightness shifts, "
            "SSIM can be fooled by blurring. Use both together."
        )

    st.subheader("Common applications")
    st.write(
        "Optical flow registration is used across many fields: aligning consecutive "
        "microscopy frames to correct stage drift, stabilising endoscopy video, "
        "correcting respiratory motion in medical imaging, stitching satellite imagery, "
        "and computing motion-compensated video compression."
    )

    st.info(
        "💡 **Tip:** For best results, keep inter-frame motion small relative to the "
        "image size (ideally < 10% of frame width). Large motions exceed the capture "
        "range of both algorithms. Use the *start frame* and *downsampling* controls "
        "to find a range with moderate, visible motion."
    )

# ── App tab ────────────────────────────────────────────────────────────────────
with tab_app:

    if not HAS_IMAGE_COMPARISON:
        st.info(
            "Install `streamlit-image-comparison` for an interactive before/after slider: "
            "`pip install streamlit-image-comparison`"
        )

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        algorithm = st.radio(
            "Algorithm:",
            ["iLK (Iterative Lukas-Kanade) - Fast", "TVL1 (Total Variation L1) - Slow"],
            help="iLK is faster, TVL1 is more detailed",
            index=0
        )

        if algorithm == "iLK (Iterative Lukas-Kanade) - Fast":
            radius = st.slider("Radius", 5, 25, 15)
        else:
            radius = 15

        show_quiver = st.checkbox("Show vector field", value=False)
        nvec = st.slider("Vectors per dimension", 10, 100, 60) if show_quiver else 60

    # ----------------------------
    # Video download helper (cached so it only downloads once per URL)
    # ----------------------------
    @st.cache_data(show_spinner="Downloading example video…")
    def download_video(url: str) -> str:
        """Download a remote video to a local temp file and return its path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            return tmp.name

    # ----------------------------
    # Video upload
    # ----------------------------

    st.subheader("Example Videos")

    example_videos = {
        "MRI heartbeat": "https://github.com/sherylposting/data/raw/a47d974d6bd8636b12c1a9a094acea9d2f535339/heartbeat.mp4",
        "Tardigrade riding a nematode": "https://github.com/sherylposting/data/raw/a47d974d6bd8636b12c1a9a094acea9d2f535339/nematode.mp4",
        "Drifting volvox algae": "https://github.com/sherylposting/data/raw/a47d974d6bd8636b12c1a9a094acea9d2f535339/volvox.mp4",
        "None (upload your own)": None,
    }

    selected_example = st.selectbox(
        "Choose an example video",
        list(example_videos.keys())
    )

    st.subheader("Upload Video")
    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    # Resolve video_source to a local file path string
    video_source = None

    if selected_example != "None (upload your own)":
        # Download example video to a temp file (cached)
        video_source = download_video(example_videos[selected_example])

    elif video_file is not None:
        # Write uploaded bytes to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.getvalue())
            video_source = tmp.name

    if video_source is None:
        st.info("Select an example video or upload your own to begin.")
        st.stop()

    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    st.info(f"📹 Video loaded: {total_frames} frames @ {fps:.1f} FPS")

    # ----------------------------
    # Frame selection controls
    # ----------------------------
    st.sidebar.header("Processing Range")

    start_frame = st.sidebar.slider(
        "Start frame",
        0,
        max(0, total_frames - 2),
        0
    )

    max_possible = max(1, total_frames - start_frame - 1)

    num_frames = st.sidebar.slider(
        "Number of frame pairs to process",
        1,
        min(200, max_possible),
        min(10, max_possible)
    )

    scale = st.sidebar.slider(
        "Frame downsampling (lowers resolution, increases speed)",
        1,
        4,
        2
    )

    if "flow_results" not in st.session_state:
        st.session_state.flow_results = None

    # ----------------------------
    # Processing
    # ----------------------------
    if st.button("🎬 Process Selected Range", type="primary"):

        progress_bar = st.progress(0)
        status = st.empty()

        cap = cv2.VideoCapture(video_source)

        for _ in range(start_frame):
            cap.read()

        frames = []
        for _ in range(num_frames + 1):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if scale > 1:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (w // scale, h // scale))
            frames.append(frame)

        cap.release()

        if len(frames) < 2:
            st.error("Not enough frames in selected range.")
            st.stop()

        results = []
        st.write(f"Processing {len(frames) - 1} frame pairs...")

        for i in range(len(frames) - 1):

            progress_bar.progress((i + 1) / (len(frames) - 1))
            status.text(f"Processing pair {i + 1}/{len(frames) - 1}")

            image0 = rgb2gray(frames[0])
            image1 = rgb2gray(frames[i + 1])

            if algorithm == "TVL1 (Total Variation L1) - Slow":
                v, u = optical_flow_tvl1(image0, image1)
            else:
                v, u = optical_flow_ilk(image0, image1, radius=radius)

            nr, nc = image0.shape
            rr, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")

            image1_warp = np.clip(
                warp(image1, np.array([rr + v, cc + u]), mode="edge"),
                0.0, 1.0
            )

            diff_before = np.abs(image1 - image0)
            diff_after  = np.abs(image1_warp - image0)

            nrmse_before = normalized_root_mse(image0, image1)
            nrmse_after  = normalized_root_mse(image0, image1_warp)

            ssim_before = structural_similarity(image0, image1, data_range=1.0)
            ssim_after  = structural_similarity(image0, image1_warp, data_range=1.0)

            def make_composite(ref, moving):
                comp = np.zeros((*ref.shape, 3))
                comp[..., 0] = moving
                comp[..., 1] = ref
                comp[..., 2] = ref
                return np.clip(comp, 0, 1)

            unreg_overlay = make_composite(image0, image1)
            reg_overlay   = make_composite(image0, image1_warp)

            # Convert warped grayscale to uint8 RGB for display
            warp_rgb = (image1_warp * 255).astype(np.uint8)
            warp_rgb = np.stack([warp_rgb, warp_rgb, warp_rgb], axis=-1)

            results.append({
                "frame0":         frames[0],
                "frame1":         frames[i + 1],
                "image1_warp":    image1_warp,
                "warp_rgb":       warp_rgb,
                "unreg_overlay":  unreg_overlay,
                "reg_overlay":    reg_overlay,
                "diff_before":    diff_before,
                "diff_after":     diff_after,
                "u":              u,
                "v":              v,
                "flow_magnitude": np.sqrt(u**2 + v**2),
                "shape":          (nr, nc),
                "nrmse_before":   nrmse_before,
                "nrmse_after":    nrmse_after,
                "ssim_before":    ssim_before,
                "ssim_after":     ssim_after,
            })

        st.session_state.flow_results = results
        progress_bar.empty()
        status.empty()
        st.success(f"✅ Processed {len(results)} frame pairs")

    # ----------------------------
    # Cached figure builders
    # ----------------------------
    @st.cache_data
    def build_overlay_fig(unreg, reg, u, v, shape, show_quiver, nvec):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(unreg)
        axes[0].set_title("Before registration  (red/cyan fringing = misalignment)")
        axes[0].axis("off")
        axes[1].imshow(reg)
        if show_quiver:
            nr, nc = shape
            step = max(nr // nvec, nc // nvec)
            y, x = np.mgrid[:nr:step, :nc:step]
            u_s = u[::step, ::step]
            v_s = v[::step, ::step]
            mag = np.sqrt(u_s**2 + v_s**2)
            mag_norm = mag / (mag.max() + 1e-8)
            colors = cm.plasma(mag_norm)
            axes[1].quiver(
                x, y, u_s, v_s,
                color=colors.reshape(-1, 4),
                angles="xy", scale_units="xy", alpha=0.9
            )
            sm = plt.cm.ScalarMappable(
                cmap="plasma",
                norm=mcolors.Normalize(vmin=0, vmax=mag.max())
            )
            sm.set_array([])
            plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04, label="displacement (px)")
        axes[1].set_title("After registration  (fringing reduced)")
        axes[1].axis("off")
        plt.tight_layout()
        return fig

    @st.cache_data
    def build_error_fig(diff_before, diff_after):
        vmax = max(diff_before.max(), diff_after.max())
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].imshow(diff_before, cmap="inferno", vmin=0, vmax=vmax)
        axes[0].set_title("Error before registration")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(diff_after, cmap="inferno", vmin=0, vmax=vmax)
        axes[1].set_title("Error after registration")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig

    @st.cache_data
    def build_flow_fig(flow_magnitude, u, v, shape, show_quiver, nvec):
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(flow_magnitude, cmap="plasma")
        ax.set_title("Displacement magnitude (pixels)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="pixels")
        if show_quiver:
            nr, nc = shape
            step = max(nr // nvec, nc // nvec)
            y, x = np.mgrid[:nr:step, :nc:step]
            u_s = u[::step, ::step]
            v_s = v[::step, ::step]
            mag = np.sqrt(u_s**2 + v_s**2)
            mag_norm = mag / (mag.max() + 1e-8)
            colors = cm.plasma(mag_norm)
            ax.quiver(
                x, y, u_s, v_s,
                color=colors.reshape(-1, 4),
                angles="xy", scale_units="xy", alpha=0.9
            )
        plt.tight_layout()
        return fig

    # ----------------------------
    # Visualization
    # ----------------------------
    if st.session_state.flow_results:

        results = st.session_state.flow_results
        st.subheader(f"Results ({len(results)} frame pairs)")

        frame_idx = st.slider("Frame pair", 0, len(results) - 1, 0)
        r = results[frame_idx]

        # ── Alignment quality metrics ──────────────────────────────────
        st.subheader("Alignment Quality")
        st.caption(
            "NRMSE (lower = better) and SSIM (higher = better) measure how well "
            "the moving frame aligns with the reference before and after registration."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("NRMSE — before", f"{r['nrmse_before']:.4f}")
        col2.metric(
            "NRMSE — after",
            f"{r['nrmse_after']:.4f}",
            delta=f"{r['nrmse_after'] - r['nrmse_before']:.4f}",
            delta_color="inverse",
        )
        col3.metric("SSIM — before", f"{r['ssim_before']:.4f}")
        col4.metric(
            "SSIM — after",
            f"{r['ssim_after']:.4f}",
            delta=f"{r['ssim_after'] - r['ssim_before']:.4f}",
        )

        # ── Output frames ─────────────────────────────────────────────
        st.subheader("Output Frames")
        st.caption(
            "The **registered frame** is the warped output — the moving frame (t+1) "
            "resampled using the estimated displacement field so it aligns with the "
            "reference (t). Compare it with the raw moving frame to see how much the "
            "warp corrected the shift."
        )
        col1, col2, col3 = st.columns(3)
        col1.image(r["frame0"],   caption="Reference (t)",                  use_container_width=True)
        col2.image(r["frame1"],   caption="Moving (t+1) — unregistered",    use_container_width=True)
        col3.image(r["warp_rgb"], caption="Registered (t+1 → t) — warped output",
                    use_container_width=True, clamp=True)

        # ── Registration comparison ────────────────────────────────────
        st.subheader("Registration: Color-Bleed Overlay")
        st.caption(
            "**Red/cyan fringing = misalignment.** "
            "The reference is encoded in cyan (G+B channels) and the other frame in red (R channel). "
            "Fringing disappears when the two frames are well-aligned. "
            "Try looking at the optical flow vector field by checking 'Show Vector Field' on the sidebar! "
            "Vector colours encode displacement magnitude (blue = small, yellow = large). "
        )
        st.pyplot(build_overlay_fig(
            r["unreg_overlay"], r["reg_overlay"],
            r["u"], r["v"], r["shape"],
            show_quiver, nvec,
        ))

        # ── Before / after slider ──────────────────────────────────────
        st.subheader("Before / After Comparison")
        if HAS_IMAGE_COMPARISON:
            st.caption(
                "Drag the slider to compare the raw moving frame (left) against the "
                "registered output (right). Structures should snap into alignment with "
                "the reference as you cross the divider."
            )
            image_comparison(
                img1=r["frame1"],
                img2=r["warp_rgb"],
                label1="Moving — before registration",
                label2="Registered — after registration",
                width=700,
                show_labels=True,
            )
        else:
            st.caption(
                "Install `streamlit-image-comparison` for an interactive drag slider. "
                "Showing static side-by-side instead."
            )
            col1, col2 = st.columns(2)
            col1.image(r["frame1"],   caption="Moving — before registration",   use_container_width=True)
            col2.image(r["warp_rgb"], caption="Registered — after registration", use_container_width=True)

        # ── Residual error maps ────────────────────────────────────────
        st.subheader("Residual Error Maps")
        st.caption(
            "Pixel-wise absolute difference between the reference and the moving frame. "
            "Brighter = larger error. A well-registered pair should be uniformly dark."
        )
        st.pyplot(build_error_fig(r["diff_before"], r["diff_after"]))

        # ── Optical flow field ─────────────────────────────────────────
        st.subheader("Optical Flow — Displacement Field")
        st.caption(
            "Heatmap of per-pixel displacement magnitude. "
            "Vector colours (when enabled) match the same plasma scale: "
            "dark purple = near-zero motion, bright yellow = maximum displacement."
        )
        st.pyplot(build_flow_fig(
            r["flow_magnitude"], r["u"], r["v"], r["shape"],
            show_quiver, nvec,
        ))

    else:
        st.info("Configure settings and press 'Process Selected Range' to begin.")