import streamlit as st
from streamlit_image_comparison import image_comparison
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.metrics import mean_squared_error
import tempfile
import urllib.request
import cv2
from scipy.ndimage import map_coordinates


# ── App tab ────────────────────────────────────────────────────────────────────
def render():
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        algorithm = st.radio(
            "Algorithm:",
            ["iLK (Iterative Lukas-Kanade) - Fast", "TVL1 (Total Variation L1) - Detailed"],
            help="iLK is faster, TVL1 is slower but more detailed",
            index=0
        )

        if algorithm == "iLK (Iterative Lukas-Kanade) - Fast":
            radius = st.slider("Radius", 5, 25, 15, help=(
                "The algorithm assumes that all movement within a patch around each pixel will stay constant. Decreasing this will increase sensitivity to fine motion, but also increase noise."
            )
            )
            tvl1_attachment = 15.0
        else:
            radius = 15
            tvl1_attachment = st.slider(
                "Attachment",
                min_value=1.0,
                max_value=50.0,
                value=15.0,
                step=1.0,
                help=(
                    "Controls the trade-off between data fidelity and flow smoothness. "
                    "Lower → smoother, more regularised flow. "
                    "Higher → flow stays closer to raw brightness differences, capturing finer motion detail."
                ),
            )

        show_quiver = st.checkbox("Show vector field (the star of the show!!)", value=False)
        nvec = st.slider("Vectors per dimension", 10, 100, 60) if show_quiver else 60

        show_inter_frame = st.checkbox(
            "Show inter-frame steps",
            value=False,
            help=(
                "For stabilization purposes, the cumulative result is being used to warp the moving image back to t=0, but the optical flow is really working between every frame. Check this to view the true 'fixed' frame."
            ),
        )

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

    st.write("This app performs optical flow between every frame, then warps the final frame using the cumulative vectors to match the first frame (t=0). Choose an example video to try it out, or upload your own!")

    example_videos = {
        "MRI heartbeat": "https://github.com/sherylposting/data/raw/refs/heads/main/opticalflow/heartbeat.mp4",
        "X-ray dog swallowing": "https://github.com/sherylposting/data/raw/refs/heads/main/opticalflow/doggo.mp4",
        "Tardigrade riding a nematode": "https://github.com/sherylposting/data/raw/refs/heads/main/opticalflow/nematode.mp4",
        "Drifting volvox algae": "https://github.com/sherylposting/data/raw/refs/heads/main/opticalflow/volvox.mp4",
        "None (upload your own)": None,
    }

    selected_example = st.selectbox(
        "Choose an example video",
        list(example_videos.keys())
    )

    video_file = None

    if selected_example == "None (upload your own)":
        st.subheader("Upload Video")
        video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    # Resolve video_source to a local file path string
    video_source = None

    if video_file is not None:
        # Write uploaded bytes to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.getvalue())
            video_source = tmp.name

    elif selected_example != "None (upload your own)":
        # Download example video to a temp file (cached)
        video_source = download_video(example_videos[selected_example])

    if video_source is None:
        st.info("Select an example video or upload your own to begin.")
        st.stop()

    cap = cv2.VideoCapture(video_source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

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
    if "video_path" not in st.session_state:
        st.session_state.video_path = None

    # ----------------------------
    # Processing
    # ----------------------------
    if st.button("Process selected range", type="primary"):

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

        # ── Initialise cumulative flow (frame 0 → frame 0 = zero displacement) ──
        image_f = rgb2gray(frames[0])
        nr, nc = image_f.shape
        rr, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        u_cum = np.zeros((nr, nc), dtype=np.float64)
        v_cum = np.zeros((nr, nc), dtype=np.float64)

        for i in range(len(frames) - 1):

            progress_bar.progress((i + 1) / (len(frames) - 1))
            status.text(f"Processing pair {i + 1}/{len(frames) - 1}")

            image_f = rgb2gray(frames[i])
            image_g = rgb2gray(frames[i + 1])

            # ── Step 1: compute flow for the small consecutive step i -> i+1 ──
            if algorithm == "TVL1 (Total Variation L1) - Detailed":
                v_step, u_step = optical_flow_tvl1(image_f, image_g, attachment=tvl1_attachment)
            else:
                v_step, u_step = optical_flow_ilk(image_f, image_g, radius=radius)

            # ── Step 2: compose cumulative flow 0->i with step flow i->i+1 ──
            # For each pixel (r,c) anchored in frame 0, follow where the
            # cumulative warp already placed it (in frame i's coords), then
            # add the step flow sampled at that warped position.
            rr_i = rr + v_cum          # row-coords in frame i
            cc_i = cc + u_cum          # col-coords in frame i

            v_step_here = map_coordinates(v_step, [rr_i, cc_i], order=1, mode="nearest")
            u_step_here = map_coordinates(u_step, [rr_i, cc_i], order=1, mode="nearest")

            v_cum = v_cum + v_step_here
            u_cum = u_cum + u_step_here

            # Use cumulative field as (u, v) for display and metrics
            u, v = u_cum.copy(), v_cum.copy()

            # ── Step 3: warp frame i+1 back to frame 0 via cumulative field ──
            image_g_warp = np.clip(
                warp(image_g, np.array([rr + v, cc + u]), mode="edge"),
                0.0, 1.0
            )

            # ── Metrics always relative to t=0 ────────────────────────────
            frame0_gray = rgb2gray(frames[0])

            diff_before = np.abs(image_g - frame0_gray)
            diff_after  = np.abs(image_g_warp - frame0_gray)

            MSE_before = mean_squared_error(frame0_gray, image_g)
            MSE_after  = mean_squared_error(frame0_gray, image_g_warp)

            ncc_before = float(np.mean(cv2.matchTemplate(
                frame0_gray.astype(np.float32), image_g.astype(np.float32), cv2.TM_CCORR_NORMED)))
            ncc_after = float(np.mean(cv2.matchTemplate(
                frame0_gray.astype(np.float32), image_g_warp.astype(np.float32), cv2.TM_CCORR_NORMED)))

            def make_composite(ref, moving):
                comp = np.zeros((*ref.shape, 3))
                comp[..., 0] = moving
                comp[..., 1] = ref
                comp[..., 2] = ref
                return np.clip(comp, 0, 1)

            unreg_overlay = make_composite(frame0_gray, image_g)
            reg_overlay   = make_composite(frame0_gray, image_g_warp)

            # Advance previous frame for next iteration
            image_f = image_g

            # Convert grayscale float frames to uint8 RGB for display
            def gray_to_rgb_uint8(arr):
                ch = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                return np.stack([ch, ch, ch], axis=-1)

            warp_rgb    = gray_to_rgb_uint8(image_g_warp)
            frame0_rgb  = gray_to_rgb_uint8(rgb2gray(frames[0]))
            frame_f_rgb = gray_to_rgb_uint8(rgb2gray(frames[i]))
            frame_g_rgb = gray_to_rgb_uint8(rgb2gray(frames[i + 1]))

            results.append({
                "frame0":         frame0_rgb,
                "frame_f":        frame_f_rgb,
                "frame_g":        frame_g_rgb,
                "image_g_warp":    image_g_warp,
                "warp_rgb":       warp_rgb,
                "unreg_overlay":  unreg_overlay,
                "reg_overlay":    reg_overlay,
                "diff_before":    diff_before,
                "diff_after":     diff_after,
                "u":              u,
                "v":              v,
                "flow_magnitude": np.sqrt(u**2 + v**2),
                "shape":          (nr, nc),
                "MSE_before":   MSE_before,
                "MSE_after":    MSE_after,
                "ncc_before":    ncc_before,
                "ncc_after":     ncc_after,
            })

        st.session_state.flow_results = results
        st.session_state.video_path = None   # invalidate cached video
        progress_bar.empty()
        status.empty()

    # ----------------------------
    # Cached figure builders
    # ----------------------------
    @st.cache_data
    def build_overlay_fig(unreg, reg, u, v, shape, show_quiver, nvec):
        plt.style.use('dark_background')
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
        axes[1].set_title("After registration (fringing reduced)")
        axes[1].axis("off")
        plt.tight_layout()
        return fig

    @st.cache_data
    def build_error_fig(diff_before, diff_after):
        plt.style.use('dark_background')
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
        plt.style.use('dark_background')
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

    @st.cache_data
    def render_result_to_bgr(r, frame_idx, total, show_quiver, nvec, show_inter_frame):
        """Render one result dict into a BGR numpy image for the output video."""
        import io as _io

        if show_inter_frame:
            ref_img   = r["frame_f"]       # per-frame: advances alongside frame_g
            ref_label = f"Inter-frame, t={frame_idx}"
        else:
            ref_img   = r["frame0"]
            ref_label = "First frame, t=0"

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(18, 10), facecolor="black")
        fig.suptitle(
            f"Frame pair {frame_idx + 1} / {total}   |   "
            f"MSE {r['MSE_before']:.4f} → {r['MSE_after']:.4f}   |   "
            f"NCC {r['ncc_before']:.4f} → {r['ncc_after']:.4f}",
            color="white", fontsize=12, y=0.98,
        )

        gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.12)

        ax_ref = fig.add_subplot(gs[0, 0])
        ax_mov = fig.add_subplot(gs[0, 1])
        ax_reg = fig.add_subplot(gs[0, 2])

        ax_ref.imshow(ref_img)
        ax_ref.set_title(f"Fixed ({ref_label})", color="white", fontsize=10)
        ax_ref.axis("off")

        ax_mov.imshow(r["frame_g"])
        if show_quiver:
            nr, nc = r["shape"]
            step = max(nr // nvec, nc // nvec, 1)
            y, x = np.mgrid[:nr:step, :nc:step]
            u_s = r["u"][::step, ::step]
            v_s = r["v"][::step, ::step]
            mag = np.sqrt(u_s**2 + v_s**2)
            colors = cm.plasma(mag / (mag.max() + 1e-8))
            ax_mov.quiver(x, y, u_s, v_s,
                            color=colors.reshape(-1, 4),
                            angles="xy", scale_units="xy", alpha=0.9)
        ax_mov.set_title(f"Moving (t={frame_idx+1}) — unregistered", color="white", fontsize=10)
        ax_mov.axis("off")

        ax_reg.imshow(r["warp_rgb"])
        ax_reg.set_title(f"Registered (t → t=0)", color="white", fontsize=10)
        ax_reg.axis("off")

        # Render figure → BGR numpy array
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ----------------------------
    # Visualization
    # ----------------------------
    if st.session_state.flow_results:
        results = st.session_state.flow_results

        # ── Build / cache video ────────────────────────────────────────
        current_settings = (show_quiver, nvec, show_inter_frame)
        if (st.session_state.video_path is None
                or st.session_state.video_settings != current_settings):

            with st.spinner("Rendering output video…"):
                import subprocess

                # Render first frame to lock the output dimensions
                first_bgr = render_result_to_bgr(
                    results[0], 0, len(results), show_quiver, nvec,
                    show_inter_frame,
                )
                h, w = first_bgr.shape[:2]

                # Write raw intermediate with mp4v, then re-encode to H.264
                with tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4") as tmp_raw:
                    raw_path = tmp_raw.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                    out_path = tmp_out.name

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(raw_path, fourcc, 8.0, (w, h))
                writer.write(first_bgr)

                prog = st.progress(0.0)
                for i, r in enumerate(results[1:], start=1):
                    bgr = render_result_to_bgr(
                        r, i, len(results), show_quiver, nvec,
                        show_inter_frame,
                    )
                    writer.write(cv2.resize(bgr, (w, h)))
                    prog.progress(i / len(results))

                writer.release()
                prog.empty()

                # Re-encode to H.264 + yuv420p so all browsers can play it
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", raw_path,
                        "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "23",
                        out_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                st.session_state.video_path     = out_path
                st.session_state.video_settings = current_settings

        # ── Play video ─────────────────────────────────────────────────
        st.subheader("Optical Flow Animation")
        st.caption(
            "Each frame shows: **fixed (the target to match to), moving (unregistered frame), registered (t→t=0).** Plays at 8 fps. "
            "Try looking at the optical flow vector field by checking 'Show Vector Field' on the sidebar! "
            "Vector colours encode displacement magnitude (blue = small, yellow = large). "
        )
        with open(st.session_state.video_path, "rb") as vf:
            st.video(vf.read(), loop=True)

        st.header("Explore the results frame-by-frame")
        frame_idx = st.slider("Frame", 0, len(results) - 1, 0)
        r = results[frame_idx]

        # ── Reference image for display (toggle only affects the visual, not metrics) ──
        if show_inter_frame:
            ref = r["frame_f"]
            ref_label = f"Fixed (t={frame_idx})"
        else:
            ref = r["frame0"]
            ref_label = "t=0"

        # Metrics are always vs t=0
        diff_before = r["diff_before"]
        diff_after  = r["diff_after"]
        unreg_ov    = r["unreg_overlay"]
        reg_ov      = r["reg_overlay"]
        MSE_before  = r["MSE_before"]
        MSE_after   = r["MSE_after"]
        ncc_before  = r["ncc_before"]
        ncc_after   = r["ncc_after"]

        # ── Alignment quality metrics ──────────────────────────────────
        st.subheader("Alignment Quality")
        st.caption(
            "Mean Squared Error (MSE) (lower = better) and Normalized Cross-Correlation (NCC) (higher = better) measure how well the moving frame aligns with the reference before and after registration."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE — before (vs t=0)", f"{MSE_before:.4f}")
        col2.metric(
            "MSE — after (vs t=0)",
            f"{MSE_after:.4f}",
            delta=f"{MSE_after - MSE_before:.4f}",
            delta_color="inverse",
        )
        col3.metric("NCC — before (vs t=0)", f"{ncc_before:.4f}")
        col4.metric(
            "NCC — after (vs t=0)",
            f"{ncc_after:.4f}",
            delta=f"{ncc_after - ncc_before:.4f}",
        )
        # ── Before / after slider ──────────────────────────────────────
        st.subheader("Before / After Comparison")
        
        if "compare_original" not in st.session_state:
            st.session_state.compare_original = False
        st.checkbox("Compare to original reference", key="compare_original")

        st.caption(
            "Drag the slider to compare the raw moving frame (left) against the "
            "registered output (right). Structures should snap into alignment with "
            "the reference as you cross the divider."
        )

        if st.session_state.compare_original:
            image_comparison(
                img1=r["frame0"],
                img2=r["warp_rgb"],
                label1="Original frame at t=0 — reference to match to",
                label2="Registered — after registration",
                width=700,
                show_labels=True,
            )

        else:
            image_comparison(
                img1=r["frame_g"],
                img2=r["warp_rgb"],
                label1="Moving — before registration",
                label2="Registered — after registration",
                width=700,
                show_labels=True,
            )

        # ── Output frames ─────────────────────────────────────────────
        st.subheader("Output Frames")
        st.caption(
            "The **registered frame** is the warped output — the moving frame (t) "
            "resampled using the estimated displacement field so it aligns with the "
            f"reference ({ref_label}). Compare it with the raw moving frame to see how much the "
            "warp corrected the shift."
        )
        col1, col2, col3 = st.columns(3)
        col1.image(ref,              caption=ref_label,                        use_container_width=True)
        col2.image(r["frame_g"],    caption="Moving (t) — unregistered",      use_container_width=True)
        col3.image(r["warp_rgb"],  caption=f"Registered (t → t=0) — warped output",
                    use_container_width=True, clamp=True)

        # ── Registration comparison ────────────────────────────────────
        st.subheader("Registration: Color-Bleed Overlay")
        st.caption(
            "**Red/cyan fringing = misalignment.** "
            "The reference is encoded in cyan (G+B channels) and the other frame in red (R channel). "
            "Fringing disappears when the two frames are well-aligned. "
        )
        st.pyplot(build_overlay_fig(
            unreg_ov, reg_ov,
            r["u"], r["v"], r["shape"],
            show_quiver, nvec,
        ))

        # ── Residual error maps ────────────────────────────────────────
        st.subheader("Residual Error Maps")
        st.caption(
            "Pixel-wise absolute difference between the reference and the moving frame. "
            "Brighter = larger error. A well-registered pair should be uniformly dark."
        )
        st.pyplot(build_error_fig(diff_before, diff_after))

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