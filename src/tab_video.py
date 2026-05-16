import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.metrics import mean_squared_error
import tempfile
import urllib.request
import cv2


# ── Video tab ────────────────────────────────────────────────────────────────────
def render():
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        algorithm = st.radio(
            "Algorithm:",
            ["iLK (Iterative Lukas-Kanade) - Fast", "TVL1 (Total Variation L1) - Detailed"],
            help="iLK is faster, TVL1 is slower but more detailed",
            index=0,
            key="tab_app_algorithm",
        )

        if algorithm == "iLK (Iterative Lukas-Kanade) - Fast":
            radius = st.slider("Radius", 5, 25, 15)
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
    if "video_settings" not in st.session_state:
        st.session_state.video_settings = None

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

            if algorithm == "TVL1 (Total Variation L1) - Detailed":
                v, u = optical_flow_tvl1(image0, image1, attachment=tvl1_attachment)
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

            MSE_before = mean_squared_error(image0, image1)
            MSE_after  = mean_squared_error(image0, image1_warp)

            ncc_before = np.mean(cv2.matchTemplate(image0.astype(np.float32), image1.astype(np.float32), cv2.TM_CCORR_NORMED))
            ncc_after = np.mean(cv2.matchTemplate(image0.astype(np.float32), image1_warp.astype(np.float32), cv2.TM_CCORR_NORMED))

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
                "MSE_before":   MSE_before,
                "MSE_after":    MSE_after,
                "ncc_before":    ncc_before,
                "ncc_after":     ncc_after,
            })

        st.session_state.flow_results = results
        st.session_state.video_path = None   # invalidate cached video
        st.session_state.video_settings = None
        progress_bar.empty()
        status.empty()
        st.success(f"✅ Processed {len(results)} frame pairs")

    # ----------------------------
    # Video frame renderer
    # ----------------------------
    def render_result_to_bgr(r, frame_idx, total, show_quiver, nvec):
        """Render one result dict into a BGR numpy image for the output video."""
        import io as _io

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(18, 10), facecolor="black")
        fig.suptitle(
            f"Frame pair {frame_idx + 1} / {total}   |   "
            f"MSE {r['MSE_before']:.4f} → {r['MSE_after']:.4f}   |   "
            f"NCC {r['ncc_before']:.4f} → {r['ncc_after']:.4f}",
            color="white", fontsize=12, y=0.98,
        )

        gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.12)

        # ── Top row: reference | moving | registered ───────────────────
        ax_mov = fig.add_subplot(gs[0, 0])
        ax_reg = fig.add_subplot(gs[0, 1])

        ax_mov.imshow(r["frame1"], cmap="gray")
        ax_mov.set_title("Moving (t+1) — unregistered", color="white", fontsize=10)
        ax_mov.axis("off")

        ax_reg.imshow(r["warp_rgb"])
        ax_reg.set_title("Registered (t+1 → t)", color="white", fontsize=10)
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
        st.subheader(f"Results — {len(results)} frame pairs")

        # ── Aggregate alignment metrics ────────────────────────────────
        st.subheader("Alignment Quality (mean across all frames)")
        st.caption(
            "Mean Squared Error (MSE) (lower = better) and Normalized Cross-Correlation "
            "(NCC) (higher = better) measure how well the moving frame aligns with the "
            "reference before and after registration."
        )
        avg_mse_b = np.mean([r["MSE_before"] for r in results])
        avg_mse_a = np.mean([r["MSE_after"]  for r in results])
        avg_ncc_b = np.mean([r["ncc_before"] for r in results])
        avg_ncc_a = np.mean([r["ncc_after"]  for r in results])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE — before", f"{avg_mse_b:.4f}")
        col2.metric("MSE — after",  f"{avg_mse_a:.4f}",
                    delta=f"{avg_mse_a - avg_mse_b:.4f}", delta_color="inverse")
        col3.metric("NCC — before", f"{avg_ncc_b:.4f}")
        col4.metric("NCC — after",  f"{avg_ncc_a:.4f}",
                    delta=f"{avg_ncc_a - avg_ncc_b:.4f}")

        # ── Build / cache video ────────────────────────────────────────
        current_settings = (show_quiver, nvec)
        if (st.session_state.video_path is None
                or st.session_state.video_settings != current_settings):

            with st.spinner("Rendering output video…"):
                import subprocess

                # Render first frame to lock the output dimensions
                first_bgr = render_result_to_bgr(
                    results[0], 0, len(results), show_quiver, nvec
                )
                h, w = first_bgr.shape[:2]

                # Write raw intermediate with mp4v, then re-encode to H.264
                with tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4") as tmp_raw:
                    raw_path = tmp_raw.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                    out_path = tmp_out.name

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(raw_path, fourcc, 10.0, (w, h))
                writer.write(first_bgr)

                prog = st.progress(0.0)
                for i, r in enumerate(results[1:], start=1):
                    bgr = render_result_to_bgr(r, i, len(results), show_quiver, nvec)
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
            "Each frame shows: **reference · moving · registered** (top row) and "
            "**before overlay · after overlay · displacement field** (bottom row). "
            "Plays at 2 fps so motion is easy to follow."
        )
        with open(st.session_state.video_path, "rb") as vf:
            st.video(vf.read(), loop=True)

    else:
        st.info("Configure settings and press '🎬 Process Selected Range' to begin.")