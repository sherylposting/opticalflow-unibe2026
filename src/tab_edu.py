import streamlit as st

# ── Educational tab ────────────────────────────────────────────────────────────
def render():
    st.header("What is image registration?")
    st.write(
        "**Image registration** is the process of mapping a target (or 'moving') image to a reference (or 'fixed') image, such that the corresponding structures in the images are aligned to each other. It is an optimization problem where we must find a geometric transformation which minimizes the displacement between the two images."
    )
    st.write(
        "Suppose you have two frames from a video. Between them, something moved, and you want to track its trajectory. For each pixel in frame A, registration attempts to guess where it ended up in frame B. Once you know that, you can warp frame B back onto frame A to align the two frames. " \
    )
    st.write(
        "While many uses of image registration include stabilizing accidental movements, or aligning multiple imaging modalities (ex. a fluorescent marker overlaid on MRI), **registration can also allow us to track the movement of objects**."
    )

    st.subheader("Linear image registration")
    st.write(
        "**Linear image registration** aligns two images using matrix transformations, such as translation, rotation, scaling, and shearing. Linear registration is effective when differences between images can be approximated by global motion, but is less effective for local deformations, such as what you would expect in a moving tissue or organism. " \
    )
    st.write("One common linear registration approach is Iterative Close Points (ICP), which minimizes the squared distance (L2 norm) between each landmark point in the moving image to its current closest point in the reference:"
    )
    st.markdown(r"""
    $$
    (R,T)= \underset{R,T}{\operatorname{argmin}} \frac{1}{N}\sum_{i=1}^N{||Rx_i+T-y_i||^2}
    $$
    """, unsafe_allow_html=True)
    st.image("https://github.com/sherylposting/data/blob/main/opticalflow/icp.gif?raw=true", width=300)

    st.header("What optical flow computes")
    st.write(
        "**Optical flow** is an example of a **non-linear image registration** method, which can apply an elastic transformation across the image instead of being limited to linear transformations. It estimates a vector field **u(x,y), v(x,y)**, with one 2D displacement vector per pixel. The vector at each location says: the content here moved Δx pixels horizontally and Δy pixels vertically. Together, these vectors form the displacement field or warp field. Typically, when people use optical flow, they only care about retrieving this vector field for motion-tracking purposes, but it can also be used for stabilization (like this app does)."
    )
    st.markdown("""
        In order to work, optical flow makes some important assumptions (which may limit its performance if you don't accommodate them!):
        1. Assumes that a moving pixel's intensity from frame A to frame B remains similar.
        2. Assumes that inter-frame motion is small (1-2 px). This means that optical flow only really works from frame-to-frame, and is weak for fast-moving objects.
    """)
    st.write(
        "This is the central problem that optical flow tries to solve:"
    )

    st.latex(r'''
    \nabla g \cdot \vec{u}+g-f=0
    ''')
    st.markdown(r"""
    *Uses the image gradient $\nabla g$ and the optimal displacement vector $\vec{u}$ to transform the moving image $g$ back to the fixed image $f$. This is an underdetermined system (since we are solving for an x and y component with only one equation), and requires some regularization or inclusion of more data.*
    """, unsafe_allow_html=True)

    st.header("The app and its algorithms, explained")
    st.markdown("This app uses the `optical_flow_ilk` and `optical_flow_tvl1` functions, which are part of the scikit-image module `skimage.registration`. Summary statistics were calculated using numpy (MSE) and OpenCV (NCC), and plots were constructed using matplotlib.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**iLK (Iterative Lucas-Kanade)**")
        st.write(
            "Assumes brightness is locally constant within a window of radius *r* "
            "around each pixel, then solves a least-squares system for the local "
            "velocity. Iterating and coarsening the pyramid extends its range. "
            "Struggles with large displacements.$^1$"
        )
        st.latex(r'''
        \sum_{x,y\in\Omega}W^2(x,y)\ (\nabla g \cdot \vec{u}+g-f)^2=0 \\
        ''')
        st.markdown(r"""
        *Assumes that all displacement vectors $\ u_x$, $u_y$ will remain constant within a local neighborhood $\ \Omega$ around each pixel. $W^2$ is a weighting function that gives greater influence to the pixels at the center of the neighborhood.*
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**TVL1 (Total Variation L1)**")
        st.write(
            "Based on minimizing an energy function. Includes an L1 data term that measures brightness consistency, plus a Total Variation (TV) regulariser which smooths and penalizes large spatial changes in the field. The L1 norm is more robust against outliers and illumination variations than the L2 norm, making it better for real-world conditions. The TV term keeps the field spatially smooth while still allowing sharp motion boundaries.$^2$"
        )
        st.latex(r'''
            E = \int_{\Omega} \left\{ \lambda \left| I_0(x) - I_1(x + u(x)) \right| + |\nabla u| \right\} \, dx
            ''')
        st.markdown(r"""
        *$I_0$ and $I_1$ are two consecutive images. $|I_1(x+u, y+v) - I_0(x,y)|$ is the L1 data term, and $|\nabla u| + |\nabla v|$ is the TV regularization. $\lambda$ is the attachment parameter, and controls the smoothness strength.*
        """, unsafe_allow_html=True)
    
    st.write(
        "This app uses optical flow to estimate the movement between each frame. It adds up the cumulative vector field, then warps the moving frame to match the first frame (t=0). Although the final warp is ultimately trying to match the first frame, the algorithm really performs the registration between every frame."
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
            "The fixed frame is placed in the **green and blue channels** (cyan), "
            "the moving frame in the **red channel**. Where the two frames disagree spatially "
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
            "using a cool→warm colormap (purple = small displacement, yellow = large). "
            "Length encodes direction and relative speed. "
        )

    with st.expander("MSE and NCC"):
        st.write(
            "**MSE** (Mean Squared Error) measures average pixel-level "
            "disagreement — lower is better. "
            "**NCC** (Normalized Cross-Correlation) is a template-matching score which broadly calculates, for each pixel, whether the mean intensity is similar to the original — higher is better."
        )

    st.subheader("Common applications")
    st.write(
        "Optical flow registration can be used for many purposes: aligning consecutive "
        "microscopy frames to correct stage drift, tracking moving objects in a video, "
        "correcting respiratory motion in medical imaging, "
        "and computing motion-compensated video compression."
    )

    st.header("References")
    st.markdown(r"""
    1. Tektonidis, M. (2018). Non-rigid multi-frame registration of cell nuclei in live cell microscopy image data (Doctoral dissertation, Heidelberg University). Heidelberg University Library. https://doi.org/10.11588/heidok.00025518
    2. Wedel, A., Pock, T., Zach, C., Cremers, D., & Bischof, H. (2008). An improved algorithm for TV-L1 optical flow. In Proceedings of the Dagstuhl Motion Workshop. Springer.
    3. Balaha, H. M., Mahmoud, A., Abou El-Ghar, M., Ghazal, M., Contractor, S., & El-Baz, A. (2025). H-UDMIR: A hybrid unsupervised deep learning framework for deformable medical image registration. IEEE Access, 13, 69705–69722. https://doi.org/10.1109/ACCESS.2025.3562092
    4. Computer Vision Group. (2025). HS2025: Computer Vision \[Lecture slides]. University of Bern. Retrieved May 2026, from ILIAS Universität Bern
    5. ARTORG Center for Biomedical Engineering Research. (2026). FS2026: Introduction to Image Analysis \[Lecture slides]. University of Bern. Retrieved May 2026, from ILIAS Universität Bern
    """, unsafe_allow_html=True)
    st.subheader("Example videos")
    st.markdown(r"""
        All videos were downloaded from Youtube using yt-dlp, a CLI tool.
        * MRI heartbeat (McMaster University): https://www.youtube.com/shorts/BL_Yn3r4Qsk
        * X-ray dog swallowing: https://www.youtube.com/watch?v=rex_N_H4zxU
        * Tardigrade riding a nematode (Nikon Small World 2024): https://www.youtube.com/watch?v=R9P-oMfSlGU
        * Drifting volvox algae (Nikon Small World 2025): https://www.youtube.com/watch?v=yjbK_GVGTPg
    """, unsafe_allow_html=True)