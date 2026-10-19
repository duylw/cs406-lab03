"""Minimalist Image Enhancing Application (Lab-03).

Simplified UI layout inspired by course lecture slides:
- Direct top-level image uploader (no sidebar/drawer).
- Clean horizontal side-by-side comparison grids.
- Centered image headers and minimal visual clutter.
"""

import streamlit as st
import numpy as np

from core import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_mean_filter,
    apply_median_filter,
    apply_laplacian_sharpen,
    apply_unsharp_mask,
    apply_sobel_edge,
    apply_prewitt_edge,
    apply_canny_edge,
    calculate_psnr,
)
from core.utils import (
    load_image_from_bytes,
    generate_sample_color_image,
)

# Page configuration
st.set_page_config(
    page_title="Image Enhancing | Lab-03",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal CSS styling
st.markdown(
    """
    <style>
    /* Hide Streamlit default sidebar toggle if unused */
    [data-testid="stSidebar"] {
        display: none;
    }
    .img-title {
        text-align: center;
        font-weight: 600;
        font-size: 1.05rem;
        color: #1E293B;
        margin-bottom: 8px;
    }
    .img-caption {
        text-align: center;
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 6px;
    }
    .output-label {
        text-align: center;
        font-weight: 500;
        color: #475569;
        font-size: 1.0rem;
        margin-top: 14px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_image() -> np.ndarray:
    """Load image directly from main page uploader or fallback to default sample."""
    st.subheader("Lab-03: Image Enhancing")

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload image (PNG, JPG, BMP) or leave blank for default sample",
            type=["png", "jpg", "jpeg", "bmp"],
            help="Upload an image from your computer.",
        )

    if uploaded_file is not None:
        try:
            image_rgb = load_image_from_bytes(uploaded_file.getvalue())
            source_text = f"Uploaded: {uploaded_file.name}"
        except Exception as err:
            st.error(f"Failed to load image: {err}")
            image_rgb = generate_sample_color_image(512, 512)
            source_text = "Default Sample (Fallback)"
    else:
        image_rgb = generate_sample_color_image(512, 512)
        source_text = "Default Sample (512 x 512)"

    with col_info:
        h, w = image_rgb.shape[:2]
        st.markdown(
            f"""
            <div style="font-size: 0.9rem; color: #64748B; padding-top: 25px;">
                <b>Source:</b> {source_text}<br>
                <b>Resolution:</b> {w} x {h} px
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    return image_rgb


def show_denoising_row(image_rgb: np.ndarray, key_prefix: str = "denoise") -> None:
    """Render horizontal Denoising row matching slide layout."""
    # Slim parameter controls directly above
    c_n1, c_n2, c_k1, c_k2 = st.columns([1.2, 1.2, 1, 1])
    with c_n1:
        noise_type = st.selectbox(
            "Noise Type",
            options=["Gaussian Noise", "Salt & Pepper Noise"],
            index=0,
            key=f"{key_prefix}_noise_type",
        )
    with c_n2:
        if noise_type == "Gaussian Noise":
            sigma = st.slider(
                "Noise Intensity (Sigma)", 5, 80, 25, 5, key=f"{key_prefix}_sigma"
            )
        else:
            amount = st.slider(
                "Noise Density", 0.01, 0.25, 0.06, 0.01, key=f"{key_prefix}_amount"
            )
    with c_k1:
        mean_k = st.select_slider(
            "Mean Kernel", options=[3, 5, 7, 9], value=3, key=f"{key_prefix}_mean_k"
        )
    with c_k2:
        median_k = st.select_slider(
            "Median Kernel", options=[3, 5, 7, 9], value=3, key=f"{key_prefix}_median_k"
        )

    # Process images
    if noise_type == "Gaussian Noise":
        noisy_img = add_gaussian_noise(image_rgb, sigma=float(sigma), seed=42)
        noisy_title = "Noisy (Gaussian)"
    else:
        noisy_img = add_salt_and_pepper_noise(image_rgb, amount=float(amount), seed=42)
        noisy_title = "Noisy (Salt & Pepper)"

    mean_img = apply_mean_filter(noisy_img, kernel_size=mean_k)
    median_img = apply_median_filter(noisy_img, kernel_size=median_k)

    psnr_mean = calculate_psnr(image_rgb, mean_img)
    psnr_median = calculate_psnr(image_rgb, median_img)

    st.write("")
    # 4 Columns side-by-side matching the reference image layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="img-title">Original</div>', unsafe_allow_html=True)
        st.image(image_rgb, width="stretch")

    with col2:
        st.markdown(f'<div class="img-title">{noisy_title}</div>', unsafe_allow_html=True)
        st.image(noisy_img, width="stretch")

    with col3:
        st.markdown('<div class="img-title">Denoised (Mean)</div>', unsafe_allow_html=True)
        st.image(mean_img, width="stretch")
        st.markdown(f'<div class="img-caption">PSNR: {psnr_mean:.2f} dB</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="img-title">Denoised (Median)</div>', unsafe_allow_html=True)
        st.image(median_img, width="stretch")
        st.markdown(f'<div class="img-caption">PSNR: {psnr_median:.2f} dB</div>', unsafe_allow_html=True)

    st.markdown('<div class="output-label">output</div>', unsafe_allow_html=True)


def show_edge_row(image_rgb: np.ndarray, key_prefix: str = "edge") -> None:
    """Render horizontal Edge Detection row with Sobel, Prewitt, and Canny."""
    # Slim parameter controls directly above
    c_s, c_cl, c_ch = st.columns([1, 1, 1])
    with c_s:
        sobel_k = st.selectbox(
            "Sobel Aperture (ksize)", options=[1, 3, 5, 7], index=1, key=f"{key_prefix}_sobel_k"
        )
    with c_cl:
        canny_low = st.slider(
            "Canny Low Threshold", 0, 200, 50, 10, key=f"{key_prefix}_canny_low"
        )
    with c_ch:
        canny_high = st.slider(
            "Canny High Threshold", canny_low, 300, 150, 10, key=f"{key_prefix}_canny_high"
        )

    # Process all 3 edge operators
    sobel_img = apply_sobel_edge(image_rgb, ksize=sobel_k)
    prewitt_img = apply_prewitt_edge(image_rgb)
    canny_img = apply_canny_edge(image_rgb, low_threshold=canny_low, high_threshold=canny_high)

    st.write("")
    # 4 Columns side-by-side
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="img-title">Original</div>', unsafe_allow_html=True)
        st.image(image_rgb, width="stretch")

    with col2:
        st.markdown('<div class="img-title">Sobel Edge</div>', unsafe_allow_html=True)
        st.image(sobel_img, clamp=True, width="stretch")

    with col3:
        st.markdown('<div class="img-title">Prewitt Edge</div>', unsafe_allow_html=True)
        st.image(prewitt_img, clamp=True, width="stretch")

    with col4:
        st.markdown('<div class="img-title">Canny Edge</div>', unsafe_allow_html=True)
        st.image(canny_img, clamp=True, width="stretch")

    st.markdown('<div class="output-label">output</div>', unsafe_allow_html=True)


def show_sharpening_row(image_rgb: np.ndarray, key_prefix: str = "sharp") -> None:
    """Render horizontal Sharpening row comparing Laplacian and Unsharp Masking."""
    # Slim controls directly above
    c_alpha, c_boost = st.columns(2)
    with c_alpha:
        laplacian_alpha = st.slider(
            "Laplacian Alpha (Strength)", 0.2, 3.0, 1.0, 0.2, key=f"{key_prefix}_laplacian_alpha"
        )
    with c_boost:
        unsharp_strength = st.slider(
            "Unsharp Mask Strength", 0.5, 4.0, 1.8, 0.2, key=f"{key_prefix}_unsharp_strength"
        )

    # Process sharpening
    sharp_laplacian = apply_laplacian_sharpen(image_rgb, alpha=laplacian_alpha)
    sharp_unsharp = apply_unsharp_mask(image_rgb, sigma=1.5, strength=unsharp_strength)

    st.write("")
    # 3 Columns side-by-side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="img-title">Original</div>', unsafe_allow_html=True)
        st.image(image_rgb, width="stretch")

    with col2:
        st.markdown('<div class="img-title">Sharpened (Laplacian)</div>', unsafe_allow_html=True)
        st.image(sharp_laplacian, width="stretch")

    with col3:
        st.markdown('<div class="img-title">Sharpened (Unsharp Mask)</div>', unsafe_allow_html=True)
        st.image(sharp_unsharp, width="stretch")

    st.markdown('<div class="output-label">output</div>', unsafe_allow_html=True)


def show_all_combined_rows(image_rgb: np.ndarray) -> None:
    """Render static consolidated rows of all techniques without duplicate widget controls."""
    # 1. Denoising
    noisy_img = add_gaussian_noise(image_rgb, sigma=25.0, seed=42)
    mean_img = apply_mean_filter(noisy_img, kernel_size=3)
    median_img = apply_median_filter(noisy_img, kernel_size=3)
    psnr_mean = calculate_psnr(image_rgb, mean_img)
    psnr_median = calculate_psnr(image_rgb, median_img)

    st.markdown("#### 1. Denoising (Gaussian Noise, 3x3 Kernel)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="img-title">Original</div>', unsafe_allow_html=True)
        st.image(image_rgb, width="stretch")
    with c2:
        st.markdown('<div class="img-title">Noisy (Gaussian)</div>', unsafe_allow_html=True)
        st.image(noisy_img, width="stretch")
    with c3:
        st.markdown('<div class="img-title">Denoised (Mean)</div>', unsafe_allow_html=True)
        st.image(mean_img, width="stretch")
        st.markdown(f'<div class="img-caption">PSNR: {psnr_mean:.2f} dB</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="img-title">Denoised (Median)</div>', unsafe_allow_html=True)
        st.image(median_img, width="stretch")
        st.markdown(f'<div class="img-caption">PSNR: {psnr_median:.2f} dB</div>', unsafe_allow_html=True)
    st.markdown('<div class="output-label">output</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2. Edge Detection
    sobel_img = apply_sobel_edge(image_rgb, ksize=3)
    prewitt_img = apply_prewitt_edge(image_rgb)
    canny_img = apply_canny_edge(image_rgb, low_threshold=50, high_threshold=150)

    st.markdown("#### 2. Edge Detection (Sobel, Prewitt, Canny)")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown('<div class="img-title">Original</div>', unsafe_allow_html=True)
        st.image(image_rgb, width="stretch")
    with e2:
        st.markdown('<div class="img-title">Sobel Edge</div>', unsafe_allow_html=True)
        st.image(sobel_img, clamp=True, width="stretch")
    with e3:
        st.markdown('<div class="img-title">Prewitt Edge</div>', unsafe_allow_html=True)
        st.image(prewitt_img, clamp=True, width="stretch")
    with e4:
        st.markdown('<div class="img-title">Canny Edge</div>', unsafe_allow_html=True)
        st.image(canny_img, clamp=True, width="stretch")
    st.markdown('<div class="output-label">output</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 3. Sharpening
    sharp_laplacian = apply_laplacian_sharpen(image_rgb, alpha=1.0)
    sharp_unsharp = apply_unsharp_mask(image_rgb, sigma=1.5, strength=1.8)

    st.markdown("#### 3. Sharpening (Laplacian & Unsharp Mask)")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown('<div class="img-title">Original</div>', unsafe_allow_html=True)
        st.image(image_rgb, width="stretch")
    with s2:
        st.markdown('<div class="img-title">Sharpened (Laplacian)</div>', unsafe_allow_html=True)
        st.image(sharp_laplacian, width="stretch")
    with s3:
        st.markdown('<div class="img-title">Sharpened (Unsharp Mask)</div>', unsafe_allow_html=True)
        st.image(sharp_unsharp, width="stretch")
    st.markdown('<div class="output-label">output</div>', unsafe_allow_html=True)


def main() -> None:
    image_rgb = get_image()

    tab_denoise, tab_edge, tab_sharpen, tab_all = st.tabs(
        [
            "Denoising (Mean & Median)",
            "Edge Detection (Sobel, Prewitt, Canny)",
            "Sharpening (Laplacian & Unsharp Mask)",
            "All Combined",
        ]
    )

    with tab_denoise:
        show_denoising_row(image_rgb, key_prefix="tab_denoise")

    with tab_edge:
        show_edge_row(image_rgb, key_prefix="tab_edge")

    with tab_sharpen:
        show_sharpening_row(image_rgb, key_prefix="tab_sharpen")

    with tab_all:
        show_all_combined_rows(image_rgb)


if __name__ == "__main__":
    main()
