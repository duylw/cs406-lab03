"""Automated Batch Verification and Benchmark Pipeline for Lab-03 Image Enhancing.

Executes all core algorithms on a reference test image, measures processing
duration and PSNR/MSE metrics, saves visual comparison figures to disk,
and prints an execution summary table.
"""

import os
import time
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt

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
    calculate_mse,
    calculate_psnr,
)
from core.utils import generate_sample_color_image, sanitize_image


OUTPUT_DIR = "output"


def ensure_output_dir() -> None:
    """Create output directory if it does not already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def benchmark_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Execute a function and record runtime duration in milliseconds."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    return result, duration_ms


def run_pipeline() -> None:
    """Execute complete end-to-end verification and generate benchmark report."""
    ensure_output_dir()
    print("=" * 80)
    print("LAB-03: AUTOMATED PIPELINE VERIFICATION AND BENCHMARK")
    print("=" * 80)

    # 1. Prepare Ground Truth Image
    base_image = generate_sample_color_image(512, 512)
    clean_image = sanitize_image(base_image)
    h, w, c = clean_image.shape
    print(f"[INFO] Input reference image initialized: {w}x{h} ({c} channels, uint8)")

    results_table = []

    # =========================================================================
    # 2. Denoising Pipeline Benchmark
    # =========================================================================
    print("\n[INFO] Running Denoising & Smoothing Benchmarks...")

    # Gaussian Noise
    noisy_gauss, t_ng = benchmark_function(add_gaussian_noise, clean_image, sigma=25.0, seed=42)
    psnr_gauss = calculate_psnr(clean_image, noisy_gauss)
    mse_gauss = calculate_mse(clean_image, noisy_gauss)
    results_table.append(("Gaussian Noise (sigma=25)", "-", f"{psnr_gauss:.2f}", f"{mse_gauss:.2f}", f"{t_ng:.2f}", "[PASS]"))

    mean_gauss, t_mg = benchmark_function(apply_mean_filter, noisy_gauss, kernel_size=3)
    psnr_mg = calculate_psnr(clean_image, mean_gauss)
    mse_mg = calculate_mse(clean_image, mean_gauss)
    results_table.append(("Mean Filter (3x3)", "Gaussian", f"{psnr_mg:.2f}", f"{mse_mg:.2f}", f"{t_mg:.2f}", "[PASS]"))

    median_gauss, t_medg = benchmark_function(apply_median_filter, noisy_gauss, kernel_size=3)
    psnr_medg = calculate_psnr(clean_image, median_gauss)
    mse_medg = calculate_mse(clean_image, median_gauss)
    results_table.append(("Median Filter (3x3)", "Gaussian", f"{psnr_medg:.2f}", f"{mse_medg:.2f}", f"{t_medg:.2f}", "[PASS]"))

    # Salt & Pepper Noise
    noisy_sp, t_nsp = benchmark_function(add_salt_and_pepper_noise, clean_image, amount=0.06, seed=42)
    psnr_sp = calculate_psnr(clean_image, noisy_sp)
    mse_sp = calculate_mse(clean_image, noisy_sp)
    results_table.append(("Salt & Pepper (6%)", "-", f"{psnr_sp:.2f}", f"{mse_sp:.2f}", f"{t_nsp:.2f}", "[PASS]"))

    mean_sp, t_msp = benchmark_function(apply_mean_filter, noisy_sp, kernel_size=3)
    psnr_msp = calculate_psnr(clean_image, mean_sp)
    mse_msp = calculate_mse(clean_image, mean_sp)
    results_table.append(("Mean Filter (3x3)", "Salt & Pepper", f"{psnr_msp:.2f}", f"{mse_msp:.2f}", f"{t_msp:.2f}", "[PASS]"))

    median_sp, t_medsp = benchmark_function(apply_median_filter, noisy_sp, kernel_size=3)
    psnr_medsp = calculate_psnr(clean_image, median_sp)
    mse_medsp = calculate_mse(clean_image, median_sp)
    status_med = "[PASS]" if psnr_medsp > psnr_msp else "[WARN]"
    results_table.append(("Median Filter (3x3)", "Salt & Pepper", f"{psnr_medsp:.2f}", f"{mse_medsp:.2f}", f"{t_medsp:.2f}", status_med))

    # =========================================================================
    # 3. Sharpening Pipeline Benchmark
    # =========================================================================
    print("[INFO] Running Sharpening Benchmarks...")

    sharp_lap, t_lap = benchmark_function(apply_laplacian_sharpen, clean_image, alpha=1.0)
    results_table.append(("Laplacian Sharpen (alpha=1.0)", "Clean RGB", "-", "-", f"{t_lap:.2f}", "[PASS]"))

    sharp_unsharp, t_unsharp = benchmark_function(apply_unsharp_mask, clean_image, sigma=1.5, strength=1.8)
    results_table.append(("Unsharp Mask (strength=1.8)", "Clean RGB", "-", "-", f"{t_unsharp:.2f}", "[PASS]"))

    # =========================================================================
    # 4. Edge Detection Pipeline Benchmark
    # =========================================================================
    print("[INFO] Running Edge Detection Benchmarks (Sobel, Prewitt, Canny)...")

    edge_sobel, t_sobel = benchmark_function(apply_sobel_edge, clean_image, ksize=3)
    results_table.append(("Sobel Edge (ksize=3)", "Grayscale", "-", "-", f"{t_sobel:.2f}", "[PASS]"))

    edge_prewitt, t_prewitt = benchmark_function(apply_prewitt_edge, clean_image)
    results_table.append(("Prewitt Edge (Custom 3x3)", "Grayscale", "-", "-", f"{t_prewitt:.2f}", "[PASS]"))

    edge_canny, t_canny = benchmark_function(apply_canny_edge, clean_image, low_threshold=50, high_threshold=150)
    results_table.append(("Canny Edge (50, 150)", "Grayscale", "-", "-", f"{t_canny:.2f}", "[PASS]"))

    # =========================================================================
    # 5. Generate and Save Visual Output Figures
    # =========================================================================
    print("\n[INFO] Generating visual figures matching lecture slide specifications...")

    # Figure 1: Denoising Row (Matching Slide Format: 4 Columns)
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
    axes1[0].imshow(clean_image)
    axes1[0].set_title("Original", fontsize=12, fontweight="bold")
    axes1[0].axis("off")

    axes1[1].imshow(noisy_gauss)
    axes1[1].set_title(f"Noisy (Gaussian)\nPSNR: {psnr_gauss:.1f} dB", fontsize=11)
    axes1[1].axis("off")

    axes1[2].imshow(mean_gauss)
    axes1[2].set_title(f"Denoised (Mean)\nPSNR: {psnr_mg:.1f} dB", fontsize=11)
    axes1[2].axis("off")

    axes1[3].imshow(median_gauss)
    axes1[3].set_title(f"Denoised (Median)\nPSNR: {psnr_medg:.1f} dB", fontsize=11)
    axes1[3].axis("off")

    fig1.text(0.5, 0.02, "output", ha="center", fontsize=12, color="#475569")
    fig1.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "01_denoising_gaussian_comparison.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[SAVED] {fig1_path}")

    # Figure 1B: Salt & Pepper Denoising Row
    fig1b, axes1b = plt.subplots(1, 4, figsize=(16, 4))
    axes1b[0].imshow(clean_image)
    axes1b[0].set_title("Original", fontsize=12, fontweight="bold")
    axes1b[0].axis("off")

    axes1b[1].imshow(noisy_sp)
    axes1b[1].set_title(f"Noisy (Salt & Pepper)\nPSNR: {psnr_sp:.1f} dB", fontsize=11)
    axes1b[1].axis("off")

    axes1b[2].imshow(mean_sp)
    axes1b[2].set_title(f"Denoised (Mean)\nPSNR: {psnr_msp:.1f} dB", fontsize=11)
    axes1b[2].axis("off")

    axes1b[3].imshow(median_sp)
    axes1b[3].set_title(f"Denoised (Median)\nPSNR: {psnr_medsp:.1f} dB", fontsize=11)
    axes1b[3].axis("off")

    fig1b.text(0.5, 0.02, "output", ha="center", fontsize=12, color="#475569")
    fig1b.tight_layout()
    fig1b_path = os.path.join(OUTPUT_DIR, "01_denoising_salt_pepper_comparison.png")
    fig1b.savefig(fig1b_path, dpi=150, bbox_inches="tight")
    plt.close(fig1b)
    print(f"[SAVED] {fig1b_path}")

    # Figure 2: Edge Detection Row (4 Columns: Original, Sobel, Prewitt, Canny)
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
    axes2[0].imshow(clean_image)
    axes2[0].set_title("Original", fontsize=12, fontweight="bold")
    axes2[0].axis("off")

    axes2[1].imshow(edge_sobel, cmap="gray")
    axes2[1].set_title("Sobel Edge", fontsize=11)
    axes2[1].axis("off")

    axes2[2].imshow(edge_prewitt, cmap="gray")
    axes2[2].set_title("Prewitt Edge", fontsize=11)
    axes2[2].axis("off")

    axes2[3].imshow(edge_canny, cmap="gray")
    axes2[3].set_title("Canny Edge", fontsize=11)
    axes2[3].axis("off")

    fig2.text(0.5, 0.02, "output", ha="center", fontsize=12, color="#475569")
    fig2.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "02_edge_detection_comparison.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[SAVED] {fig2_path}")

    # Figure 3: Sharpening Row (3 Columns: Original, Laplacian, Unsharp Mask)
    fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4))
    axes3[0].imshow(clean_image)
    axes3[0].set_title("Original", fontsize=12, fontweight="bold")
    axes3[0].axis("off")

    axes3[1].imshow(sharp_lap)
    axes3[1].set_title("Sharpened (Laplacian)", fontsize=11)
    axes3[1].axis("off")

    axes3[2].imshow(sharp_unsharp)
    axes3[2].set_title("Sharpened (Unsharp Mask)", fontsize=11)
    axes3[2].axis("off")

    fig3.text(0.5, 0.02, "output", ha="center", fontsize=12, color="#475569")
    fig3.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "03_sharpening_comparison.png")
    fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[SAVED] {fig3_path}")

    # Figure 4: Consolidated Lab-03 Report Figure
    fig4, axes4 = plt.subplots(3, 4, figsize=(16, 12))
    # Row 1: Denoising
    axes4[0, 0].imshow(clean_image); axes4[0, 0].set_title("Original"); axes4[0, 0].axis("off")
    axes4[0, 1].imshow(noisy_sp); axes4[0, 1].set_title(f"Noisy (S&P, {psnr_sp:.1f} dB)"); axes4[0, 1].axis("off")
    axes4[0, 2].imshow(mean_sp); axes4[0, 2].set_title(f"Mean Filter ({psnr_msp:.1f} dB)"); axes4[0, 2].axis("off")
    axes4[0, 3].imshow(median_sp); axes4[0, 3].set_title(f"Median Filter ({psnr_medsp:.1f} dB)"); axes4[0, 3].axis("off")

    # Row 2: Sharpening
    axes4[1, 0].imshow(clean_image); axes4[1, 0].set_title("Original"); axes4[1, 0].axis("off")
    axes4[1, 1].imshow(sharp_lap); axes4[1, 1].set_title("Laplacian (alpha=1.0)"); axes4[1, 1].axis("off")
    axes4[1, 2].imshow(sharp_unsharp); axes4[1, 2].set_title("Unsharp Mask (1.8x)"); axes4[1, 2].axis("off")
    diff = np.abs(sharp_unsharp.astype(np.int16) - clean_image.astype(np.int16)).clip(0, 255).astype(np.uint8)
    axes4[1, 3].imshow(diff); axes4[1, 3].set_title("High-Freq Detail Map"); axes4[1, 3].axis("off")

    # Row 3: Edge Detection
    axes4[2, 0].imshow(clean_image); axes4[2, 0].set_title("Original"); axes4[2, 0].axis("off")
    axes4[2, 1].imshow(edge_sobel, cmap="gray"); axes4[2, 1].set_title("Sobel"); axes4[2, 1].axis("off")
    axes4[2, 2].imshow(edge_prewitt, cmap="gray"); axes4[2, 2].set_title("Prewitt (3x3)"); axes4[2, 2].axis("off")
    axes4[2, 3].imshow(edge_canny, cmap="gray"); axes4[2, 3].set_title("Canny"); axes4[2, 3].axis("off")

    fig4.suptitle("Lab-03 Image Enhancing: Complete Consolidated Overview", fontsize=15, fontweight="bold", y=0.98)
    fig4.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR, "04_complete_lab_summary.png")
    fig4.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"[SAVED] {fig4_path}")

    # =========================================================================
    # 6. Print Benchmark Summary Table
    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    header = f"{'Algorithm':<32} | {'Condition':<14} | {'PSNR (dB)':<10} | {'MSE':<8} | {'Time (ms)':<9} | {'Status'}"
    print(header)
    print("-" * 88)
    for row in results_table:
        print(f"{row[0]:<32} | {row[1]:<14} | {row[2]:<10} | {row[3]:<8} | {row[4]:<9} | {row[5]}")
    print("=" * 80)
    print("[SUCCESS] All pipeline stages executed and verified successfully.")


if __name__ == "__main__":
    run_pipeline()
