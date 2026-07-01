"""
compression_comparison.py
==========================
Comprehensive Benchmark: Text-Aware Video Compression vs Standard Algorithms.

Compares:
  - H.264 / AVC (libx264)   — at multiple CRF levels → 85%, 70%, 60% space targets
  - H.265 / HEVC (libx265)  — same quality targets
  - VP9                     — same quality targets
  - Our Proposed Method     — Text-Aware DCT (bg_quality=30/50/70, text_quality=90)

Metrics collected per method & quality level:
  - Original file size (MB)
  - Compressed file size (MB)
  - Space saved (%)
  - Compression ratio
  - PSNR  (Peak Signal-to-Noise Ratio, dB)
  - SSIM  (Structural Similarity Index)
  - Encoding time (seconds)
  - Bits-per-pixel (bpp)

Outputs (all inside comparison_output/):
  - comparison_results.json           — raw numbers
  - comparison_report.txt             — human-readable report
  - charts/
      bar_filesize.png                — file size side-by-side bars
      bar_compression_ratio.png       — compression ratio per method
      bar_ssim.png                    — SSIM quality comparison
      bar_psnr.png                    — PSNR quality comparison
      bar_encoding_time.png           — encoding time
      scatter_quality_vs_ratio.png    — quality vs compression tradeoff
      radar_method_profile.png        — radar chart per method
      heatmap_metrics.png             — metric heatmap
      flowchart_proposed.png          — our proposed pipeline diagram
      flowchart_standard.png          — standard codec pipeline diagram
      combined_dashboard.png          — all-in-one summary dashboard
      target_comparison.png           — 85/70/60% space target comparison

Usage:
    ./craft_env/bin/python3 compression_comparison.py
    ./craft_env/bin/python3 compression_comparison.py --input TESTFILE3.mp4 --frames 150
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import shutil
import textwrap
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import PatchCollection
from scipy.fft import dctn, idctn
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = "comparison_output"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

# Codec configurations mapping target space-saving to codec CRF/quality
CODEC_CONFIGS = {
    "H.264 (85% target)":  {"codec": "h264",  "crf": 28, "target_pct": 85},
    "H.264 (70% target)":  {"codec": "h264",  "crf": 35, "target_pct": 70},
    "H.264 (60% target)":  {"codec": "h264",  "crf": 40, "target_pct": 60},
    "H.265 (85% target)":  {"codec": "h265",  "crf": 24, "target_pct": 85},
    "H.265 (70% target)":  {"codec": "h265",  "crf": 32, "target_pct": 70},
    "H.265 (60% target)":  {"codec": "h265",  "crf": 38, "target_pct": 60},
    "VP9 (85% target)":    {"codec": "vp9",   "crf": 33, "target_pct": 85},
    "VP9 (70% target)":    {"codec": "vp9",   "crf": 40, "target_pct": 70},
    "VP9 (60% target)":    {"codec": "vp9",   "crf": 48, "target_pct": 60},
}

# Our proposed method at three compression strengths
PROPOSED_CONFIGS = {
    "Proposed (85% target)": {"text_quality": 90, "bg_quality": 65, "ssim_threshold": 0.98, "target_pct": 85},
    "Proposed (70% target)": {"text_quality": 90, "bg_quality": 35, "ssim_threshold": 0.95, "target_pct": 70},
    "Proposed (60% target)": {"text_quality": 85, "bg_quality": 20, "ssim_threshold": 0.92, "target_pct": 60},
}

# Color palette for charts
COLORS = {
    "H.264":    "#2196F3",   # Blue
    "H.265":    "#4CAF50",   # Green
    "VP9":      "#FF9800",   # Orange
    "Proposed": "#E91E63",   # Pink/Red
}

TARGET_COLORS = {
    "85%": "#66BB6A",
    "70%": "#FFA726",
    "60%": "#EF5350",
}

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def mb(path):
    """File size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def sample_frames(video_path, max_frames=60):
    """Uniformly sample frames from a video. Returns list of BGR arrays."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), min(max_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def compute_video_quality(original_path, compressed_path, max_frames=40):
    """
    Compute average PSNR and SSIM between the original and compressed video.
    Compares corresponding frames (uniformly sampled).
    """
    orig_frames = sample_frames(original_path, max_frames)
    comp_frames = sample_frames(compressed_path, max_frames)
    n = min(len(orig_frames), len(comp_frames))
    if n == 0:
        return 0.0, 0.0

    psnr_vals, ssim_vals = [], []
    for i in range(n):
        o = orig_frames[i]
        c = comp_frames[i]
        # Resize compressed to original shape if needed
        if o.shape != c.shape:
            c = cv2.resize(c, (o.shape[1], o.shape[0]))
        og = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
        cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        try:
            psnr_vals.append(psnr_metric(og, cg, data_range=255))
        except Exception:
            psnr_vals.append(0.0)
        try:
            ssim_val, _ = ssim_metric(og, cg, full=True)
            ssim_vals.append(float(ssim_val))
        except Exception:
            ssim_vals.append(0.0)

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def get_video_info(video_path):
    """Return (fps, frame_count, width, height) for a video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, fc, w, h


# ─────────────────────────────────────────────────────────────────
# STANDARD CODEC COMPRESSION  (via ffmpeg)
# ─────────────────────────────────────────────────────────────────

def compress_with_codec(input_path, output_path, codec, crf, max_frames=None):
    """
    Compress video with a standard codec using ffmpeg.
    Returns encoding_time_seconds.
    """
    vf_filter = ""
    if max_frames:
        vf_filter = f"-vframes {max_frames}"

    if codec == "h264":
        cmd = (f'ffmpeg -y -i "{input_path}" {vf_filter} '
               f'-c:v libx264 -crf {crf} -preset fast -an '
               f'"{output_path}" 2>&1')
    elif codec == "h265":
        cmd = (f'ffmpeg -y -i "{input_path}" {vf_filter} '
               f'-c:v libx265 -crf {crf} -preset fast -an '
               f'"{output_path}" 2>&1')
    elif codec == "vp9":
        cmd = (f'ffmpeg -y -i "{input_path}" {vf_filter} '
               f'-c:v libvpx-vp9 -crf {crf} -b:v 0 -an '
               f'"{output_path}" 2>&1')
    else:
        raise ValueError(f"Unknown codec: {codec}")

    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    enc_time = time.time() - t0

    if result.returncode != 0 and not os.path.exists(output_path):
        print(f"  [WARN] ffmpeg error for {codec} crf={crf}: {result.stderr[-200:]}")
    return enc_time


# ─────────────────────────────────────────────────────────────────
# PROPOSED TEXT-AWARE COMPRESSION  (inline, no CRAFT — pure DCT)
# ─────────────────────────────────────────────────────────────────

MIN_BLOCK_SIZE = 8
MAX_DEPTH_Q    = 7
VARIANCE_THRESHOLD = 50.0


def _partition(gray, x, y, w, h, depth=0):
    block = gray[y:h, x:w]
    bh, bw = h - y, w - x
    if bh <= 0 or bw <= 0:
        return []
    variance = np.var(block.astype(np.float32))
    if bh <= MIN_BLOCK_SIZE or bw <= MIN_BLOCK_SIZE or depth >= MAX_DEPTH_Q or variance <= VARIANCE_THRESHOLD:
        return [(x, y, w, h, depth, variance)]
    mx, my = x + bw // 2, y + bh // 2
    leaves = []
    leaves += _partition(gray, x,  y,  mx, my, depth + 1)
    leaves += _partition(gray, mx, y,  w,  my, depth + 1)
    leaves += _partition(gray, x,  my, mx, h,  depth + 1)
    leaves += _partition(gray, mx, my, w,  h,  depth + 1)
    return leaves


def _quality_to_step(q):
    q = max(1, min(100, q))
    step = 5000 / q if q < 50 else 200 - 2 * q
    return max(1.0, step)


def _dct_block(block_f, quality):
    step = _quality_to_step(quality)
    coeffs = dctn(block_f, type=2, norm="ortho")
    quantized = np.round(coeffs / step) * step
    return np.clip(idctn(quantized, type=2, norm="ortho"), 0, 255)


def compress_frame_proposed(frame_bgr, text_quality, bg_quality):
    """Compress a single frame using the proposed text-aware DCT method (no CRAFT)."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    result = np.zeros_like(frame_bgr, dtype=np.float32)
    leaf_blocks = _partition(gray, 0, 0, w, h)

    for (bx, by, bw, bh, depth, variance) in leaf_blocks:
        is_uniform = variance <= VARIANCE_THRESHOLD
        # Without CRAFT, label all uniform blocks as background, rest text
        is_text = (variance > 300)  # heuristic: high variance ≈ text/detail
        quality = text_quality if is_text else bg_quality

        if is_uniform and not is_text and (bh - by) >= 16 and (bw - bx) >= 16:
            for c in range(3):
                result[by:bh, bx:bw, c] = np.mean(frame_bgr[by:bh, bx:bw, c].astype(np.float32))
            continue

        for c in range(3):
            result[by:bh, bx:bw, c] = _dct_block(frame_bgr[by:bh, bx:bw, c].astype(np.float32), quality)

    return np.clip(result, 0, 255).astype(np.uint8)


def compress_proposed_method(input_path, output_path, text_quality, bg_quality,
                              ssim_threshold, max_frames=None):
    """
    Run our proposed text-aware DCT compression on a video.
    Returns encoding_time_seconds.
    """
    cap = cv2.VideoCapture(input_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total = min(total, max_frames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

    ref_frame = None
    frame_idx = 0
    t0 = time.time()

    while cap.isOpened() and frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Temporal filter (SSIM-based)
        if ref_frame is not None:
            rg = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sim, _ = ssim_metric(rg, fg, full=True)
            if sim >= ssim_threshold:
                writer.write(ref_frame)
                continue

        compressed = compress_frame_proposed(frame, text_quality, bg_quality)
        writer.write(compressed)
        ref_frame = compressed

    enc_time = time.time() - t0
    cap.release()
    writer.release()
    return enc_time


# ─────────────────────────────────────────────────────────────────
# BENCHMARKING ENGINE
# ─────────────────────────────────────────────────────────────────

def run_benchmark(input_path, max_frames=120):
    """
    Run the full benchmark. Returns a dict of results.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    input_size = mb(input_path)
    fps, total_frames, width, height = get_video_info(input_path)
    video_name = Path(input_path).stem
    num_pixels = width * height

    print(f"\n{'='*70}")
    print(f"COMPRESSION BENCHMARK — {video_name}")
    print(f"{'='*70}")
    print(f"  Input:      {input_path}")
    print(f"  Size:       {input_size:.2f} MB")
    print(f"  Resolution: {width}×{height}  |  FPS: {fps:.1f}  |  Frames: {total_frames}")
    print(f"  Benchmarking {max_frames} frames per method")
    print(f"{'='*70}\n")

    results = {}

    # ── Standard codecs ─────────────────────────────────────────
    for name, cfg in CODEC_CONFIGS.items():
        codec = cfg["codec"]
        crf   = cfg["crf"]
        print(f"  [{codec.upper()}] {name}  (CRF={crf})...")
        out_path = os.path.join(OUTPUT_DIR, f"_tmp_{codec}_crf{crf}.mp4")

        enc_time = compress_with_codec(input_path, out_path, codec, crf, max_frames)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            comp_size = mb(out_path)
            space_saved = (1 - comp_size / input_size) * 100
            ratio = input_size / comp_size if comp_size > 0 else 0
            print(f"    Encoding: {enc_time:.1f}s | Size: {comp_size:.2f}MB | Saved: {space_saved:.1f}%")

            print(f"    Computing PSNR/SSIM...", end="", flush=True)
            psnr_val, ssim_val = compute_video_quality(input_path, out_path, max_frames=30)
            print(f"  PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}")

            bpp = (comp_size * 8 * 1024 * 1024) / (max_frames * num_pixels) if max_frames > 0 else 0
        else:
            comp_size = input_size
            space_saved = 0.0
            ratio = 1.0
            psnr_val = 0.0
            ssim_val = 0.0
            bpp = 0.0
            print(f"    [FAILED] using fallback zeros")

        results[name] = {
            "method_family": codec.upper().replace("264","H.264").replace("265","H.265"),
            "codec": codec,
            "target_pct": cfg["target_pct"],
            "original_size_mb": input_size,
            "compressed_size_mb": comp_size,
            "space_saved_pct": space_saved,
            "compression_ratio": ratio,
            "psnr_db": psnr_val,
            "ssim": ssim_val,
            "encoding_time_s": enc_time,
            "bpp": bpp,
        }

        # Clean temp
        if os.path.exists(out_path):
            os.remove(out_path)

    # ── Proposed method ─────────────────────────────────────────
    for name, cfg in PROPOSED_CONFIGS.items():
        tq = cfg["text_quality"]
        bq = cfg["bg_quality"]
        st = cfg["ssim_threshold"]
        print(f"  [PROPOSED] {name}  (text_q={tq}, bg_q={bq}, ssim_thresh={st})...")
        out_path = os.path.join(OUTPUT_DIR, f"_tmp_proposed_tq{tq}_bq{bq}.mp4")

        enc_time = compress_proposed_method(input_path, out_path, tq, bq, st, max_frames)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            comp_size = mb(out_path)
            space_saved = (1 - comp_size / input_size) * 100
            ratio = input_size / comp_size if comp_size > 0 else 0
            print(f"    Encoding: {enc_time:.1f}s | Size: {comp_size:.2f}MB | Saved: {space_saved:.1f}%")

            print(f"    Computing PSNR/SSIM...", end="", flush=True)
            psnr_val, ssim_val = compute_video_quality(input_path, out_path, max_frames=30)
            print(f"  PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}")

            bpp = (comp_size * 8 * 1024 * 1024) / (max_frames * num_pixels) if max_frames > 0 else 0
        else:
            comp_size = input_size
            space_saved = 0.0
            ratio = 1.0
            psnr_val = 0.0
            ssim_val = 0.0
            bpp = 0.0
            print(f"    [FAILED] using fallback zeros")

        results[name] = {
            "method_family": "Proposed",
            "codec": "proposed",
            "target_pct": cfg["target_pct"],
            "text_quality": tq,
            "bg_quality": bq,
            "original_size_mb": input_size,
            "compressed_size_mb": comp_size,
            "space_saved_pct": space_saved,
            "compression_ratio": ratio,
            "psnr_db": psnr_val,
            "ssim": ssim_val,
            "encoding_time_s": enc_time,
            "bpp": bpp,
        }

        if os.path.exists(out_path):
            os.remove(out_path)

    return results, {
        "video_name": video_name,
        "input_path": str(input_path),
        "input_size_mb": input_size,
        "fps": fps,
        "total_frames": total_frames,
        "resolution": f"{width}×{height}",
        "benchmark_frames": max_frames,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────
# CHART GENERATION
# ─────────────────────────────────────────────────────────────────

def _family_color(name):
    if "H.264" in name or "264" in name or "h264" in name:
        return COLORS["H.264"]
    if "H.265" in name or "265" in name or "h265" in name:
        return COLORS["H.265"]
    if "VP9" in name or "vp9" in name:
        return COLORS["VP9"]
    if "Proposed" in name:
        return COLORS["Proposed"]
    return "#999999"


def _target_hatch(target_pct):
    return {85: "", 70: "//", 60: "xx"}.get(target_pct, "")


def _sorted_keys(results):
    """Sort result keys by family and target."""
    order = ["H.264", "H.265", "VP9", "Proposed"]
    def sort_key(k):
        r = results[k]
        fam = r.get("method_family", "Unknown")
        fi = order.index(fam) if fam in order else 99
        return (fi, r.get("target_pct", 0))
    return sorted(results.keys(), key=sort_key)


# ── 1. Bar: File Size Comparison ──────────────────────────────────
def plot_filesize(results, meta, path):
    keys = _sorted_keys(results)
    orig_sz = meta["input_size_mb"]
    comp_sizes = [results[k]["compressed_size_mb"] for k in keys]
    colors = [_family_color(k) for k in keys]
    labels = [k.replace(" target)", "%)").replace(" (", "\n(") for k in keys]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    x = np.arange(len(keys))
    bars = ax.bar(x, comp_sizes, color=colors, width=0.6, edgecolor="#333", linewidth=0.8, zorder=3)
    ax.axhline(orig_sz, color="#FFD700", linestyle="--", linewidth=1.8, label=f"Original ({orig_sz:.2f} MB)", zorder=4)

    for bar, val in zip(bars, comp_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}MB",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color="white")
    ax.set_ylabel("Compressed Size (MB)", color="white", fontsize=12)
    ax.set_title("📦 Compressed File Size Comparison\n(Lower is Better)", color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.grid(axis="y", color="#333", alpha=0.5, zorder=0)
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    # Legend for families
    patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    ax.legend(handles=patches, loc="upper right", facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 2. Bar: Compression Ratio ─────────────────────────────────────
def plot_compression_ratio(results, path):
    keys = _sorted_keys(results)
    ratios = [results[k]["compression_ratio"] for k in keys]
    colors = [_family_color(k) for k in keys]
    labels = [k.replace(" target)", "%)").replace(" (", "\n(") for k in keys]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    x = np.arange(len(keys))
    bars = ax.bar(x, ratios, color=colors, width=0.6, edgecolor="#333", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.2f}×",
                ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color="white")
    ax.set_ylabel("Compression Ratio (×)", color="white", fontsize=12)
    ax.set_title("📊 Compression Ratio Comparison\n(Higher is Better)", color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.grid(axis="y", color="#333", alpha=0.5, zorder=0)

    patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    ax.legend(handles=patches, loc="upper left", facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 3. Bar: SSIM Quality ─────────────────────────────────────────
def plot_ssim(results, path):
    keys = _sorted_keys(results)
    ssim_vals = [results[k]["ssim"] for k in keys]
    colors = [_family_color(k) for k in keys]
    labels = [k.replace(" target)", "%)").replace(" (", "\n(") for k in keys]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    x = np.arange(len(keys))
    bars = ax.bar(x, ssim_vals, color=colors, width=0.6, edgecolor="#333", linewidth=0.8, zorder=3)
    ax.axhline(0.9, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.8, label="Good Quality (0.9)", zorder=4)
    ax.axhline(0.8, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.8, label="Acceptable (0.8)", zorder=4)

    for bar, val in zip(bars, ssim_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color="white")
    ax.set_ylabel("SSIM (0–1, higher is better)", color="white", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("🎯 SSIM Quality Comparison\n(Higher is Better — 1.0 = Perfect)", color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.grid(axis="y", color="#333", alpha=0.5, zorder=0)

    patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    ax.legend(handles=patches + [
        mpatches.Patch(color="#4CAF50", label="Good (0.9)"),
        mpatches.Patch(color="#FF9800", label="Acceptable (0.8)"),
    ], loc="lower right", facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 4. Bar: PSNR ─────────────────────────────────────────────────
def plot_psnr(results, path):
    keys = _sorted_keys(results)
    psnr_vals = [results[k]["psnr_db"] for k in keys]
    colors = [_family_color(k) for k in keys]
    labels = [k.replace(" target)", "%)").replace(" (", "\n(") for k in keys]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    x = np.arange(len(keys))
    bars = ax.bar(x, psnr_vals, color=colors, width=0.6, edgecolor="#333", linewidth=0.8, zorder=3)
    ax.axhline(35, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.8, label="Excellent (35 dB)")
    ax.axhline(30, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.8, label="Good (30 dB)")
    ax.axhline(25, color="#F44336", linestyle="--", linewidth=1.5, alpha=0.8, label="Acceptable (25 dB)")

    for bar, val in zip(bars, psnr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}dB",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color="white")
    ax.set_ylabel("PSNR (dB, higher is better)", color="white", fontsize=12)
    ax.set_title("📡 PSNR Quality Comparison\n(Higher is Better)", color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.grid(axis="y", color="#333", alpha=0.5, zorder=0)

    patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    ax.legend(handles=patches, loc="upper right", facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 5. Bar: Encoding Time ─────────────────────────────────────────
def plot_encoding_time(results, path):
    keys = _sorted_keys(results)
    times = [results[k]["encoding_time_s"] for k in keys]
    colors = [_family_color(k) for k in keys]
    labels = [k.replace(" target)", "%)").replace(" (", "\n(") for k in keys]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    x = np.arange(len(keys))
    bars = ax.bar(x, times, color=colors, width=0.6, edgecolor="#333", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}s",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color="white")
    ax.set_ylabel("Encoding Time (seconds)", color="white", fontsize=12)
    ax.set_title("⏱ Encoding Time Comparison\n(Lower is Better)", color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.grid(axis="y", color="#333", alpha=0.5, zorder=0)

    patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    ax.legend(handles=patches, loc="upper right", facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 6. Scatter: Quality vs Compression Ratio ─────────────────────
def plot_quality_vs_ratio(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#0F1117")

    for ax, (metric, label, ylabel) in zip(axes, [
        ("ssim", "SSIM", "SSIM (Higher = Better)"),
        ("psnr_db", "PSNR", "PSNR in dB (Higher = Better)"),
    ]):
        ax.set_facecolor("#0F1117")
        families = {}
        for name, r in results.items():
            fam = r["method_family"]
            families.setdefault(fam, {"ratios": [], "quals": [], "targets": []})
            families[fam]["ratios"].append(r["compression_ratio"])
            families[fam]["quals"].append(r[metric])
            families[fam]["targets"].append(r["target_pct"])

        for fam, data in families.items():
            color = COLORS.get(fam, "#999")
            ax.scatter(data["ratios"], data["quals"], c=color, s=180, zorder=5,
                       edgecolors="white", linewidths=0.8, label=fam)
            # Connect points with line
            sorted_pairs = sorted(zip(data["ratios"], data["quals"]))
            xs, ys = zip(*sorted_pairs)
            ax.plot(xs, ys, color=color, alpha=0.4, linewidth=1.5, linestyle="--", zorder=4)
            # Annotate target %
            for rx, qy, tgt in zip(data["ratios"], data["quals"], data["targets"]):
                ax.annotate(f"{tgt}%", (rx, qy), textcoords="offset points",
                            xytext=(5, 5), fontsize=8, color=color, fontweight="bold")

        ax.set_xlabel("Compression Ratio (×)", color="white", fontsize=11)
        ax.set_ylabel(ylabel, color="white", fontsize=11)
        ax.set_title(f"Quality vs Compression Ratio\n({label})", color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.grid(color="#333", alpha=0.4, zorder=0)
        ax.legend(facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=10)

    fig.suptitle("🔍 Quality-Compression Trade-off", color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 7. Target Comparison (85% / 70% / 60%) ───────────────────────
def plot_target_comparison(results, path):
    """Compare all methods head-to-head at each space-saving target."""
    targets = [85, 70, 60]
    metrics = ["ssim", "psnr_db", "compression_ratio", "encoding_time_s"]
    metric_labels = ["SSIM ↑", "PSNR (dB) ↑", "Comp. Ratio ↑", "Encode Time (s) ↓"]

    families = ["H264", "H265", "VP9", "Proposed"]
    family_map = {
        "h264": "H264", "h265": "H265", "vp9": "VP9", "proposed": "Proposed"
    }
    family_colors = {
        "H264": COLORS["H.264"], "H265": COLORS["H.265"],
        "VP9": COLORS["VP9"], "Proposed": COLORS["Proposed"]
    }

    fig, axes = plt.subplots(len(targets), len(metrics), figsize=(22, 14))
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle("📊 Method vs Target Space Savings — Side-by-Side Comparison",
                 color="white", fontsize=15, fontweight="bold", y=1.01)

    for row, target in enumerate(targets):
        for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row][col]
            ax.set_facecolor("#111827")

            # Gather data for this target
            vals, cols_, fams = [], [], []
            for name, r in results.items():
                if r["target_pct"] == target:
                    fam = family_map.get(r["codec"], "Other")
                    vals.append(r[metric])
                    cols_.append(family_colors.get(fam, "#999"))
                    fams.append(fam)

            xpos = np.arange(len(fams))
            bars = ax.bar(xpos, vals, color=cols_, edgecolor="#333", linewidth=0.6, zorder=3)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7, color="white", fontweight="bold")

            ax.set_xticks(xpos)
            ax.set_xticklabels(fams, color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            ax.spines[:].set_color("#333")
            ax.grid(axis="y", color="#333", alpha=0.4, zorder=0)

            if col == 0:
                ax.set_ylabel(f"Target: {target}% space\nsaved", color=TARGET_COLORS[f"{target}%"],
                              fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(mlabel, color="white", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 8. Radar Chart ───────────────────────────────────────────────
def plot_radar(results, path):
    """Radar chart comparing method families at 70% target."""
    categories = ["SSIM", "PSNR\n(norm)", "Comp\nRatio", "Encode\nSpeed", "Space\nSaved"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#111827")

    family_data = {}
    for name, r in results.items():
        if r["target_pct"] != 70:
            continue
        fam = r["method_family"]
        family_data[fam] = r

    # Normalize each metric 0–1 across families
    all_ssim = [r["ssim"] for r in family_data.values()]
    all_psnr = [r["psnr_db"] for r in family_data.values()]
    all_ratio = [r["compression_ratio"] for r in family_data.values()]
    all_time = [r["encoding_time_s"] for r in family_data.values()]
    all_space = [r["space_saved_pct"] for r in family_data.values()]

    def norm(val, vals):
        mn, mx = min(vals), max(vals)
        return (val - mn) / (mx - mn + 1e-9)

    for fam, r in family_data.items():
        color = COLORS.get(fam, "#999")
        vals = [
            norm(r["ssim"], all_ssim),
            norm(r["psnr_db"], all_psnr),
            norm(r["compression_ratio"], all_ratio),
            1 - norm(r["encoding_time_s"], all_time),  # lower time = better
            norm(r["space_saved_pct"], all_space),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2.5, linestyle="solid", label=fam, zorder=5)
        ax.fill(angles, vals, color=color, alpha=0.18, zorder=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#aaa", fontsize=8)
    ax.spines["polar"].set_color("#333")
    ax.grid(color="#444", alpha=0.5)
    ax.set_title("📡 Method Profile Radar Chart\n(70% Space Saving Target — Normalized)",
                 color="white", fontsize=13, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 9. Heatmap ───────────────────────────────────────────────────
def plot_heatmap(results, path):
    """Metric heatmap — all methods × all metrics."""
    keys = _sorted_keys(results)
    short_keys = [k.replace(" target)", "%)").replace(" (", " (") for k in keys]
    metrics = ["space_saved_pct", "compression_ratio", "ssim", "psnr_db", "encoding_time_s"]
    metric_labels = ["Space Saved (%)", "Comp. Ratio (×)", "SSIM", "PSNR (dB)", "Encode Time (s)"]

    data = np.array([[results[k][m] for m in metrics] for k in keys])

    # Normalize each column 0-1 for color (but keep raw values in cells)
    norm_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        col_vals = data[:, col]
        mn, mx = col_vals.min(), col_vals.max()
        norm_data[:, col] = (col_vals - mn) / (mx - mn + 1e-9)

    # Invert encoding time (lower = better → should be greener)
    norm_data[:, 4] = 1 - norm_data[:, 4]

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, color="white", fontsize=11, fontweight="bold", rotation=15, ha="right")
    ax.set_yticks(range(len(short_keys)))
    ax.set_yticklabels(short_keys, color="white", fontsize=9)

    for i in range(len(keys)):
        for j in range(len(metrics)):
            val = data[i, j]
            txt = f"{val:.2f}" if metrics[j] not in ["space_saved_pct"] else f"{val:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if norm_data[i, j] > 0.5 else "white", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Normalized Performance (Green=Best)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    ax.set_title("🌡 Performance Heatmap — All Methods & Metrics",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 10. Flowchart: Proposed Method ───────────────────────────────
def plot_flowchart_proposed(path):
    fig, ax = plt.subplots(figsize=(14, 20))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis("off")

    def box(x, y, w, h, text, color, text_color="white", font_size=10, radius=0.3):
        fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle=f"round,pad={radius}",
                                facecolor=color, edgecolor="#ffffff55", linewidth=1.5, zorder=5)
        ax.add_patch(fancy)
        ax.text(x, y, text, ha="center", va="center", fontsize=font_size,
                color=text_color, fontweight="bold", zorder=6,
                wrap=True, multialignment="center")

    def diamond(x, y, w, h, text, color):
        pts = np.array([[x, y + h/2], [x + w/2, y], [x, y - h/2], [x - w/2, y]])
        poly = plt.Polygon(pts, closed=True, facecolor=color, edgecolor="#ffffff55", linewidth=1.5, zorder=5)
        ax.add_patch(poly)
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", zorder=6, multialignment="center")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#aaaaff", lw=2.0), zorder=4)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, color="#aaffaa", fontsize=8.5, fontweight="bold", zorder=7)

    ax.text(5, 21.3, "🔷 Proposed Text-Aware Video Compression Pipeline",
            ha="center", va="center", fontsize=14, color="white", fontweight="bold")
    ax.text(5, 20.8, "Text-Aware DCT  ·  Reference Frame Temporal Filtering",
            ha="center", va="center", fontsize=10, color="#aaaaff")

    # ── Blocks ──
    box(5, 20.2, 5, 0.7, "📹  Input Video", "#1565C0", font_size=11)
    arrow(5, 19.85, 5, 19.25)
    box(5, 18.95, 5, 0.7, "🎞  Frame Extraction", "#1976D2", font_size=10)
    arrow(5, 18.6, 5, 17.95)
    box(5, 17.65, 5.5, 0.8, "⚡ TECHNIQUE 2: Temporal Filter\n(SSIM-based Frame Skipping)", "#4A148C", font_size=9.5)
    arrow(5, 17.25, 5, 16.6)
    diamond(5, 16.2, 5, 0.8, "SSIM ≥ threshold?", "#6A1B9A")
    ax.text(7.8, 16.2, "YES → Skip Frame", color="#66BB6A", fontsize=9, va="center", fontweight="bold")
    arrow(7.65, 16.2, 9.0, 16.2)
    box(9.0, 16.2, 1.6, 0.6, "Reuse Last\nFrame", "#1B5E20", font_size=8)
    arrow(5, 15.8, 5, 15.2, "NO")
    box(5, 14.9, 5.5, 0.8, "⚡ TECHNIQUE 1: Text-Aware Spatial\nCompression (9-Step Pipeline)", "#B71C1C", font_size=9.5)

    steps = [
        ("Step 1 · Preprocessing\n(Resize · Normalize · Noise Reduction)", "#C62828"),
        ("Step 2 · CRAFT Text Detection\n(Character + Link Score Maps)", "#AD1457"),
        ("Step 3 · Binary Text Mask\n(255=Text  |  0=Background)", "#6A1B9A"),
        ("Step 4 · Adaptive Quadtree Partitioning\n(Recursive Block Subdivision)", "#283593"),
        ("Step 5 · Block Variance Analysis\n(High Variance → Subdivide Further)", "#00695C"),
        ("Step 6 · Block Classification\n(Text Block  |  Background Block)", "#E65100"),
        ("Step 7 · DCT Quantization\n(Text: Q=90 high  |  BG: Q=30 low)", "#4E342E"),
        ("Step 8 · Uniform BG Simplification\n(Flat regions → Average color fill)", "#1A237E"),
        ("Step 9 · Inverse DCT & Reconstruction\n(IDCT + Stitch blocks → Final Frame)", "#1B5E20"),
    ]
    y = 14.3
    for i, (text, color) in enumerate(steps):
        box(5, y, 5.8, 0.72, text, color, font_size=8.5)
        if i < len(steps) - 1:
            arrow(5, y - 0.36, 5, y - 0.64)
        y -= 1.0

    y_final = y + 0.28
    arrow(5, y_final - 0.4, 5, y_final - 1.0)
    box(5, y_final - 1.3, 5.5, 0.8,
        "🗂 Output Files\n_compressed.mp4  ·  _detected.mp4\n_text_heatmap.mp4  ·  analysis/", "#263238", font_size=8.5)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 11. Flowchart: Standard Codec Pipeline ────────────────────────
def plot_flowchart_standard(path):
    fig, ax = plt.subplots(figsize=(14, 16))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18)
    ax.axis("off")

    def box(x, y, w, h, text, color, font_size=10):
        fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle="round,pad=0.25",
                                facecolor=color, edgecolor="#ffffff44", linewidth=1.5, zorder=5)
        ax.add_patch(fancy)
        ax.text(x, y, text, ha="center", va="center", fontsize=font_size,
                color="white", fontweight="bold", zorder=6, multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#aaaaff", lw=2.0), zorder=4)

    ax.text(6, 17.5, "🔶 Standard Video Codec Pipeline",
            ha="center", va="center", fontsize=14, color="white", fontweight="bold")
    ax.text(6, 17.0, "H.264 (AVC)  ·  H.265 (HEVC)  ·  VP9",
            ha="center", va="center", fontsize=10, color="#ffcc80")

    # Three columns: H264, H265, VP9
    columns = [
        ("H.264 / AVC\n(libx264)", "#0D47A1", 2.0),
        ("H.265 / HEVC\n(libx265)", "#1B5E20", 6.0),
        ("VP9\n(libvpx-vp9)", "#E65100", 10.0),
    ]

    for (name, color, cx) in columns:
        box(cx, 16.2, 3.2, 0.7, name, color, font_size=11)

    steps_shared = [
        ("Input Raw Video Frames", "#263238"),
        ("Intra-Frame Prediction\n(Spatial Redundancy Removal)", "#37474F"),
        ("Inter-Frame Prediction\n(Temporal Motion Estimation)", "#455A64"),
        ("Discrete Cosine Transform\n(DCT — uniform quantization)", "#546E7A"),
        ("Entropy Coding\n(CABAC / CAVLC / ANS)", "#607D8B"),
        ("Bitstream Packing\n(Container: MP4 / WebM)", "#78909C"),
    ]

    y = 15.2
    for cx, _ in [(2.0, ""), (6.0, ""), (10.0, "")]:
        arrow(cx, 15.85, cx, 15.55)

    for i, (step_text, color) in enumerate(steps_shared):
        for cx, _ in [(2.0, ""), (6.0, ""), (10.0, "")]:
            box(cx, y, 3.2, 0.75, step_text, color, font_size=8.5)
            if i < len(steps_shared) - 1:
                arrow(cx, y - 0.375, cx, y - 0.625)
        y -= 1.0

    # Final decode
    y_out = y + 0.2
    for cx, _ in [(2.0, ""), (6.0, ""), (10.0, "")]:
        arrow(cx, y_out, cx, y_out - 0.5)
        box(cx, y_out - 0.8, 3.2, 0.6, "Compressed Output\nMP4 / WebM", "#1B5E20", font_size=8.5)

    # Differentiators
    differentiators = [
        (2.0, "Pros: Fast, universal\nCons: Moderate compression", "#0D47A1"),
        (6.0, "Pros: 40–50% better than H.264\nCons: Slow encoding", "#1B5E20"),
        (10.0, "Pros: Royalty-free, good quality\nCons: Very slow encoding", "#E65100"),
    ]
    for cx, text, color in differentiators:
        box(cx, y_out - 1.7, 3.4, 0.75, text, color, font_size=8)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ── 12. All-in-One Dashboard ─────────────────────────────────────
def plot_dashboard(results, meta, path):
    """One-page summary dashboard."""
    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

    def ax_style(ax, title):
        ax.set_facecolor("#111827")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#333")
        ax.grid(color="#333", alpha=0.4, zorder=0)

    keys = _sorted_keys(results)
    short = [k.replace(" target)", "%)").replace(" (", "\n(") for k in keys]
    colors = [_family_color(k) for k in keys]

    # A) File Size
    ax1 = fig.add_subplot(gs[0, :2])
    ax_style(ax1, "📦 File Size (MB)")
    x = np.arange(len(keys))
    bars = ax1.bar(x, [results[k]["compressed_size_mb"] for k in keys], color=colors, edgecolor="#333", zorder=3)
    ax1.axhline(meta["input_size_mb"], color="#FFD700", linestyle="--", lw=1.5, label="Original")
    ax1.set_xticks(x); ax1.set_xticklabels(short, rotation=35, ha="right", color="white", fontsize=7)
    ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#555")

    # B) Compression Ratio
    ax2 = fig.add_subplot(gs[0, 2:])
    ax_style(ax2, "📊 Compression Ratio (×)")
    bars2 = ax2.bar(x, [results[k]["compression_ratio"] for k in keys], color=colors, edgecolor="#333", zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(short, rotation=35, ha="right", color="white", fontsize=7)
    for b, v in zip(bars2, [results[k]["compression_ratio"] for k in keys]):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}×", ha="center", va="bottom", fontsize=6.5, color="white")

    # C) SSIM
    ax3 = fig.add_subplot(gs[1, :2])
    ax_style(ax3, "🎯 SSIM Quality")
    bars3 = ax3.bar(x, [results[k]["ssim"] for k in keys], color=colors, edgecolor="#333", zorder=3)
    ax3.axhline(0.9, color="#4CAF50", linestyle="--", lw=1.2, label="Good (0.9)")
    ax3.set_ylim(0, 1.1); ax3.set_xticks(x); ax3.set_xticklabels(short, rotation=35, ha="right", color="white", fontsize=7)
    ax3.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#555")

    # D) PSNR
    ax4 = fig.add_subplot(gs[1, 2:])
    ax_style(ax4, "📡 PSNR (dB)")
    bars4 = ax4.bar(x, [results[k]["psnr_db"] for k in keys], color=colors, edgecolor="#333", zorder=3)
    ax4.axhline(35, color="#4CAF50", linestyle="--", lw=1.2, label="Excellent (35dB)")
    ax4.axhline(30, color="#FF9800", linestyle="--", lw=1.2, label="Good (30dB)")
    ax4.set_xticks(x); ax4.set_xticklabels(short, rotation=35, ha="right", color="white", fontsize=7)
    ax4.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#555")

    # E) Encoding Time
    ax5 = fig.add_subplot(gs[2, :2])
    ax_style(ax5, "⏱ Encoding Time (s)")
    bars5 = ax5.bar(x, [results[k]["encoding_time_s"] for k in keys], color=colors, edgecolor="#333", zorder=3)
    ax5.set_xticks(x); ax5.set_xticklabels(short, rotation=35, ha="right", color="white", fontsize=7)

    # F) Space Saved %
    ax6 = fig.add_subplot(gs[2, 2:])
    ax_style(ax6, "💾 Space Saved (%)")
    bars6 = ax6.bar(x, [results[k]["space_saved_pct"] for k in keys], color=colors, edgecolor="#333", zorder=3)
    ax6.axhline(85, color=TARGET_COLORS["85%"], linestyle="--", lw=1.2, label="85% target")
    ax6.axhline(70, color=TARGET_COLORS["70%"], linestyle="--", lw=1.2, label="70% target")
    ax6.axhline(60, color=TARGET_COLORS["60%"], linestyle="--", lw=1.2, label="60% target")
    ax6.set_xticks(x); ax6.set_xticklabels(short, rotation=35, ha="right", color="white", fontsize=7)
    ax6.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#555")

    # Legend for families
    patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    fig.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.0, 1.0),
               facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=10)

    fig.suptitle(
        f"🎬 Compression Benchmark Dashboard — {meta['video_name']}\n"
        f"{meta['resolution']}  ·  {meta['fps']:.0f}fps  ·  {meta['benchmark_frames']} frames benchmarked  ·  "
        f"Original size: {meta['input_size_mb']:.2f} MB  ·  Generated: {meta['generated']}",
        color="white", fontsize=13, fontweight="bold", y=1.02
    )

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────

def generate_report(results, meta, report_path):
    keys = _sorted_keys(results)
    with open(report_path, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("COMPRESSION ALGORITHM COMPARISON REPORT\n")
        f.write(f"Generated : {meta['generated']}\n")
        f.write(f"Video     : {meta['video_name']}\n")
        f.write(f"Input     : {meta['input_path']}\n")
        f.write(f"Size      : {meta['input_size_mb']:.2f} MB\n")
        f.write(f"Resolution: {meta['resolution']}\n")
        f.write(f"FPS       : {meta['fps']:.1f}\n")
        f.write(f"Frames benchmarked: {meta['benchmark_frames']}\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"{'Method':<35} {'Size(MB)':>9} {'Saved%':>8} {'Ratio':>7} {'SSIM':>8} {'PSNR(dB)':>9} {'EncTime(s)':>11}\n")
        f.write("-" * 90 + "\n")
        for k in keys:
            r = results[k]
            f.write(f"{k:<35} {r['compressed_size_mb']:>9.2f} {r['space_saved_pct']:>8.1f} "
                    f"{r['compression_ratio']:>7.2f} {r['ssim']:>8.4f} {r['psnr_db']:>9.2f} "
                    f"{r['encoding_time_s']:>11.2f}\n")
        f.write("-" * 90 + "\n\n")

        for target in [85, 70, 60]:
            f.write(f"\n{'─'*90}\n")
            f.write(f"TARGET: {target}% Space Savings\n")
            f.write(f"{'─'*90}\n")
            f.write(f"{'Method':<35} {'SSIM':>8} {'PSNR':>8} {'Ratio':>7} {'Time(s)':>9} {'Saved%':>8}\n")
            f.write(f"{'─'*35} {'─'*8} {'─'*8} {'─'*7} {'─'*9} {'─'*8}\n")
            for k in keys:
                r = results[k]
                if r["target_pct"] == target:
                    f.write(f"{k:<35} {r['ssim']:>8.4f} {r['psnr_db']:>8.2f} "
                            f"{r['compression_ratio']:>7.2f} {r['encoding_time_s']:>9.2f} "
                            f"{r['space_saved_pct']:>8.1f}\n")

        f.write("\n\n" + "=" * 90 + "\n")
        f.write("METHODOLOGY NOTES\n")
        f.write("=" * 90 + "\n")
        f.write(textwrap.dedent("""
        QUALITY METRICS:
          SSIM  — Structural Similarity Index Measure (0–1, 1=perfect).
                  Measures perceptual similarity by luminance, contrast, structure.
                  > 0.9  : Excellent  |  0.8–0.9: Good  |  < 0.8: Degraded
          PSNR  — Peak Signal-to-Noise Ratio (dB).
                  Measures reconstruction fidelity in pixel domain.
                  > 35dB : Excellent  |  30–35dB: Good   |  < 25dB: Poor

        COMPRESSION METHODS:
          H.264 — Industry-standard AVC codec. CRF: 28 (85%), 35 (70%), 40 (60%).
          H.265 — Modern HEVC codec (~50% better than H.264). CRF: 24 (85%), 32 (70%), 38 (60%).
          VP9   — Google's royalty-free codec. CRF: 33 (85%), 40 (70%), 48 (60%).
          Proposed — Text-Aware DCT + SSIM Temporal Filtering:
                     - Aggressive background compression (Q=20–65)
                     - High-quality text region preservation (Q=85–90)
                     - SSIM-based frame skipping (threshold 0.92–0.98)
                     - Adaptive quadtree block partitioning
                     - Uniform background simplification (mean-fill)

        BENCHMARK SETUP:
          Frames: First N frames of the input video are used.
          SSIM/PSNR: Computed on uniformly sampled corresponding frames.
          Encoding time: Wall-clock time from first frame read to file close.
        """))
    print(f"  ✓ {os.path.basename(report_path)}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compression Algorithm Comparison Benchmark")
    parser.add_argument("--input", default="TESTFILE3.mp4",
                        help="Input video file (default: TESTFILE3.mp4)")
    parser.add_argument("--frames", type=int, default=120,
                        help="Number of frames to benchmark (default: 120)")
    parser.add_argument("--outdir", default="comparison_output",
                        help="Output directory (default: comparison_output)")
    args = parser.parse_args()

    global OUTPUT_DIR, CHARTS_DIR
    OUTPUT_DIR = args.outdir
    CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Find input video
    input_path = args.input
    if not os.path.exists(input_path):
        # Try mp4s in current dir
        available = list(Path(".").glob("*.mp4"))
        if not available:
            print(f"[ERROR] No video found. Please specify with --input")
            sys.exit(1)
        # Pick smallest for quick benchmarking
        input_path = str(sorted(available, key=lambda p: p.stat().st_size)[0])
        print(f"[INFO] Using video: {input_path}")

    # ── Run Benchmark ────────────────────────────────────────────
    results, meta = run_benchmark(input_path, max_frames=args.frames)

    # ── Save JSON ────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=4)
    print(f"\n  ✓ comparison_results.json")

    # ── Report ───────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "comparison_report.txt")
    print(f"\n[REPORT] Generating text report...")
    generate_report(results, meta, report_path)

    # ── Charts ───────────────────────────────────────────────────
    print(f"\n[CHARTS] Generating {CHARTS_DIR}...")
    plot_filesize(results, meta,          os.path.join(CHARTS_DIR, "bar_filesize.png"))
    plot_compression_ratio(results,       os.path.join(CHARTS_DIR, "bar_compression_ratio.png"))
    plot_ssim(results,                    os.path.join(CHARTS_DIR, "bar_ssim.png"))
    plot_psnr(results,                    os.path.join(CHARTS_DIR, "bar_psnr.png"))
    plot_encoding_time(results,           os.path.join(CHARTS_DIR, "bar_encoding_time.png"))
    plot_quality_vs_ratio(results,        os.path.join(CHARTS_DIR, "scatter_quality_vs_ratio.png"))
    plot_target_comparison(results,       os.path.join(CHARTS_DIR, "target_comparison.png"))
    plot_radar(results,                   os.path.join(CHARTS_DIR, "radar_method_profile.png"))
    plot_heatmap(results,                 os.path.join(CHARTS_DIR, "heatmap_metrics.png"))
    plot_flowchart_proposed(              os.path.join(CHARTS_DIR, "flowchart_proposed.png"))
    plot_flowchart_standard(              os.path.join(CHARTS_DIR, "flowchart_standard.png"))
    plot_dashboard(results, meta,         os.path.join(CHARTS_DIR, "combined_dashboard.png"))

    # ── Final Summary ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"  ├── comparison_results.json")
    print(f"  ├── comparison_report.txt")
    print(f"  └── charts/  ({len(os.listdir(CHARTS_DIR))} charts generated)")
    for f in sorted(os.listdir(CHARTS_DIR)):
        sz = os.path.getsize(os.path.join(CHARTS_DIR, f)) / 1024
        print(f"      ├── {f}  ({sz:.0f} KB)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
