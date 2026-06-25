"""
proposed_compressor.py
======================
Text-Aware Video Compression — Proposed Algorithm.

Core Idea:
  1. Detect text regions in each frame using CRAFT + RefineNet.
  2. Build a quadtree on the frame — but NEVER split a quadrant
     that contains any text pixel.  Text regions stay as one block.
  3. Non-text quadrants keep splitting until:
       - Block variance ≈ 0  (single colour)
       - Block size reaches 8×8 pixels
  4. Text blocks   → DCT with high quality (preserves sharpness).
  5. Background blocks → DCT with low quality  OR  mean-fill if uniform.
  6. Temporal filter: SSIM-based frame skipping (skip frames that haven't
     changed enough to justify reprocessing).

target_ssim drives everything:
  - Higher ssim_target → gentler bg_quality  + stricter temporal threshold
  - Lower  ssim_target → aggressive bg_quality + looser  temporal threshold
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.fft import dctn, idctn
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# Add parent directory so we can import base_compressor
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.base_compressor import BaseCompressor, CompressResult

# ── CRAFT imports (from existing project) ─────────────────────────
try:
    from cutils_mod import (
        load_craftnet_model,
        load_refinenet_model,
        get_prediction,
    )
    CRAFT_AVAILABLE = True
except ImportError:
    CRAFT_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────
MIN_BLOCK   = 8          # Minimum quadtree block size (pixels)
MAX_DEPTH   = 8          # Maximum recursion depth
VAR_THRESH  = 50.0       # Variance below this = single colour → stop splitting


# ══════════════════════════════════════════════════════════════════
# Quadtree (text-region-aware)
# ══════════════════════════════════════════════════════════════════

def _qt_partition(gray: np.ndarray,
                  text_mask: np.ndarray,
                  x: int, y: int, w: int, h: int,
                  depth: int = 0):
    """
    Recursively partition [y:h, x:w] of `gray`.

    Returns list of tuples:
        (x, y, w, h, depth, block_type)
        block_type ∈ {"text", "uniform_bg", "bg"}
    """
    bh, bw = h - y, w - x
    if bh <= 0 or bw <= 0:
        return []

    # ── Rule 1: if ANY pixel in this block is text → keep whole block ──
    if np.any(text_mask[y:h, x:w] > 0):
        return [(x, y, w, h, depth, "text")]

    block    = gray[y:h, x:w].astype(np.float32)
    variance = float(np.var(block))

    # ── Rule 2: single colour or minimum size → leaf background block ──
    if bh <= MIN_BLOCK or bw <= MIN_BLOCK or depth >= MAX_DEPTH or variance <= VAR_THRESH:
        btype = "uniform_bg" if variance <= VAR_THRESH else "bg"
        return [(x, y, w, h, depth, btype)]

    # ── Split into 4 quadrants ─────────────────────────────────────
    mx, my = x + bw // 2, y + bh // 2
    leaves = []
    leaves += _qt_partition(gray, text_mask, x,  y,  mx, my, depth + 1)
    leaves += _qt_partition(gray, text_mask, mx, y,  w,  my, depth + 1)
    leaves += _qt_partition(gray, text_mask, x,  my, mx, h,  depth + 1)
    leaves += _qt_partition(gray, text_mask, mx, my, w,  h,  depth + 1)
    return leaves


# ══════════════════════════════════════════════════════════════════
# DCT helpers
# ══════════════════════════════════════════════════════════════════

def _quality_to_step(quality: int) -> float:
    q = max(1, min(100, quality))
    step = (5000 / q) if q < 50 else (200 - 2 * q)
    return max(1.0, step)


def _dct_compress(block_f: np.ndarray, quality: int) -> np.ndarray:
    step   = _quality_to_step(quality)
    coeffs = dctn(block_f, type=2, norm="ortho")
    quant  = np.round(coeffs / step) * step
    return np.clip(idctn(quant, type=2, norm="ortho"), 0, 255)


# ══════════════════════════════════════════════════════════════════
# SSIM → parameter mapping
# ══════════════════════════════════════════════════════════════════

def _ssim_to_params(target_ssim: float):
    """
    Map target_ssim (0–1) to compression parameters.
    Higher ssim → better quality (gentler compression).
    """
    t = float(np.clip(target_ssim, 0.5, 1.0))

    # bg_quality:       ssim 0.5 → Q=10,  ssim 1.0 → Q=90
    bg_quality = int(10 + (t - 0.5) / 0.5 * 80)

    # ssim_threshold:   ssim 0.5 → 0.80,  ssim 1.0 → 0.99
    ssim_threshold = 0.80 + (t - 0.5) / 0.5 * 0.19

    text_quality = 90   # Text quality is always fixed — never compress text
    return bg_quality, text_quality, ssim_threshold


# ══════════════════════════════════════════════════════════════════
# Frame compression
# ══════════════════════════════════════════════════════════════════

def _compress_frame(frame_bgr:   np.ndarray,
                    text_mask:   np.ndarray,
                    bg_quality:  int,
                    text_quality: int) -> tuple:
    """
    Compress one frame using text-aware quadtree + DCT.
    Returns (compressed_frame, block_stats).
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    out  = np.zeros_like(frame_bgr, dtype=np.float32)

    blocks   = _qt_partition(gray, text_mask, 0, 0, w, h)
    stats    = {"text": 0, "bg": 0, "uniform_bg": 0}

    for (bx, by, bw, bh, depth, btype) in blocks:
        stats[btype] = stats.get(btype, 0) + 1

        if btype == "uniform_bg" and (bh - by) >= 4 and (bw - bx) >= 4:
            # Single colour → fill with average (cheapest, best ratio)
            for c in range(3):
                out[by:bh, bx:bw, c] = float(
                    np.mean(frame_bgr[by:bh, bx:bw, c]))

        elif btype == "text":
            for c in range(3):
                out[by:bh, bx:bw, c] = _dct_compress(
                    frame_bgr[by:bh, bx:bw, c].astype(np.float32),
                    text_quality)
        else:  # regular background
            for c in range(3):
                out[by:bh, bx:bw, c] = _dct_compress(
                    frame_bgr[by:bh, bx:bw, c].astype(np.float32),
                    bg_quality)

    return np.clip(out, 0, 255).astype(np.uint8), stats


# ══════════════════════════════════════════════════════════════════
# Quality measurement helper
# ══════════════════════════════════════════════════════════════════

def _measure_quality(orig_path: str, comp_path: str,
                     n_frames: int = 30) -> tuple:
    cap_o = cv2.VideoCapture(orig_path)
    cap_c = cv2.VideoCapture(comp_path)
    total = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0),
                          min(n_frames, total), dtype=int)
    psnr_vals, ssim_vals = [], []
    for idx in indices:
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_c.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ro, fo = cap_o.read()
        rc, fc = cap_c.read()
        if not (ro and rc):
            continue
        if fo.shape != fc.shape:
            fc = cv2.resize(fc, (fo.shape[1], fo.shape[0]))
        go = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY)
        gc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        try:
            psnr_vals.append(psnr_fn(go, gc, data_range=255))
        except Exception:
            pass
        try:
            sv, _ = ssim_fn(go, gc, full=True)
            ssim_vals.append(float(sv))
        except Exception:
            pass
    cap_o.release()
    cap_c.release()
    psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    return psnr, ssim


# ══════════════════════════════════════════════════════════════════
# ProposedCompressor — main class
# ══════════════════════════════════════════════════════════════════

class ProposedCompressor(BaseCompressor):
    NAME = "proposed"

    def __init__(self,
                 use_refinenet:    bool = True,
                 text_threshold:   float = 0.85,
                 link_threshold:   float = 0.60,
                 long_size:        int   = 1280):
        super().__init__()
        self.use_refinenet  = use_refinenet
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.long_size      = long_size
        self._craft_net     = None
        self._refine_net    = None
        self._loaded_device = None

    # ── Model loading ───────────────────────────────────────────

    def _load_models(self, device: str):
        if not CRAFT_AVAILABLE:
            print("  [WARN] CRAFT not available — using heuristic text detection.")
            return
        if self._loaded_device == device:
            return  # Already loaded for this device
        use_gpu = device in ("mps", "cuda")
        print(f"  [PROPOSED] Loading CRAFT on {device.upper()}...")
        self._craft_net     = load_craftnet_model(cuda=use_gpu)
        self._refine_net    = load_refinenet_model(cuda=use_gpu) if self.use_refinenet else None
        self._loaded_device = device

    # ── Text detection ──────────────────────────────────────────

    def _detect_text(self, frame_bgr: np.ndarray, device: str):
        """Returns binary text mask (255=text, 0=bg)."""
        h, w = frame_bgr.shape[:2]
        mask  = np.zeros((h, w), dtype=np.uint8)

        if CRAFT_AVAILABLE and self._craft_net is not None:
            frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            use_gpu    = device in ("mps", "cuda")
            pred       = get_prediction(
                image          = frame_rgb,
                craft_net      = self._craft_net,
                refine_net     = self._refine_net,
                text_threshold = self.text_threshold,
                link_threshold = self.link_threshold,
                low_text       = 0.4,
                cuda           = use_gpu,
                long_size      = self.long_size,
            )
            boxes = pred["boxes"]
            for box in boxes:
                pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
                pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
                cv2.fillPoly(mask, [pts], 255)
        else:
            # Heuristic: high-gradient regions as proxy for text
            gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            lap      = cv2.Laplacian(gray, cv2.CV_64F)
            lap_abs  = np.abs(lap).astype(np.uint8)
            _, mask  = cv2.threshold(lap_abs, 20, 255, cv2.THRESH_BINARY)

        return mask

    # ── Main compress ───────────────────────────────────────────

    def compress(
        self,
        input_path:  str,
        output_path: str,
        target_ssim: float,
        device:      str = "cpu",
        max_frames:  Optional[int] = None,
    ) -> CompressResult:

        self._load_models(device)
        self.ensure_output_dir(output_path)

        bg_quality, text_quality, ssim_threshold = _ssim_to_params(target_ssim)
        print(f"  [PROPOSED/{device.upper()}] target_ssim={target_ssim:.2f} → "
              f"bg_q={bg_quality}, text_q={text_quality}, "
              f"ssim_thresh={ssim_threshold:.3f}")

        cap = cv2.VideoCapture(input_path)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

        ref_frame   = None
        last_comp   = None
        skipped     = 0
        processed   = 0
        frame_idx   = 0
        t0          = time.time()

        while cap.isOpened() and frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ── Temporal filter ───────────────────────────────
            if ref_frame is not None:
                rg = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                fg = cv2.cvtColor(frame,     cv2.COLOR_BGR2GRAY)
                sim, _ = ssim_fn(rg, fg, full=True)
                if float(sim) >= ssim_threshold:
                    writer.write(last_comp if last_comp is not None else frame)
                    skipped += 1
                    continue

            # ── Text detection + quadtree + DCT ───────────────
            text_mask             = self._detect_text(frame, device)
            comp_frame, _bstats  = _compress_frame(
                frame, text_mask, bg_quality, text_quality)

            writer.write(comp_frame)
            ref_frame  = frame
            last_comp  = comp_frame
            processed += 1

        enc_time = time.time() - t0
        cap.release()
        writer.release()

        # ── Metrics ───────────────────────────────────────────
        orig_mb = self.file_size_mb(input_path)
        comp_mb = self.file_size_mb(output_path)
        space_saved  = (1 - comp_mb / orig_mb) * 100 if orig_mb > 0 else 0
        comp_ratio   = orig_mb / comp_mb if comp_mb > 0 else 0
        psnr, ssim_v = _measure_quality(input_path, output_path)

        return CompressResult(
            algorithm          = self.NAME,
            device             = device,
            input_path         = input_path,
            output_path        = output_path,
            target_ssim        = target_ssim,
            original_size_mb   = orig_mb,
            compressed_size_mb = comp_mb,
            space_saved_pct    = space_saved,
            compression_ratio  = comp_ratio,
            achieved_ssim      = ssim_v,
            achieved_psnr      = psnr,
            encode_time_s      = enc_time,
            frames_total       = frame_idx,
            frames_skipped     = skipped,
            frames_processed   = processed,
            extra              = {
                "bg_quality":      bg_quality,
                "text_quality":    text_quality,
                "ssim_threshold":  ssim_threshold,
                "craft_available": CRAFT_AVAILABLE,
            },
        )
