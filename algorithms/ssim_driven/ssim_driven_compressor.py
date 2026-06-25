"""
ssim_driven_compressor.py
=========================
SSIM-Driven H.264 — the external algorithm that also uses SSIM as its
compression control signal (just like our proposed method).

Algorithm:
  Binary search on H.264 CRF until the compressed video achieves an SSIM
  within tolerance of the user's target_ssim input.

  lo_crf = 0,  hi_crf = 51
  while iterations < max_iter and |achieved - target| > tolerance:
      mid_crf = (lo + hi) / 2
      compress clip with CRF = mid_crf
      achieved_ssim = measure_ssim(original, compressed)
      if achieved_ssim > target:  lo = mid   (too good → more compression)
      else:                       hi = mid   (too bad  → less compression)

This is the "fair fight" comparison algorithm:
  Both our proposed method AND this algorithm accept target_ssim as input.
  The difference is that our method also preserves text regions explicitly.

device:
  "mps"/"videotoolbox" → h264_videotoolbox (Apple HW)
  "cpu"                → libx264 (software)
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.base_compressor import BaseCompressor, CompressResult


def _quick_ssim(orig_path: str, comp_path: str, n: int = 15) -> float:
    """Fast SSIM estimate on n uniformly-sampled frames."""
    cap_o = cv2.VideoCapture(orig_path)
    cap_c = cv2.VideoCapture(comp_path)
    total = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, max(total-1, 0), min(n, total), dtype=int)
    vals  = []
    for i in idxs:
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_c.set(cv2.CAP_PROP_POS_FRAMES, i)
        ro, fo = cap_o.read(); rc, fc = cap_c.read()
        if not (ro and rc): continue
        if fo.shape != fc.shape:
            fc = cv2.resize(fc, (fo.shape[1], fo.shape[0]))
        go = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY)
        gc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        try:
            sv, _ = ssim_fn(go, gc, full=True)
            vals.append(float(sv))
        except: pass
    cap_o.release(); cap_c.release()
    return float(np.mean(vals)) if vals else 0.0


def _full_quality(orig_path: str, comp_path: str, n: int = 30):
    cap_o = cv2.VideoCapture(orig_path)
    cap_c = cv2.VideoCapture(comp_path)
    total = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, max(total-1, 0), min(n, total), dtype=int)
    ps, ss = [], []
    for i in idxs:
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_c.set(cv2.CAP_PROP_POS_FRAMES, i)
        ro, fo = cap_o.read(); rc, fc = cap_c.read()
        if not (ro and rc): continue
        if fo.shape != fc.shape:
            fc = cv2.resize(fc, (fo.shape[1], fo.shape[0]))
        go = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY)
        gc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        try: ps.append(psnr_fn(go, gc, data_range=255))
        except: pass
        try:
            sv, _ = ssim_fn(go, gc, full=True)
            ss.append(float(sv))
        except: pass
    cap_o.release(); cap_c.release()
    return (float(np.mean(ps)) if ps else 0.0,
            float(np.mean(ss)) if ss else 0.0)


def _run_ffmpeg(input_path, output_path, crf, max_frames, use_hw):
    vf  = f"-vframes {max_frames}" if max_frames else ""
    if use_hw:
        cmd = (f'ffmpeg -y -hwaccel videotoolbox -i "{input_path}" {vf} '
               f'-c:v h264_videotoolbox -q:v {max(1, crf//2)} -an '
               f'"{output_path}" 2>&1')
    else:
        cmd = (f'ffmpeg -y -i "{input_path}" {vf} '
               f'-c:v libx264 -crf {crf} -preset fast -an '
               f'"{output_path}" 2>&1')
    subprocess.run(cmd, shell=True, capture_output=True)


class SSIMDrivenCompressor(BaseCompressor):
    """
    H.264 with binary-search CRF to hit target SSIM.
    This is the 'fair comparison' external SSIM-driven codec.
    """
    NAME = "ssim_driven"

    def __init__(self,
                 tolerance:  float = 0.01,
                 max_iter:   int   = 8,
                 probe_frames: int = 60):
        super().__init__()
        self.tolerance    = tolerance
        self.max_iter     = max_iter
        self.probe_frames = probe_frames  # frames used during binary search

    def compress(
        self,
        input_path:  str,
        output_path: str,
        target_ssim: float,
        device:      str = "cpu",
        max_frames:  Optional[int] = None,
    ) -> CompressResult:

        self.ensure_output_dir(output_path)
        use_hw    = device in ("mps", "videotoolbox")
        dev_label = "videotoolbox" if use_hw else "cpu"

        print(f"  [SSIM_DRIVEN/{dev_label.upper()}] "
              f"Binary searching CRF for target_ssim={target_ssim:.3f} ...")

        # ── Phase 1: Binary search on probe clip ───────────────
        lo, hi    = 0, 51
        best_crf  = 26  # fallback
        iters     = 0
        total_enc = 0.0

        with tempfile.TemporaryDirectory() as tmp:
            probe_path = os.path.join(tmp, "probe.mp4")

            while iters < self.max_iter and (hi - lo) > 1:
                mid = (lo + hi) // 2
                t0  = time.time()
                _run_ffmpeg(input_path, probe_path, mid,
                            self.probe_frames, use_hw)
                total_enc += time.time() - t0

                if not os.path.exists(probe_path) or os.path.getsize(probe_path) == 0:
                    lo = mid  # encoding failed — assume over-compressed
                    iters += 1
                    continue

                achieved = _quick_ssim(input_path, probe_path)
                print(f"    iter {iters+1}: CRF={mid}  achieved_ssim={achieved:.4f}")

                if achieved >= target_ssim:
                    lo = mid  # can afford more compression
                else:
                    hi = mid  # need less compression

                best_crf = lo
                iters   += 1

        print(f"  [SSIM_DRIVEN] Found optimal CRF={best_crf}")

        # ── Phase 2: Final encode at optimal CRF ──────────────
        t0 = time.time()
        _run_ffmpeg(input_path, output_path, best_crf, max_frames, use_hw)
        final_enc = time.time() - t0
        enc_time  = total_enc + final_enc

        orig_mb = self.file_size_mb(input_path)
        comp_mb = self.file_size_mb(output_path)
        space   = (1 - comp_mb / orig_mb) * 100 if orig_mb > 0 else 0
        ratio   = orig_mb / comp_mb if comp_mb > 0 else 0

        cap = cv2.VideoCapture(input_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        psnr, ssim_v = _full_quality(input_path, output_path) if os.path.exists(output_path) else (0.0, 0.0)

        return CompressResult(
            algorithm          = self.NAME,
            device             = dev_label,
            input_path         = input_path,
            output_path        = output_path,
            target_ssim        = target_ssim,
            original_size_mb   = orig_mb,
            compressed_size_mb = comp_mb,
            space_saved_pct    = space,
            compression_ratio  = ratio,
            achieved_ssim      = ssim_v,
            achieved_psnr      = psnr,
            encode_time_s      = enc_time,
            frames_total       = n_frames,
            extra              = {
                "optimal_crf":  best_crf,
                "search_iters": iters,
                "probe_enc_s":  total_enc,
                "final_enc_s":  final_enc,
            },
        )
