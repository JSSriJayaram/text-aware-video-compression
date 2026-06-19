"""
h264_compressor.py
==================
H.264 / AVC compression via ffmpeg libx264.

target_ssim → CRF mapping:
  ssim 1.0  →  CRF 18  (near-lossless)
  ssim 0.90 →  CRF 26
  ssim 0.80 →  CRF 33
  ssim 0.70 →  CRF 39
  ssim 0.50 →  CRF 51  (worst quality)

device:
  "mps" / "videotoolbox"  → ffmpeg -hwaccel videotoolbox -c:v h264_videotoolbox
  "cpu"                   → ffmpeg -c:v libx264 (software)
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.base_compressor import BaseCompressor, CompressResult


def _ssim_to_crf(target_ssim: float) -> int:
    """Map target SSIM (0.5–1.0) to H.264 CRF (51–18). Linear mapping."""
    t   = float(np.clip(target_ssim, 0.5, 1.0))
    crf = 51 - int((t - 0.5) / 0.5 * 33)   # 0.5→51,  1.0→18
    return int(np.clip(crf, 0, 51))


def _measure_quality(orig_path, comp_path, n=30):
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


class H264Compressor(BaseCompressor):
    NAME = "h264"

    def compress(
        self,
        input_path:  str,
        output_path: str,
        target_ssim: float,
        device:      str = "cpu",
        max_frames:  Optional[int] = None,
    ) -> CompressResult:

        self.ensure_output_dir(output_path)
        crf = _ssim_to_crf(target_ssim)
        vf  = f"-vframes {max_frames}" if max_frames else ""

        # Build ffmpeg command depending on device
        if device in ("mps", "videotoolbox"):
            # Apple HW encoder via VideoToolbox
            cmd = (
                f'ffmpeg -y -hwaccel videotoolbox -i "{input_path}" {vf} '
                f'-c:v h264_videotoolbox -q:v {max(1, crf//2)} -an '
                f'"{output_path}" 2>&1'
            )
            dev_label = "videotoolbox"
        else:
            cmd = (
                f'ffmpeg -y -i "{input_path}" {vf} '
                f'-c:v libx264 -crf {crf} -preset fast -an '
                f'"{output_path}" 2>&1'
            )
            dev_label = "cpu"

        print(f"  [H264/{dev_label.upper()}] target_ssim={target_ssim:.2f} → CRF={crf}")
        t0  = time.time()
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        enc = time.time() - t0

        orig_mb = self.file_size_mb(input_path)
        comp_mb = self.file_size_mb(output_path)
        space   = (1 - comp_mb / orig_mb) * 100 if orig_mb > 0 else 0
        ratio   = orig_mb / comp_mb if comp_mb > 0 else 0

        cap = cv2.VideoCapture(input_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        psnr, ssim_v = _measure_quality(input_path, output_path) if os.path.exists(output_path) else (0.0, 0.0)

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
            encode_time_s      = enc,
            frames_total       = n_frames,
            extra              = {"crf": crf, "ffmpeg_device": dev_label},
        )
