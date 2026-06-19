"""
vp9_compressor.py
=================
VP9 compression via ffmpeg libvpx-vp9.

target_ssim → CRF mapping:
  ssim 1.0  →  CRF 15
  ssim 0.90 →  CRF 30
  ssim 0.80 →  CRF 40
  ssim 0.70 →  CRF 48
  ssim 0.50 →  CRF 63

device:
  VP9 has no Apple HW encoder in ffmpeg.
  "mps" mode: ffmpeg with -threads 0 (max CPU threads as best available)
  "cpu" mode: ffmpeg with -threads 1 (single-threaded, worst case)
  This gives a meaningful CPU thread-count comparison.
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


def _ssim_to_crf_vp9(target_ssim: float) -> int:
    t   = float(np.clip(target_ssim, 0.5, 1.0))
    crf = 63 - int((t - 0.5) / 0.5 * 48)   # 0.5→63,  1.0→15
    return int(np.clip(crf, 0, 63))


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


class VP9Compressor(BaseCompressor):
    NAME = "vp9"

    def compress(
        self,
        input_path:  str,
        output_path: str,
        target_ssim: float,
        device:      str = "cpu",
        max_frames:  Optional[int] = None,
    ) -> CompressResult:

        self.ensure_output_dir(output_path)
        crf     = _ssim_to_crf_vp9(target_ssim)
        vf      = f"-vframes {max_frames}" if max_frames else ""
        threads = "0" if device in ("mps", "videotoolbox") else "1"
        dev_label = "multithreaded" if threads == "0" else "singlethread"

        # VP9 two-pass would be ideal but slow — use single pass constrained
        cmd = (
            f'ffmpeg -y -i "{input_path}" {vf} '
            f'-c:v libvpx-vp9 -crf {crf} -b:v 0 '
            f'-threads {threads} -an '
            f'"{output_path}" 2>&1'
        )

        print(f"  [VP9/{dev_label.upper()}] target_ssim={target_ssim:.2f} → CRF={crf}, threads={threads}")
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
            extra              = {"crf": crf, "threads": threads},
        )
