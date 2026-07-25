"""
benchmark_runner.py
===================
Orchestrates the full benchmark:
  - Runs all 5 algorithms × 2 devices (MPS + CPU)
  - Saves each output to benchmark_output/<algorithm>/<device>/
  - Collects all CompressResult objects
  - Returns structured results dict for the chart generator
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# ── Algorithm imports ──────────────────────────────────────────────
from algorithms.proposed.proposed_compressor    import ProposedCompressor
from algorithms.h264.h264_compressor            import H264Compressor
from algorithms.h265.h265_compressor            import H265Compressor
from algorithms.vp9.vp9_compressor              import VP9Compressor
from algorithms.ssim_driven.ssim_driven_compressor import SSIMDrivenCompressor
from algorithms.base_compressor                 import CompressResult


# ══════════════════════════════════════════════════════════════════
# Device helper
# ══════════════════════════════════════════════════════════════════

def detect_mps() -> bool:
    """
    Returns True if Apple Silicon GPU acceleration is available.
    Uses platform check first (works even when torch is broken).
    """
    import platform, subprocess
    if platform.processor() == "arm" or platform.machine() == "arm64":
        return True   # Apple Silicon — VideoToolbox + MPS available
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# BenchmarkRunner
# ══════════════════════════════════════════════════════════════════

class BenchmarkRunner:

    ALGORITHMS = {
        "proposed":    ProposedCompressor,
        "h264":        H264Compressor,
        "h265":        H265Compressor,
        "vp9":         VP9Compressor,
        "ssim_driven": SSIMDrivenCompressor,
    }

    def __init__(self,
                 input_path:   str,
                 target_ssim:  float = 0.90,
                 max_frames:   Optional[int] = 120,
                 output_dir:   str = "benchmark_output",
                 algorithms:   Optional[List[str]] = None,
                 run_mps:      bool = True,
                 run_cpu:      bool = True):

        self.input_path  = input_path
        self.target_ssim = target_ssim
        self.max_frames  = max_frames
        self.output_dir  = output_dir
        self.algo_names  = algorithms or list(self.ALGORITHMS.keys())
        self.mps_avail   = detect_mps()
        # Honour user intent — but warn if MPS isn't detected
        self.run_mps     = run_mps
        self.run_cpu     = run_cpu
        if run_mps and not self.mps_avail:
            print("  [WARN] MPS not detected via torch, but Apple Silicon found — "
                  "using VideoToolbox for hardware acceleration.")

    def _out_path(self, algo: str, device: str) -> str:
        stem = Path(self.input_path).stem
        d    = os.path.join(self.output_dir, algo, device)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{stem}_compressed.mp4")

    def run(self) -> dict:
        video_name = Path(self.input_path).stem
        devices    = []
        if self.run_mps:  devices.append("mps")
        if self.run_cpu:  devices.append("cpu")

        if not devices:
            raise ValueError("At least one of run_mps / run_cpu must be True.")

        print("\n" + "=" * 70)
        print(f"BENCHMARK — {video_name}")
        print(f"  target_ssim  : {self.target_ssim}")
        print(f"  max_frames   : {self.max_frames}")
        print(f"  algorithms   : {self.algo_names}")
        print(f"  devices      : {devices}")
        print(f"  MPS available: {self.mps_avail}")
        print("=" * 70)

        results  = {}   # key: f"{algo}_{device}"  value: CompressResult
        meta     = {
            "video_name":   video_name,
            "input_path":   self.input_path,
            "target_ssim":  self.target_ssim,
            "max_frames":   self.max_frames,
            "mps_available": self.mps_avail,
            "generated":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        total_runs = len(self.algo_names) * len(devices)
        run_idx    = 0

        for algo_name in self.algo_names:
            if algo_name not in self.ALGORITHMS:
                print(f"  [WARN] Unknown algorithm '{algo_name}' — skipped.")
                continue

            compressor_cls = self.ALGORITHMS[algo_name]
            compressor     = compressor_cls()

            for device in devices:
                run_idx += 1
                key     = f"{algo_name}_{device}"
                out_p   = self._out_path(algo_name, device)

                print(f"\n[{run_idx}/{total_runs}] {algo_name.upper()} on {device.upper()}")
                print(f"  Output: {out_p}")

                try:
                    result = compressor.compress(
                        input_path  = self.input_path,
                        output_path = out_p,
                        target_ssim = self.target_ssim,
                        device      = device,
                        max_frames  = self.max_frames,
                    )
                    results[key] = result
                    print(f"  Done: size={result.compressed_size_mb:.2f}MB  "
                          f"saved={result.space_saved_pct:.1f}%  "
                          f"SSIM={result.achieved_ssim:.4f}  "
                          f"PSNR={result.achieved_psnr:.1f}dB  "
                          f"time={result.encode_time_s:.1f}s")

                except Exception as e:
                    import traceback
                    print(f"  [ERROR] {e}")
                    traceback.print_exc()

        # ── Compute MPS speedup for each algorithm ─────────────
        speedups = {}
        for algo_name in self.algo_names:
            mps_key = f"{algo_name}_mps"
            cpu_key = f"{algo_name}_cpu"
            if mps_key in results and cpu_key in results:
                mps_t = results[mps_key].encode_time_s
                cpu_t = results[cpu_key].encode_time_s
                speedups[algo_name] = cpu_t / mps_t if mps_t > 0 else 1.0

        # ── Save JSON (merge with existing results) ────────────
        json_out  = os.path.join(self.output_dir, "benchmark_results.json")
        existing_results = {}
        if os.path.exists(json_out):
            try:
                with open(json_out) as ef:
                    existing_data = json.load(ef)
                existing_results = existing_data.get("results", {})
                print(f"\n  Merging with {len(existing_results)} existing results...")
            except Exception:
                pass
        # Merge: new results override same key, old results kept
        merged = {**existing_results, **{k: v.to_dict() for k, v in results.items()}}
        # Recompute speedups from merged
        merged_speedups = {}
        for aname in set(k.rsplit("_", 1)[0] for k in merged):
            cpu_k = next((k for k in merged if k.startswith(aname) and
                          merged[k].get("device","") in ("cpu","singlethread")), None)
            hw_k  = next((k for k in merged if k.startswith(aname) and
                          merged[k].get("device","") in ("mps","videotoolbox","multithreaded")), None)
            if cpu_k and hw_k:
                ct = merged[cpu_k].get("encode_time_s", 1)
                ht = merged[hw_k].get("encode_time_s", 1)
                merged_speedups[aname] = round(ct / max(ht, 0.001), 3)
        json_data = {
            "meta":     meta,
            "results":  merged,
            "speedups": merged_speedups,
        }
        os.makedirs(self.output_dir, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"\n  Saved: {json_out}")

        return {"results": results, "speedups": speedups, "meta": meta}
