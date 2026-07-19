"""
generate_final_report.py
========================
Merges CPU + MPS benchmark results from benchmark_output/benchmark_results.json,
runs the optimizer, and generates the complete final chart set + text report.

Run after both benchmark passes (CPU + MPS) are complete:
    ./craft_env/bin/python3 generate_final_report.py
"""

import os
import sys
import json
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTPUT_DIR = "benchmark_output"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
REPORT_PATH = os.path.join(OUTPUT_DIR, "final_report.txt")
os.makedirs(CHARTS_DIR, exist_ok=True)

ALG_COLORS = {
    "proposed":    "#E91E63",
    "h264":        "#2196F3",
    "h265":        "#4CAF50",
    "vp9":         "#FF9800",
    "ssim_driven": "#9C27B0",
}

# ─────────────────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────────────────

def load_results():
    path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(path) as f:
        data = json.load(f)
    return data["meta"], data["results"], data.get("speedups", {})


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _color(key):
    for name, c in ALG_COLORS.items():
        if name in key:
            return c
    return "#aaaaaa"

def _label(key):
    parts = key.split("_")
    alg   = parts[0].upper()
    dev   = "_".join(parts[1:]).upper() if len(parts) > 1 else ""
    dev_short = {"CPU": "CPU", "MPS": "HW", "VIDEOTOOLBOX": "HW",
                 "MULTITHREADED": "MT", "SINGLETHREAD": "ST"}.get(dev, dev)
    return f"{alg}\n({dev_short})"

def _ax_dark(ax, title="", ylabel=""):
    ax.set_facecolor("#111827")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#333")
    ax.grid(axis="y", color="#333", alpha=0.4, zorder=0)
    if title:
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color="white", fontsize=10)


# ─────────────────────────────────────────────────────────
# Compute speedups from loaded results
# ─────────────────────────────────────────────────────────

def compute_speedups(results):
    speedups = {}
    algos = set(k.rsplit("_", 1)[0] for k in results)
    for alg in algos:
        # Find cpu key and mps/hw key
        cpu_key = None
        hw_key  = None
        for k in results:
            if k.startswith(alg):
                dev = results[k].get("device","")
                if dev in ("cpu", "singlethread"):
                    cpu_key = k
                elif dev in ("mps", "videotoolbox", "multithreaded"):
                    hw_key  = k
        if cpu_key and hw_key:
            cpu_t = results[cpu_key].get("encode_time_s", 1)
            hw_t  = results[hw_key].get("encode_time_s", 1)
            speedups[alg] = cpu_t / max(hw_t, 0.001)
    return speedups


# ─────────────────────────────────────────────────────────
# Chart 1 — Grouped CPU vs MPS bar chart for all metrics
# ─────────────────────────────────────────────────────────

def plot_grouped_cpu_vs_hw(results, meta, charts_dir):
    """Side-by-side CPU vs HW bars for every algorithm."""
    algos  = sorted(set(k.rsplit("_", 1)[0] for k in results))
    metrics = [
        ("compressed_size_mb",  "File Size (MB)",         False),
        ("space_saved_pct",     "Space Saved (%)",        True),
        ("achieved_ssim",       "SSIM",                   True),
        ("achieved_psnr",       "PSNR (dB)",              True),
        ("encode_time_s",       "Encode Time (s)",        False),
        ("compression_ratio",   "Compression Ratio (x)",  True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(
        f"CPU vs Hardware Acceleration — All Metrics\n"
        f"Video: {meta.get('video_name','')}  |  target_ssim={meta.get('target_ssim','')}",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )

    for idx, (metric, ylabel, hib) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        _ax_dark(ax, ylabel, f"{'Higher' if hib else 'Lower'} is Better")

        x        = np.arange(len(algos))
        cpu_vals = []
        hw_vals  = []
        for alg in algos:
            cpu_k = next((k for k in results if k.startswith(alg) and
                          results[k].get("device","") in ("cpu","singlethread")), None)
            hw_k  = next((k for k in results if k.startswith(alg) and
                          results[k].get("device","") in ("mps","videotoolbox","multithreaded")), None)
            cpu_vals.append(results[cpu_k].get(metric, 0) if cpu_k else 0)
            hw_vals.append(results[hw_k].get(metric, 0)   if hw_k  else 0)

        w = 0.35
        b1 = ax.bar(x - w/2, cpu_vals, width=w, label="CPU",
                    color=[_color(a+"_cpu") for a in algos],
                    edgecolor="#333", linewidth=0.6, alpha=0.85, zorder=3)
        b2 = ax.bar(x + w/2, hw_vals,  width=w, label="HW (VideoToolbox/MPS)",
                    color=[_color(a+"_mps") for a in algos],
                    edgecolor="white", linewidth=0.6, alpha=1.0,
                    hatch="///", zorder=3)

        for bar, v in zip(list(b1)+list(b2), cpu_vals+hw_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v * 1.02,
                        f"{v:.2f}" if v < 10 else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7,
                        color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos],
                           color="white", fontsize=9)

    # Shared legend
    leg_patches = [
        mpatches.Patch(facecolor="#555", label="CPU (solid)"),
        mpatches.Patch(facecolor="#aaa", hatch="///", label="HW / VideoToolbox (hatched)"),
    ]
    fig.legend(handles=leg_patches, loc="upper right",
               facecolor="#1a1a2e", edgecolor="#555",
               labelcolor="white", fontsize=10)

    plt.tight_layout()
    path = os.path.join(charts_dir, "grouped_cpu_vs_hw.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  Saved: grouped_cpu_vs_hw.png")


# ─────────────────────────────────────────────────────────
# Chart 2 — MPS Speedup bar chart
# ─────────────────────────────────────────────────────────

def plot_speedup(speedups, charts_dir):
    if not speedups:
        print("  [SKIP] No speedup data (need both CPU and HW results).")
        return
    algos = sorted(speedups.keys())
    vals  = [speedups[a] for a in algos]
    cols  = [_color(a)   for a in algos]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F1117")
    _ax_dark(ax, "Hardware (VideoToolbox/MPS) Speedup over Software CPU",
             "Speedup Factor (x)")

    x    = np.arange(len(algos))
    bars = ax.bar(x, vals, color=cols, width=0.5,
                  edgecolor="#333", linewidth=0.8, zorder=3)
    ax.axhline(1.0, color="#FFD700", linestyle="--",
               linewidth=1.8, label="1× (no speedup)", zorder=4)
    ax.axhline(2.0, color="#4CAF50", linestyle=":",
               linewidth=1.2, label="2× speedup", zorder=4, alpha=0.7)

    for bar, v in zip(bars, vals):
        clr = "#66BB6A" if v >= 2 else ("#FFD700" if v >= 1 else "#EF5350")
        ax.text(bar.get_x() + bar.get_width()/2,
                v + 0.03, f"{v:.2f}×",
                ha="center", va="bottom",
                fontsize=12, color=clr, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos],
                       color="white", fontsize=11)
    ax.legend(facecolor="#1a1a2e", labelcolor="white",
              edgecolor="#555", fontsize=10)

    plt.tight_layout()
    path = os.path.join(charts_dir, "bar_mps_speedup.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  Saved: bar_mps_speedup.png")


# ─────────────────────────────────────────────────────────
# Chart 3 — Full combined dashboard (all results)
# ─────────────────────────────────────────────────────────

def plot_full_dashboard(results, speedups, meta, charts_dir):
    keys   = sorted(results.keys())
    labels = [_label(k) for k in keys]
    cols   = [_color(k)  for k in keys]

    def vals(metric):
        return [results[k].get(metric, 0) for k in keys]

    fig = plt.figure(figsize=(26, 20))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.45)

    panels = [
        (gs[0, :2], "compressed_size_mb",  "File Size (MB)",          False),
        (gs[0, 2:], "compression_ratio",   "Compression Ratio (x)",   True),
        (gs[1, :2], "achieved_ssim",       "SSIM Quality",             True),
        (gs[1, 2:], "achieved_psnr",       "PSNR (dB)",               True),
        (gs[2, :2], "encode_time_s",       "Encoding Time (s)",        False),
        (gs[2, 2:], "space_saved_pct",     "Space Saved (%)",          True),
    ]

    x = np.arange(len(keys))
    for spec, metric, ylabel, hib in panels:
        ax = fig.add_subplot(spec)
        _ax_dark(ax, ylabel)
        v  = vals(metric)
        bars = ax.bar(x, v, color=cols, edgecolor="#333",
                      linewidth=0.6, zorder=3)
        for bar, bv in zip(bars, v):
            if bv > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bv * 1.02,
                        f"{bv:.3f}" if bv < 10 else f"{bv:.1f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color="white", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="white", fontsize=7,
                           rotation=30, ha="right")
        ax.set_ylabel(f"{'Higher' if hib else 'Lower'} is better",
                      color="#aaa", fontsize=8)

    if speedups:
        ax_sp = fig.add_axes([0.01, 0.01, 0.24, 0.22])
        ax_sp.set_facecolor("#111827")
        sk = sorted(speedups.keys())
        sv = [speedups[k] for k in sk]
        sc = [_color(k)   for k in sk]
        ax_sp.bar(range(len(sk)), sv, color=sc, edgecolor="#333", zorder=3)
        ax_sp.axhline(1.0, color="#FFD700", linestyle="--", lw=1.5, zorder=4)
        ax_sp.set_xticks(range(len(sk)))
        ax_sp.set_xticklabels([k.upper() for k in sk],
                              color="white", fontsize=8)
        ax_sp.set_title("HW Speedup (x)", color="white",
                        fontsize=10, fontweight="bold")
        ax_sp.tick_params(colors="white", labelsize=7)
        ax_sp.spines[:].set_color("#333")
        ax_sp.grid(axis="y", color="#333", alpha=0.4)

    patches = [mpatches.Patch(color=c, label=n.upper())
               for n, c in ALG_COLORS.items()]
    fig.legend(handles=patches, loc="upper right",
               facecolor="#1a1a2e", edgecolor="#555",
               labelcolor="white", fontsize=10)
    fig.suptitle(
        f"Complete Benchmark Dashboard — {meta.get('video_name','')}  "
        f"| target_ssim={meta.get('target_ssim','')}  "
        f"| frames={meta.get('max_frames','')}  "
        f"| {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )
    path = os.path.join(charts_dir, "final_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  Saved: final_dashboard.png")


# ─────────────────────────────────────────────────────────
# Text Report
# ─────────────────────────────────────────────────────────

def write_report(results, speedups, meta):
    keys = sorted(results.keys())
    with open(REPORT_PATH, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("COMPRESSION ALGORITHM COMPARISON — FINAL REPORT\n")
        f.write(f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video      : {meta.get('video_name','')}\n")
        f.write(f"Input      : {meta.get('input_path','')}\n")
        f.write(f"target_ssim: {meta.get('target_ssim','')}\n")
        f.write(f"Frames     : {meta.get('max_frames','')}\n")
        f.write("=" * 90 + "\n\n")

        # Main table
        hdr = f"{'Method':<30} {'Device':<15} {'Size(MB)':>9} {'Saved%':>8} {'Ratio':>7} {'SSIM':>8} {'PSNR(dB)':>9} {'Time(s)':>8}"
        f.write(hdr + "\n")
        f.write("-" * 90 + "\n")
        for k in keys:
            r = results[k]
            alg = r.get("algorithm","").upper()
            dev = r.get("device","").upper()
            f.write(f"{alg:<30} {dev:<15} "
                    f"{r.get('compressed_size_mb',0):>9.2f} "
                    f"{r.get('space_saved_pct',0):>8.1f} "
                    f"{r.get('compression_ratio',0):>7.2f} "
                    f"{r.get('achieved_ssim',0):>8.4f} "
                    f"{r.get('achieved_psnr',0):>9.2f} "
                    f"{r.get('encode_time_s',0):>8.2f}\n")

        # Speedup section
        if speedups:
            f.write("\n" + "=" * 90 + "\n")
            f.write("HARDWARE vs CPU SPEEDUP\n")
            f.write("=" * 90 + "\n")
            f.write(f"{'Algorithm':<25} {'Speedup':>10}\n")
            f.write("-" * 35 + "\n")
            for alg, sp in sorted(speedups.items()):
                tag = " ← FASTEST" if sp == max(speedups.values()) else ""
                f.write(f"{alg.upper():<25} {sp:>10.2f}x{tag}\n")

        # Optimizer section
        opt_path = os.path.join(OUTPUT_DIR, "optimizer", "optimal_params.json")
        if os.path.exists(opt_path):
            with open(opt_path) as of:
                opt = json.load(of)
            f.write("\n" + "=" * 90 + "\n")
            f.write("OPTIMIZER RESULT (COBYLA — Minimize Time, Maintain Quality)\n")
            f.write("=" * 90 + "\n")
            f.write(f"  Optimal bg_quality    : {opt.get('bg_quality')}\n")
            f.write(f"  Optimal ssim_threshold: {opt.get('ssim_threshold')}\n")
            f.write(f"  text_quality (fixed)  : {opt.get('text_quality')} [TEXT IS NEVER COMPRESSED]\n")
            f.write(f"  Predicted time        : {opt.get('predicted_time_s',0):.3f}s\n")
            f.write(f"  Predicted SSIM        : {opt.get('predicted_ssim',0):.4f}\n")
            f.write(f"  Predicted ratio       : {opt.get('predicted_ratio',0):.2f}x\n")
            f.write(f"  COBYLA success        : {opt.get('cobyla_success')}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write(textwrap.dedent("""
METHODOLOGY
===========
Metrics:
  SSIM  — Structural Similarity (0-1). Perceptual quality: luminance+contrast+structure.
          >0.90 = Excellent | 0.80-0.90 = Good | <0.80 = Degraded
  PSNR  — Peak Signal-to-Noise Ratio (dB). Pixel reconstruction fidelity.
          >35dB = Excellent | 30-35dB = Good | <25dB = Poor

Algorithms:
  Proposed     — Text-Aware DCT + SSIM Temporal Frame Skipping.
                 Quadtree NEVER splits text regions. Text = Q90, BG = Q(f(ssim)).
  H.264 / AVC  — ffmpeg libx264 (CPU) / h264_videotoolbox (Apple HW).
  H.265 / HEVC — ffmpeg libx265 (CPU) / hevc_videotoolbox (Apple HW).
  VP9          — ffmpeg libvpx-vp9. Multithreaded vs Single-threaded comparison.
  SSIM-Driven  — H.264 with binary-search CRF to hit target SSIM exactly.

Hardware Acceleration:
  CPU mode : Software encoder (libx264 / libx265 / libvpx-vp9).
  HW mode  : Apple VideoToolbox encoder (h264_videotoolbox / hevc_videotoolbox).
  Speedup  : cpu_time / hw_time.

Optimizer:
  Method     : COBYLA (Constrained Optimization by Linear Approximations).
  Objective  : Minimize encoding time.
  Constraint : achieved_ssim >= target_ssim.
  Search     : Grid profile (16 combos) → Polynomial surrogate → COBYLA minimize.
"""))

    print(f"  Saved: final_report.txt")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("FINAL REPORT GENERATOR")
    print("=" * 60)

    meta, results, _ = load_results()
    speedups = compute_speedups(results)
    print(f"  Loaded {len(results)} result records")
    print(f"  Speedups: {speedups}")

    print(f"\n[CHARTS] Generating...")
    plot_grouped_cpu_vs_hw(results, meta, CHARTS_DIR)
    plot_speedup(speedups, CHARTS_DIR)
    plot_full_dashboard(results, speedups, meta, CHARTS_DIR)

    print(f"\n[REPORT] Writing...")
    write_report(results, speedups, meta)

    # Also run optimizer if not done
    opt_done = os.path.exists(os.path.join(OUTPUT_DIR, "optimizer", "optimal_params.json"))
    if not opt_done:
        print(f"\n[OPTIMIZER] Running...")
        from optimizer import ProposedOptimizer
        opt_dir   = os.path.join(OUTPUT_DIR, "optimizer")
        optimizer = ProposedOptimizer(
            input_path   = meta["input_path"],
            output_dir   = opt_dir,
            probe_frames = 20,
            grid_points  = 4,
        )
        optimizer.run(target_ssim=meta.get("target_ssim", 0.9))
    else:
        print(f"\n[OPTIMIZER] Already done — skipping grid profiling.")
        from optimizer import ProposedOptimizer
        opt_dir   = os.path.join(OUTPUT_DIR, "optimizer")
        optimizer = ProposedOptimizer(
            input_path   = meta["input_path"],
            output_dir   = opt_dir,
            probe_frames = 20,
            grid_points  = 4,
        )
        # Reload grid data if exists
        grid_path = os.path.join(opt_dir, "grid_profile.json")
        if os.path.exists(grid_path):
            with open(grid_path) as f:
                optimizer.grid_data = json.load(f)
            optimizer.fit_surrogate()
            optimizer.plot_all()

    print(f"\n{'='*60}")
    print("DONE — All outputs in benchmark_output/")
    print(f"{'='*60}")
    n_charts = len(os.listdir(CHARTS_DIR))
    print(f"  Charts  : {CHARTS_DIR}/ ({n_charts} files)")
    print(f"  Report  : {REPORT_PATH}")
    print(f"  JSON    : {OUTPUT_DIR}/benchmark_results.json")
    for fn in sorted(os.listdir(CHARTS_DIR)):
        sz = os.path.getsize(os.path.join(CHARTS_DIR, fn)) // 1024
        print(f"    {fn:45s} {sz:4d} KB")


if __name__ == "__main__":
    main()
