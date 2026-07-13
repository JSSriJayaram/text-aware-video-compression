"""
chart_generator.py
==================
Generates all comparison charts from benchmark results.
All charts use a dark theme and are saved to benchmark_output/charts/.
"""

import os
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from algorithms.base_compressor import CompressResult

# ── Color scheme ──────────────────────────────────────────────────
ALG_COLORS = {
    "proposed":    "#E91E63",   # Pink   — our method
    "h264":        "#2196F3",   # Blue
    "h265":        "#4CAF50",   # Green
    "vp9":         "#FF9800",   # Orange
    "ssim_driven": "#9C27B0",   # Purple
}
DEV_PATTERNS = {"mps": "", "cpu": "///", "videotoolbox": "", "multithreaded": "", "singlethread": "///"}


def _color(key: str) -> str:
    for name, c in ALG_COLORS.items():
        if name in key:
            return c
    return "#aaaaaa"


def _bar_label(key: str) -> str:
    """Turn 'proposed_mps' → 'Proposed\n(MPS)'"""
    parts = key.split("_")
    algo  = parts[0].upper()
    dev   = parts[-1].upper() if len(parts) > 1 else ""
    return f"{algo}\n({dev})"


def _ax_dark(ax, title=""):
    ax.set_facecolor("#111827")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#333")
    ax.grid(axis="y", color="#333", alpha=0.4, zorder=0)
    if title:
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)


# ══════════════════════════════════════════════════════════════════
# Individual charts
# ══════════════════════════════════════════════════════════════════

def _bar_chart(results, output_dir, metric, ylabel, title, filename,
               higher_is_better=True, ref_lines=None):
    keys   = sorted(results.keys())
    vals   = [getattr(results[k], metric, 0) for k in keys]
    labels = [_bar_label(k) for k in keys]
    cols   = [_color(k)     for k in keys]

    fig, ax = plt.subplots(figsize=(max(12, len(keys)*1.4), 7))
    fig.patch.set_facecolor("#0F1117")
    _ax_dark(ax, title)

    x    = np.arange(len(keys))
    bars = ax.bar(x, vals, color=cols, width=0.6, edgecolor="#333",
                  linewidth=0.8, zorder=3)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + max(vals) * 0.01,
                f"{v:.3f}" if v < 10 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=8,
                color="white", fontweight="bold", zorder=6)

    if ref_lines:
        for (rval, rlabel, rcol) in ref_lines:
            ax.axhline(rval, color=rcol, linestyle="--",
                       linewidth=1.5, label=rlabel, alpha=0.9)
        ax.legend(facecolor="#1a1a2e", labelcolor="white",
                  edgecolor="#555", fontsize=9)

    arr = f"Higher is Better" if higher_is_better else "Lower is Better"
    ax.set_ylabel(f"{ylabel}   ({arr})", color="white", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9)

    # Legend: algorithms
    patches = [mpatches.Patch(color=c, label=n.upper())
               for n, c in ALG_COLORS.items()]
    ax.legend(handles=patches, loc="upper right",
              facecolor="#1a1a2e", edgecolor="#555",
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  Saved: {filename}")


def plot_filesize(results, output_dir):
    orig = next(iter(results.values())).original_size_mb
    _bar_chart(results, output_dir,
               "compressed_size_mb", "Size (MB)",
               "Compressed File Size",
               "bar_filesize.png",
               higher_is_better=False,
               ref_lines=[(orig, f"Original ({orig:.2f} MB)", "#FFD700")])


def plot_space_saved(results, output_dir):
    _bar_chart(results, output_dir,
               "space_saved_pct", "Space Saved (%)",
               "Space Savings",
               "bar_space_saved.png",
               higher_is_better=True,
               ref_lines=[(85, "85% target", "#66BB6A"),
                          (70, "70% target", "#FFA726"),
                          (60, "60% target", "#EF5350")])


def plot_compression_ratio(results, output_dir):
    _bar_chart(results, output_dir,
               "compression_ratio", "Compression Ratio (x)",
               "Compression Ratio",
               "bar_compression_ratio.png",
               higher_is_better=True)


def plot_ssim(results, output_dir):
    _bar_chart(results, output_dir,
               "achieved_ssim", "SSIM (0-1)",
               "Video Quality — SSIM",
               "bar_ssim.png",
               higher_is_better=True,
               ref_lines=[(0.9, "Excellent (0.9)", "#4CAF50"),
                          (0.8, "Acceptable (0.8)", "#FF9800")])


def plot_psnr(results, output_dir):
    _bar_chart(results, output_dir,
               "achieved_psnr", "PSNR (dB)",
               "Video Quality — PSNR",
               "bar_psnr.png",
               higher_is_better=True,
               ref_lines=[(35, "Excellent (35dB)", "#4CAF50"),
                          (30, "Good (30dB)",      "#FF9800")])


def plot_encoding_time(results, output_dir):
    _bar_chart(results, output_dir,
               "encode_time_s", "Encoding Time (s)",
               "Encoding Time",
               "bar_encoding_time.png",
               higher_is_better=False)


def plot_mps_speedup(speedups: dict, output_dir: str):
    if not speedups:
        return
    keys  = sorted(speedups.keys())
    vals  = [speedups[k] for k in keys]
    cols  = [_color(k)   for k in keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F1117")
    _ax_dark(ax, "MPS (GPU) Speedup over CPU")

    x    = np.arange(len(keys))
    bars = ax.bar(x, vals, color=cols, width=0.5, edgecolor="#333",
                  linewidth=0.8, zorder=3)
    ax.axhline(1.0, color="#FFD700", linestyle="--",
               linewidth=1.5, label="No speedup (1×)", zorder=4)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + 0.02, f"{v:.2f}×",
                ha="center", va="bottom", fontsize=10,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in keys], color="white", fontsize=10)
    ax.set_ylabel("Speedup (CPU time / MPS time)", color="white", fontsize=11)
    ax.legend(facecolor="#1a1a2e", labelcolor="white",
              edgecolor="#555", fontsize=10)

    plt.tight_layout()
    out = os.path.join(output_dir, "bar_mps_speedup.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  Saved: bar_mps_speedup.png")


def plot_scatter_quality_ratio(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0F1117")

    for ax, (metric, ylabel) in zip(axes, [
        ("achieved_ssim",  "SSIM"),
        ("achieved_psnr",  "PSNR (dB)"),
    ]):
        _ax_dark(ax, f"Quality vs Compression Ratio ({ylabel})")
        ax.grid(color="#333", alpha=0.4, zorder=0)

        # Group by algorithm family
        families = {}
        for key, r in results.items():
            fam = key.split("_")[0]
            families.setdefault(fam, []).append(r)

        for fam, rlist in families.items():
            c  = ALG_COLORS.get(fam, "#999")
            xs = [r.compression_ratio      for r in rlist]
            ys = [getattr(r, metric, 0)    for r in rlist]
            ds = [r.device                 for r in rlist]
            ax.scatter(xs, ys, c=c, s=150, zorder=5,
                       edgecolors="white", linewidths=0.6, label=fam.upper())
            for x_, y_, d_ in zip(xs, ys, ds):
                ax.annotate(d_.upper(), (x_, y_),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color=c)

        ax.set_xlabel("Compression Ratio (x)", color="white", fontsize=11)
        ax.set_ylabel(ylabel, color="white", fontsize=11)
        ax.legend(facecolor="#1a1a2e", edgecolor="#555",
                  labelcolor="white", fontsize=9)

    fig.suptitle("Quality vs Compression Trade-off",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "scatter_quality_vs_ratio.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  Saved: scatter_quality_vs_ratio.png")


def plot_heatmap(results, output_dir):
    keys    = sorted(results.keys())
    metrics = ["space_saved_pct", "compression_ratio",
               "achieved_ssim",  "achieved_psnr", "encode_time_s"]
    mlabels = ["Space Saved (%)", "Comp. Ratio",
               "SSIM",            "PSNR (dB)",   "Encode Time (s)"]

    data = np.array([[getattr(results[k], m, 0) for m in metrics] for k in keys])
    norm = np.zeros_like(data)
    for col in range(data.shape[1]):
        cv  = data[:, col]
        mn, mx = cv.min(), cv.max()
        norm[:, col] = (cv - mn) / (mx - mn + 1e-9)
    norm[:, 4] = 1 - norm[:, 4]   # encode_time: lower = better = greener

    fig, ax = plt.subplots(figsize=(14, max(6, len(keys)*0.7 + 2)))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    im = ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(mlabels)))
    ax.set_xticklabels(mlabels, color="white", fontsize=10, rotation=15, ha="right")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([_bar_label(k).replace("\n", " ") for k in keys],
                       color="white", fontsize=9)

    for i in range(len(keys)):
        for j, m in enumerate(metrics):
            v   = data[i, j]
            txt = f"{v:.1f}%" if m == "space_saved_pct" else (
                  f"{v:.3f}" if v < 10 else f"{v:.1f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if norm[i, j] > 0.5 else "white",
                    fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Normalized (Green=Best)", color="white", fontsize=10)
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    ax.set_title("Performance Heatmap — All Algorithms & Devices",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "heatmap_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  Saved: heatmap_metrics.png")


def plot_radar(results, output_dir):
    """Radar chart comparing algorithm families (using MPS device if available)."""
    categories = ["SSIM", "PSNR\n(norm)", "Comp\nRatio", "Encode\nSpeed", "Space\nSaved"]
    N      = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    families = {}
    for key, r in results.items():
        fam = key.split("_")[0]
        if fam not in families or "mps" in key:   # prefer MPS
            families[fam] = r

    # Normalise across families
    def nrm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn)/(mx - mn + 1e-9) for v in vals]

    ssim_vals  = [r.achieved_ssim      for r in families.values()]
    psnr_vals  = [r.achieved_psnr      for r in families.values()]
    ratio_vals = [r.compression_ratio  for r in families.values()]
    time_vals  = [r.encode_time_s      for r in families.values()]
    space_vals = [r.space_saved_pct    for r in families.values()]

    nssi  = nrm(ssim_vals)
    npsnr = nrm(psnr_vals)
    nrat  = nrm(ratio_vals)
    nspd  = [1 - v for v in nrm(time_vals)]
    nspc  = nrm(space_vals)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#111827")

    for i, (fam, r) in enumerate(families.items()):
        c    = ALG_COLORS.get(fam, "#999")
        vals = [nssi[i], npsnr[i], nrat[i], nspd[i], nspc[i]]
        vals += [vals[0]]
        ax.plot(angles, vals, color=c, linewidth=2.5, label=fam.upper())
        ax.fill(angles, vals, color=c, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="#aaa", fontsize=8)
    ax.spines["polar"].set_color("#333")
    ax.grid(color="#444", alpha=0.5)
    ax.set_title("Method Profile — Radar Chart",
                 color="white", fontsize=13, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              facecolor="#1a1a2e", edgecolor="#555",
              labelcolor="white", fontsize=11)

    plt.tight_layout()
    out = os.path.join(output_dir, "radar_method_profile.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  Saved: radar_method_profile.png")


def plot_dashboard(results, speedups, meta, output_dir):
    """All-in-one single-page dashboard."""
    keys   = sorted(results.keys())
    labels = [_bar_label(k) for k in keys]
    cols   = [_color(k) for k in keys]

    def vals(metric):
        return [getattr(results[k], metric, 0) for k in keys]

    fig = plt.figure(figsize=(26, 18))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

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
        ax.bar(x, v, color=cols, edgecolor="#333", linewidth=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="white", fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(f"{'Higher' if hib else 'Lower'} is Better",
                      color="#aaa", fontsize=8)

    # MPS speedup inset
    if speedups:
        ax_sp = fig.add_axes([0.01, 0.01, 0.22, 0.20])
        ax_sp.set_facecolor("#111827")
        sp_k = sorted(speedups.keys())
        sp_v = [speedups[k] for k in sp_k]
        sp_c = [ALG_COLORS.get(k, "#999") for k in sp_k]
        ax_sp.bar(range(len(sp_k)), sp_v, color=sp_c, edgecolor="#333", zorder=3)
        ax_sp.axhline(1.0, color="#FFD700", linestyle="--", lw=1.2)
        ax_sp.set_xticks(range(len(sp_k)))
        ax_sp.set_xticklabels([k.upper() for k in sp_k], color="white", fontsize=7)
        ax_sp.set_title("MPS Speedup (x)", color="white", fontsize=9, fontweight="bold")
        ax_sp.tick_params(colors="white", labelsize=7)
        ax_sp.spines[:].set_color("#333")
        ax_sp.grid(axis="y", color="#333", alpha=0.4)

    patches = [mpatches.Patch(color=c, label=n.upper())
               for n, c in ALG_COLORS.items()]
    fig.legend(handles=patches, loc="upper right",
               facecolor="#1a1a2e", edgecolor="#555",
               labelcolor="white", fontsize=10)

    fig.suptitle(
        f"Benchmark Dashboard — {meta.get('video_name','video')}  |  "
        f"target_ssim={meta.get('target_ssim','')}  |  "
        f"frames={meta.get('max_frames','')}  |  {meta.get('generated','')}",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )

    out = os.path.join(output_dir, "combined_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  Saved: combined_dashboard.png")


# ══════════════════════════════════════════════════════════════════
# Master function
# ══════════════════════════════════════════════════════════════════

def generate_all_charts(results: Dict[str, CompressResult],
                        speedups: dict,
                        meta:    dict,
                        output_dir: str = "benchmark_output/charts"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[CHARTS] Saving to {output_dir}/")

    plot_filesize(results,          output_dir)
    plot_space_saved(results,       output_dir)
    plot_compression_ratio(results, output_dir)
    plot_ssim(results,              output_dir)
    plot_psnr(results,              output_dir)
    plot_encoding_time(results,     output_dir)
    plot_mps_speedup(speedups,      output_dir)
    plot_scatter_quality_ratio(results, output_dir)
    plot_heatmap(results,           output_dir)
    plot_radar(results,             output_dir)
    plot_dashboard(results, speedups, meta, output_dir)

    n = len(os.listdir(output_dir))
    print(f"\n  {n} charts generated in {output_dir}/")
