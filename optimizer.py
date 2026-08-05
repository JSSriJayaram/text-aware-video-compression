"""
optimizer.py
============
Multi-Objective Optimization Engine for the Proposed Text-Aware Compressor.

Given a user's constraint (desired SSIM OR desired compression ratio),
find the optimal (bg_quality, ssim_threshold) parameters that:
  - MINIMIZE encoding time
  - SATISFY the quality / compression constraint
  - NEVER degrade text regions (text_quality is always fixed at 90)

Techniques used:
  1. Grid Profiling       — sample 25 parameter combinations, measure all 3 objectives
  2. Polynomial Surrogate — fit a 2D polynomial surface to the profiled data
  3. COBYLA               — scipy constrained optimization on the surrogate surface
  4. Pareto Front         — identify the non-dominated solutions across all 3 objectives
  5. Gradient Descent     — used on the smooth surrogate surface to find minimum
"""

import os
import sys
import json
import time
import itertools
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from algorithms.proposed.proposed_compressor import (
    _ssim_to_params, _qt_partition, _compress_frame
)

try:
    from cutils_mod import load_craftnet_model, load_refinenet_model, get_prediction
    CRAFT_AVAILABLE = True
except ImportError:
    CRAFT_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════
# Inline frame metrics (no file I/O overhead during profiling)
# ══════════════════════════════════════════════════════════════════

def _frame_ssim(orig: np.ndarray, comp: np.ndarray) -> float:
    og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
    if og.shape != cg.shape:
        cg = cv2.resize(cg, (og.shape[1], og.shape[0]))
    try:
        v, _ = ssim_fn(og, cg, full=True)
        return float(v)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════
# Quick video profiler — tests one (bg_quality, ssim_threshold) combo
# ══════════════════════════════════════════════════════════════════

def _profile_params(input_path:      str,
                    bg_quality:      int,
                    ssim_threshold:  float,
                    text_quality:    int   = 90,
                    probe_frames:    int   = 30,
                    text_mask_cache: dict  = None,
                    device:          str   = "cpu") -> Tuple[float, float, float]:
    """
    Compress `probe_frames` frames with given params.
    Returns (encode_time_s, mean_ssim, compression_ratio).
    """
    cap   = cv2.VideoCapture(input_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ── Sample probe frames ────────────────────────────────────
    sample_idx = np.linspace(0, max(total - 1, 0),
                             min(probe_frames, total), dtype=int)

    cap       = cv2.VideoCapture(input_path)
    ref_frame = None
    enc_bytes = 0
    orig_bytes= 0
    ssim_vals = []
    encode_times = []

    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        # Temporal filter
        if ref_frame is not None:
            rg = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            fg = cv2.cvtColor(frame,     cv2.COLOR_BGR2GRAY)
            sim, _ = ssim_fn(rg, fg, full=True)
            if float(sim) >= ssim_threshold:
                continue  # skipped

        # Use cached text mask or heuristic
        if text_mask_cache and idx in text_mask_cache:
            text_mask = text_mask_cache[idx]
        else:
            gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap       = np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.uint8)
            _, text_mask = cv2.threshold(lap, 20, 255, cv2.THRESH_BINARY)

        t0 = time.time()
        comp_frame, _ = _compress_frame(frame, text_mask, bg_quality, text_quality)
        encode_times.append(time.time() - t0)

        # Estimate size via JPEG compression (proxy for DCT-compressed bytes)
        _, buf_orig = cv2.imencode('.jpg', frame,      [cv2.IMWRITE_JPEG_QUALITY, 95])
        _, buf_comp = cv2.imencode('.jpg', comp_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        orig_bytes += len(buf_orig)
        enc_bytes  += len(buf_comp)

        ssim_vals.append(_frame_ssim(frame, comp_frame))
        ref_frame = frame

    cap.release()

    mean_ssim  = float(np.mean(ssim_vals))  if ssim_vals  else 0.0
    total_time = float(np.sum(encode_times)) if encode_times else 0.0
    ratio      = orig_bytes / max(enc_bytes, 1)

    return total_time, mean_ssim, ratio


# ══════════════════════════════════════════════════════════════════
# Pareto front helper
# ══════════════════════════════════════════════════════════════════

def _is_pareto_dominated(costs: np.ndarray) -> np.ndarray:
    """
    Given costs array of shape (N, 3) where lower is better for all,
    return boolean array: dominated[i] = True if point i is dominated.
    """
    n = len(costs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is <= i on all dims and < on at least one
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                dominated[i] = True
                break
    return dominated


# ══════════════════════════════════════════════════════════════════
# Optimizer
# ══════════════════════════════════════════════════════════════════

class ProposedOptimizer:
    """
    Multi-objective optimizer for the Proposed compression algorithm.

    Parameters tuned:
      bg_quality      ∈ [10, 90]
      ssim_threshold  ∈ [0.80, 0.99]
      text_quality    = 90  (fixed — never compress text)

    Objectives (all to minimize for Pareto front):
      f1 = encode_time
      f2 = 1 - achieved_ssim   (want max ssim → minimize 1-ssim)
      f3 = 1 / compression_ratio  (want max ratio → minimize 1/ratio)
    """

    BG_QUALITY_RANGE     = (10, 90)
    SSIM_THRESH_RANGE    = (0.80, 0.99)
    TEXT_QUALITY         = 90

    def __init__(self,
                 input_path:    str,
                 output_dir:    str = "benchmark_output/optimizer",
                 probe_frames:  int = 25,
                 grid_points:   int = 5):
        self.input_path   = input_path
        self.output_dir   = output_dir
        self.probe_frames = probe_frames
        self.grid_points  = grid_points
        os.makedirs(output_dir, exist_ok=True)

        self.grid_data    = []   # list of (bg_q, ssim_t, time, ssim, ratio)
        self.poly_models  = {}   # surrogate polynomials

    # ── Phase 1: Grid Profiling ────────────────────────────────

    def profile_grid(self):
        bg_vals   = np.linspace(*self.BG_QUALITY_RANGE,   self.grid_points, dtype=int)
        st_vals   = np.linspace(*self.SSIM_THRESH_RANGE,  self.grid_points)
        combos    = list(itertools.product(bg_vals, st_vals))
        total     = len(combos)

        print(f"\n[OPTIMIZER] Grid profiling {total} parameter combinations...")

        for i, (bg_q, st) in enumerate(combos):
            print(f"  [{i+1}/{total}] bg_quality={bg_q}, ssim_threshold={st:.3f}")
            enc_t, ssim_v, ratio = _profile_params(
                self.input_path, int(bg_q), float(st),
                self.TEXT_QUALITY, self.probe_frames
            )
            self.grid_data.append({
                "bg_quality":     int(bg_q),
                "ssim_threshold": float(st),
                "encode_time":    enc_t,
                "achieved_ssim":  ssim_v,
                "compression_ratio": ratio,
                # Pareto costs (all minimize)
                "f1_time":   enc_t,
                "f2_quality": 1.0 - ssim_v,
                "f3_size":    1.0 / max(ratio, 1e-6),
            })
            print(f"    → time={enc_t:.3f}s  ssim={ssim_v:.4f}  ratio={ratio:.2f}×")

        # Save grid data
        with open(os.path.join(self.output_dir, "grid_profile.json"), "w") as f:
            json.dump(self.grid_data, f, indent=4)
        print(f"  Grid data saved.")

    # ── Phase 2: Polynomial Surrogate Model ───────────────────

    def fit_surrogate(self):
        """Fit degree-2 polynomial to each objective over (bg_quality, ssim_threshold)."""
        from numpy.polynomial import polynomial as P

        X = np.array([[d["bg_quality"], d["ssim_threshold"]] for d in self.grid_data])
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

        for metric in ["encode_time", "achieved_ssim", "compression_ratio"]:
            Y = np.array([d[metric] for d in self.grid_data])
            # Build polynomial feature matrix [1, x1, x2, x1^2, x2^2, x1*x2]
            F = np.column_stack([
                np.ones(len(X_norm)),
                X_norm[:, 0], X_norm[:, 1],
                X_norm[:, 0]**2, X_norm[:, 1]**2,
                X_norm[:, 0] * X_norm[:, 1],
            ])
            coeffs, *_ = np.linalg.lstsq(F, Y, rcond=None)
            self.poly_models[metric] = {
                "coeffs": coeffs.tolist(),
                "X_mean": X.mean(axis=0).tolist(),
                "X_std":  X.std(axis=0).tolist(),
            }
        print("  Surrogate models fitted.")

    def _surrogate_predict(self, bg_q: float, st: float, metric: str) -> float:
        m       = self.poly_models[metric]
        xm      = np.array(m["X_mean"])
        xs      = np.array(m["X_std"]) + 1e-9
        x_norm  = np.array([(bg_q - xm[0]) / xs[0],
                             (st   - xm[1]) / xs[1]])
        F       = np.array([1, x_norm[0], x_norm[1],
                            x_norm[0]**2, x_norm[1]**2,
                            x_norm[0] * x_norm[1]])
        return float(np.dot(m["coeffs"], F))

    # ── Phase 3: COBYLA Constrained Optimization ──────────────

    def optimize(self,
                 target_ssim:       Optional[float] = None,
                 target_ratio:      Optional[float] = None) -> dict:
        """
        Find (bg_quality, ssim_threshold) that minimizes encode_time
        subject to quality and/or compression ratio constraints.

        Args:
            target_ssim:  Desired minimum achieved SSIM.
            target_ratio: Desired minimum compression ratio.

        Returns:
            dict with optimal parameters and predicted metrics.
        """
        if not self.poly_models:
            self.fit_surrogate()

        # Objective: minimize encode_time
        def objective(x):
            return self._surrogate_predict(x[0], x[1], "encode_time")

        # Constraints
        constraints = []
        if target_ssim is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda x: (
                    self._surrogate_predict(x[0], x[1], "achieved_ssim")
                    - target_ssim
                )
            })
        if target_ratio is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda x: (
                    self._surrogate_predict(x[0], x[1], "compression_ratio")
                    - target_ratio
                )
            })

        # Bounds — keep the search inside the region the grid actually profiled.
        # Outside it the polynomial surrogate is extrapolating and its predictions
        # are meaningless, so an unbounded search can wander off and then get
        # silently clipped back to the edge (looking like a boundary optimum).
        bounds = [
            self.BG_QUALITY_RANGE,
            self.SSIM_THRESH_RANGE,
        ]

        # Starting point (middle of grid)
        x0 = [
            (self.BG_QUALITY_RANGE[0]  + self.BG_QUALITY_RANGE[1])  / 2,
            (self.SSIM_THRESH_RANGE[0] + self.SSIM_THRESH_RANGE[1]) / 2,
        ]

        print(f"\n[OPTIMIZER] COBYLA optimization...")
        result = minimize(
            objective,
            x0,
            method="COBYLA",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "rhobeg": 5.0},
        )

        opt_bg_q = float(np.clip(result.x[0], *self.BG_QUALITY_RANGE))
        opt_st   = float(np.clip(result.x[1], *self.SSIM_THRESH_RANGE))

        pred_time  = self._surrogate_predict(opt_bg_q, opt_st, "encode_time")
        pred_ssim  = self._surrogate_predict(opt_bg_q, opt_st, "achieved_ssim")
        pred_ratio = self._surrogate_predict(opt_bg_q, opt_st, "compression_ratio")

        optimal = {
            "bg_quality":        int(round(opt_bg_q)),
            "ssim_threshold":    round(opt_st, 3),
            "text_quality":      self.TEXT_QUALITY,
            "predicted_time_s":  pred_time,
            "predicted_ssim":    pred_ssim,
            "predicted_ratio":   pred_ratio,
            "target_ssim":       target_ssim,
            "target_ratio":      target_ratio,
            "cobyla_success":    bool(result.success),
            "cobyla_message":    str(result.message),
        }

        with open(os.path.join(self.output_dir, "optimal_params.json"), "w") as f:
            json.dump(optimal, f, indent=4)

        print(f"  Optimal bg_quality     : {optimal['bg_quality']}")
        print(f"  Optimal ssim_threshold : {optimal['ssim_threshold']}")
        print(f"  Predicted encode_time  : {pred_time:.3f}s")
        print(f"  Predicted SSIM         : {pred_ssim:.4f}")
        print(f"  Predicted ratio        : {pred_ratio:.2f}×")
        return optimal

    # ── Phase 4: Pareto Front ─────────────────────────────────

    def compute_pareto(self) -> list:
        costs   = np.array([[d["f1_time"], d["f2_quality"], d["f3_size"]]
                            for d in self.grid_data])
        dominated = _is_pareto_dominated(costs)
        pareto    = [d for d, dom in zip(self.grid_data, dominated) if not dom]
        print(f"  Pareto front: {len(pareto)} / {len(self.grid_data)} points are non-dominated.")
        return pareto

    # ── Phase 5: Plot Charts ──────────────────────────────────

    def plot_all(self):
        self._plot_pareto_front()
        self._plot_surface()
        self._plot_parameter_sensitivity()

    def _plot_pareto_front(self):
        pareto = self.compute_pareto()
        all_d  = self.grid_data

        fig = plt.figure(figsize=(14, 6))
        fig.patch.set_facecolor("#0D1117")

        # 2D: time vs ssim
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_facecolor("#111827")
        all_t  = [d["encode_time"]   for d in all_d]
        all_s  = [d["achieved_ssim"] for d in all_d]
        par_t  = [d["encode_time"]   for d in pareto]
        par_s  = [d["achieved_ssim"] for d in pareto]

        ax1.scatter(all_t,  all_s,  c="#4A90D9", s=60,  alpha=0.5,
                    label="All grid points", zorder=3)
        ax1.scatter(par_t,  par_s,  c="#E91E63", s=120, marker="*",
                    label="Pareto front",    zorder=5, edgecolors="white", lw=0.5)

        # annotate pareto points
        for d in pareto:
            ax1.annotate(f"bg={d['bg_quality']}\nst={d['ssim_threshold']:.2f}",
                         (d["encode_time"], d["achieved_ssim"]),
                         fontsize=6.5, color="#aaffaa",
                         xytext=(4, 4), textcoords="offset points")

        ax1.set_xlabel("Encode Time (s)", color="white")
        ax1.set_ylabel("Achieved SSIM",   color="white")
        ax1.set_title("Pareto Front: Time vs Quality",
                      color="white", fontweight="bold")
        ax1.legend(facecolor="#1a1a2e", labelcolor="white",
                   edgecolor="#555", fontsize=9)
        ax1.tick_params(colors="white")
        ax1.spines[:].set_color("#333")
        ax1.grid(color="#333", alpha=0.4)

        # 2D: time vs ratio
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_facecolor("#111827")
        all_r = [d["compression_ratio"] for d in all_d]
        par_r = [d["compression_ratio"] for d in pareto]

        ax2.scatter(all_t,  all_r,  c="#4A90D9", s=60,  alpha=0.5,
                    label="All grid points", zorder=3)
        ax2.scatter(par_t,  par_r,  c="#E91E63", s=120, marker="*",
                    label="Pareto front",    zorder=5, edgecolors="white", lw=0.5)

        ax2.set_xlabel("Encode Time (s)",       color="white")
        ax2.set_ylabel("Compression Ratio (x)", color="white")
        ax2.set_title("Pareto Front: Time vs Compression",
                      color="white", fontweight="bold")
        ax2.legend(facecolor="#1a1a2e", labelcolor="white",
                   edgecolor="#555", fontsize=9)
        ax2.tick_params(colors="white")
        ax2.spines[:].set_color("#333")
        ax2.grid(color="#333", alpha=0.4)

        fig.suptitle("Multi-Objective Pareto Front — Proposed Algorithm",
                     color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(self.output_dir, "pareto_front.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D1117")
        plt.close()
        print(f"  Saved: {out}")

    def _plot_surface(self):
        if not self.poly_models:
            self.fit_surrogate()

        bg_range = np.linspace(*self.BG_QUALITY_RANGE,  30)
        st_range = np.linspace(*self.SSIM_THRESH_RANGE, 30)
        BG, ST   = np.meshgrid(bg_range, st_range)

        metrics = {
            "Encode Time (s)":       "encode_time",
            "Achieved SSIM":         "achieved_ssim",
            "Compression Ratio (x)": "compression_ratio",
        }

        fig = plt.figure(figsize=(20, 6))
        fig.patch.set_facecolor("#0D1117")

        for col, (title, metric) in enumerate(metrics.items()):
            Z = np.zeros_like(BG)
            for i in range(BG.shape[0]):
                for j in range(BG.shape[1]):
                    Z[i, j] = self._surrogate_predict(BG[i, j], ST[i, j], metric)

            ax = fig.add_subplot(1, 3, col + 1, projection="3d")
            ax.set_facecolor("#111827")
            surf = ax.plot_surface(BG, ST, Z, cmap="plasma",
                                   edgecolor="none", alpha=0.85)
            ax.set_xlabel("bg_quality",     color="white", fontsize=9)
            ax.set_ylabel("ssim_threshold", color="white", fontsize=9)
            ax.set_zlabel(title,            color="white", fontsize=9)
            ax.set_title(title, color="white", fontweight="bold", fontsize=11)
            ax.tick_params(colors="white", labelsize=7)
            fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)

        fig.suptitle("Surrogate Model: Parameter Space Surface",
                     color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(self.output_dir, "optimization_surface.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D1117")
        plt.close()
        print(f"  Saved: {out}")

    def _plot_parameter_sensitivity(self):
        """Show how each objective changes with each parameter independently."""
        bg_vals = np.linspace(*self.BG_QUALITY_RANGE, 20)
        st_vals = np.linspace(*self.SSIM_THRESH_RANGE, 20)
        metrics = ["encode_time", "achieved_ssim", "compression_ratio"]
        labels  = ["Encode Time (s)", "Achieved SSIM", "Compression Ratio (x)"]
        colors  = ["#2196F3", "#4CAF50", "#FF9800"]

        mid_st = (self.SSIM_THRESH_RANGE[0] + self.SSIM_THRESH_RANGE[1]) / 2
        mid_bg = (self.BG_QUALITY_RANGE[0]  + self.BG_QUALITY_RANGE[1])  / 2

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("Parameter Sensitivity Analysis",
                     color="white", fontsize=14, fontweight="bold")

        for col, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            # Vary bg_quality, fix ssim_threshold at midpoint
            vals_bg = [self._surrogate_predict(bg, mid_st, metric) for bg in bg_vals]
            ax = axes[0][col]
            ax.set_facecolor("#111827")
            ax.plot(bg_vals, vals_bg, color=color, linewidth=2.5)
            ax.set_xlabel("bg_quality", color="white")
            ax.set_ylabel(label, color="white")
            ax.set_title(f"{label}\nvs bg_quality", color="white", fontweight="bold")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333")
            ax.grid(color="#333", alpha=0.4)

            # Vary ssim_threshold, fix bg_quality at midpoint
            vals_st = [self._surrogate_predict(mid_bg, st, metric) for st in st_vals]
            ax2 = axes[1][col]
            ax2.set_facecolor("#111827")
            ax2.plot(st_vals, vals_st, color=color, linewidth=2.5, linestyle="--")
            ax2.set_xlabel("ssim_threshold", color="white")
            ax2.set_ylabel(label, color="white")
            ax2.set_title(f"{label}\nvs ssim_threshold", color="white", fontweight="bold")
            ax2.tick_params(colors="white")
            ax2.spines[:].set_color("#333")
            ax2.grid(color="#333", alpha=0.4)

        plt.tight_layout()
        out = os.path.join(self.output_dir, "parameter_sensitivity.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D1117")
        plt.close()
        print(f"  Saved: {out}")

    # ── Full run ──────────────────────────────────────────────

    def run(self, target_ssim: Optional[float] = None,
            target_ratio: Optional[float] = None) -> dict:
        print("\n" + "="*60)
        print("OPTIMIZER — Phase 1: Grid Profiling")
        print("="*60)
        self.profile_grid()

        print("\n" + "="*60)
        print("OPTIMIZER — Phase 2: Surrogate Model Fitting")
        print("="*60)
        self.fit_surrogate()

        print("\n" + "="*60)
        print("OPTIMIZER — Phase 3: COBYLA Optimization")
        print("="*60)
        optimal = self.optimize(target_ssim=target_ssim,
                                target_ratio=target_ratio)

        print("\n" + "="*60)
        print("OPTIMIZER — Phase 4: Pareto Front + Charts")
        print("="*60)
        self.plot_all()

        return optimal
