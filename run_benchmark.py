"""
run_benchmark.py
================
CLI entry point for the full benchmark + optimizer pipeline.

Usage examples:
  # Full benchmark (all algorithms, MPS + CPU)
  ./craft_env/bin/python3 run_benchmark.py --input TESTFILE3.mp4 --ssim 0.90

  # With optimization (find fastest params for target quality)
  ./craft_env/bin/python3 run_benchmark.py --input TESTFILE3.mp4 --ssim 0.90 --optimize

  # CPU only
  ./craft_env/bin/python3 run_benchmark.py --input TESTFILE3.mp4 --ssim 0.85 --no-mps

  # Specific algorithms only
  ./craft_env/bin/python3 run_benchmark.py --input TESTFILE3.mp4 --ssim 0.90 \
      --algorithms proposed h264 ssim_driven

  # Custom frame count and output directory
  ./craft_env/bin/python3 run_benchmark.py --input TESTFILE3.mp4 --ssim 0.90 \
      --frames 60 --outdir my_results
"""

import argparse
import os
import sys
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Text-Aware Video Compression Benchmark + Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", default="TESTFILE3.mp4",
                   help="Input video file (default: TESTFILE3.mp4)")
    p.add_argument("--ssim", type=float, default=0.90,
                   help="Target SSIM quality 0.0–1.0. "
                        "Higher = better quality + less compression. "
                        "(default: 0.90)")
    p.add_argument("--frames", type=int, default=120,
                   help="Max frames to benchmark per algorithm (default: 120). "
                        "Use fewer for quick tests.")
    p.add_argument("--outdir", default="benchmark_output",
                   help="Output directory (default: benchmark_output)")
    p.add_argument("--algorithms", nargs="+",
                   choices=["proposed", "h264", "h265", "vp9", "ssim_driven"],
                   default=["proposed", "h264", "h265", "vp9", "ssim_driven"],
                   help="Which algorithms to run (default: all)")
    p.add_argument("--no-mps", action="store_true",
                   help="Skip MPS (Apple GPU) runs, CPU only")
    p.add_argument("--no-cpu", action="store_true",
                   help="Skip CPU runs, MPS only")
    p.add_argument("--optimize", action="store_true",
                   help="Run the multi-objective optimizer after benchmarking")
    p.add_argument("--opt-frames", type=int, default=20,
                   help="Frames used per optimizer probe (default: 20). "
                        "Fewer = faster optimizer but less accurate.")
    p.add_argument("--opt-grid", type=int, default=5,
                   help="Grid points per axis for optimizer (default: 5 → 25 combos)")
    p.add_argument("--target-ratio", type=float, default=None,
                   help="Optional: desired minimum compression ratio for optimizer")
    return p.parse_args()


def print_banner(args):
    print("\n" + "=" * 70)
    print("  TEXT-AWARE VIDEO COMPRESSION — FULL BENCHMARK")
    print("=" * 70)
    print(f"  Input         : {args.input}")
    print(f"  Target SSIM   : {args.ssim}  "
          f"({'high quality' if args.ssim >= 0.9 else 'balanced' if args.ssim >= 0.75 else 'aggressive compression'})")
    print(f"  Max frames    : {args.frames}")
    print(f"  Algorithms    : {', '.join(args.algorithms)}")
    print(f"  Run MPS       : {not args.no_mps}")
    print(f"  Run CPU       : {not args.no_cpu}")
    print(f"  Run optimizer : {args.optimize}")
    print(f"  Output dir    : {args.outdir}")
    print("=" * 70)


def main():
    args = parse_args()

    # ── Validate input ─────────────────────────────────────────
    if not os.path.exists(args.input):
        available = list(Path(".").glob("*.mp4"))
        print(f"\n[ERROR] Video not found: {args.input}")
        if available:
            print("  Available .mp4 files:")
            for v in available:
                print(f"    {v.name}  ({v.stat().st_size/1024/1024:.1f} MB)")
        sys.exit(1)

    if args.no_mps and args.no_cpu:
        print("[ERROR] Cannot use --no-mps and --no-cpu together.")
        sys.exit(1)

    print_banner(args)

    # ── Phase 1: Benchmark ─────────────────────────────────────
    from benchmark_runner import BenchmarkRunner

    runner = BenchmarkRunner(
        input_path  = args.input,
        target_ssim = args.ssim,
        max_frames  = args.frames,
        output_dir  = args.outdir,
        algorithms  = args.algorithms,
        run_mps     = not args.no_mps,
        run_cpu     = not args.no_cpu,
    )
    bench_data = runner.run()
    results    = bench_data["results"]
    speedups   = bench_data["speedups"]
    meta       = bench_data["meta"]

    # ── Phase 2: Charts ────────────────────────────────────────
    from chart_generator import generate_all_charts

    charts_dir = os.path.join(args.outdir, "charts")
    generate_all_charts(results, speedups, meta, charts_dir)

    # ── Phase 3 (optional): Optimizer ─────────────────────────
    if args.optimize:
        from optimizer import ProposedOptimizer

        opt_dir = os.path.join(args.outdir, "optimizer")
        print(f"\n{'='*70}")
        print(f"  OPTIMIZER — Grid: {args.opt_grid}x{args.opt_grid}, "
              f"Probe frames: {args.opt_frames}")
        print(f"  Constraint: SSIM >= {args.ssim}"
              + (f",  Ratio >= {args.target_ratio}" if args.target_ratio else ""))
        print(f"{'='*70}")

        optimizer = ProposedOptimizer(
            input_path   = args.input,
            output_dir   = opt_dir,
            probe_frames = args.opt_frames,
            grid_points  = args.opt_grid,
        )
        optimal = optimizer.run(
            target_ssim  = args.ssim,
            target_ratio = args.target_ratio,
        )

        print(f"\n{'='*70}")
        print("  OPTIMIZER RESULT")
        print(f"{'='*70}")
        print(f"  Optimal bg_quality    : {optimal['bg_quality']}")
        print(f"  Optimal ssim_threshold: {optimal['ssim_threshold']}")
        print(f"  text_quality (fixed)  : {optimal['text_quality']}")
        print(f"  Predicted time        : {optimal['predicted_time_s']:.3f}s")
        print(f"  Predicted SSIM        : {optimal['predicted_ssim']:.4f}")
        print(f"  Predicted ratio       : {optimal['predicted_ratio']:.2f}x")

    # ── Final summary ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"  All outputs in: {args.outdir}/")
    print()

    # Print results table
    all_keys = sorted(results.keys())
    print(f"  {'Method':<25} {'Size(MB)':>9} {'Saved%':>8} {'SSIM':>7} "
          f"{'PSNR(dB)':>9} {'Time(s)':>8}")
    print("  " + "-"*70)
    for k in all_keys:
        r = results[k]
        nm = f"{r.algorithm}/{r.device}"
        print(f"  {nm:<25} {r.compressed_size_mb:>9.2f} "
              f"{r.space_saved_pct:>8.1f} {r.achieved_ssim:>7.4f} "
              f"{r.achieved_psnr:>9.2f} {r.encode_time_s:>8.2f}")

    print(f"\n  JSON results : {args.outdir}/benchmark_results.json")
    print(f"  Charts       : {args.outdir}/charts/  "
          f"({len(os.listdir(os.path.join(args.outdir,'charts'))) if os.path.isdir(os.path.join(args.outdir,'charts')) else 0} files)")
    if args.optimize:
        print(f"  Optimizer    : {args.outdir}/optimizer/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
