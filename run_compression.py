"""
run_compression.py
==================
Entry point for the Text-Aware Video Compression System.

Usage:
    python run_compression.py
    python run_compression.py --input VIDEO-2025-11-23-19-16-52.mp4
    python run_compression.py --input TESTFILE3.mp4 VIDEO-2025-11-23-19-16-52.mp4
    python run_compression.py --input TESTFILE3.mp4 --ssim 0.92 --bg-quality 25
    python run_compression.py --input TESTFILE3.mp4 --outdir ./my_output

Output structure (per video):
    output/<video_name>/
        ├── <video_name>_compressed.mp4          — DCT-compressed video
        ├── <video_name>_detected.mp4            — bounding box overlay video
        ├── <video_name>_text_heatmap.mp4        — CRAFT text score heatmap
        ├── <video_name>_link_heatmap.mp4        — CRAFT link score heatmap
        ├── <video_name>_text_mask_overlay.mp4   — text mask overlay
        ├── <video_name>_TextRegion.txt          — per-frame text coordinates
        ├── analysis/
        │   ├── analysis_report.txt              — detailed analysis report
        │   ├── statistics.json                  — machine-readable stats
        │   └── performance_analysis.png         — performance plots
        └── dct_frames/
            ├── frame_0001_dct.jpg               — DCT compressed frame
            ├── frame_0001_dct.gif               — DCT quadtree GIF animation
            └── ...
"""

import argparse
import os
import sys
from pathlib import Path

from text_aware_compressor import compress_video, compress_multiple_videos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-Aware Video Compression System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", nargs="+", default=["TESTFILE3.mp4"],
        help="One or more input video file paths (default: TESTFILE3.mp4)"
    )
    parser.add_argument(
        "--outdir", default="./output",
        help="Base output directory (default: ./output). A subfolder is created for each video."
    )
    parser.add_argument(
        "--ssim", type=float, default=0.95,
        help="SSIM threshold for Technique 2 frame skipping (default: 0.95)"
    )
    parser.add_argument(
        "--method", default="ssim", choices=["ssim", "mse", "pixel"],
        help="Frame comparison method for Technique 2 (default: ssim)"
    )
    parser.add_argument(
        "--txt-thresh", type=float, default=0.85,
        help="CRAFT text confidence threshold (default: 0.85)"
    )
    parser.add_argument(
        "--lnk-thresh", type=float, default=0.6,
        help="CRAFT link/affinity threshold (default: 0.6)"
    )
    parser.add_argument(
        "--txt-quality", type=int, default=90,
        help="DCT quality for text blocks (1–100, default: 90)"
    )
    parser.add_argument(
        "--bg-quality", type=int, default=30,
        help="DCT quality for background blocks (1–100, default: 30)"
    )
    parser.add_argument(
        "--no-refine", action="store_true",
        help="Disable RefineNet (faster, less accurate text boundary linking)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    valid_inputs = []
    for inp in args.input:
        if os.path.exists(inp):
            valid_inputs.append(inp)
        else:
            print(f"[WARN] Video not found: {inp} — SKIPPED")

    if not valid_inputs:
        print(f"\n[ERROR] No valid input videos found.")
        print(f"\nAvailable videos in current directory:")
        for f in sorted(Path(".").glob("*.mp4")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}  ({size_mb:.1f} MB)")
        sys.exit(1)

    # ── Print configuration ───────────────────────────────────────────────────
    print("=" * 70)
    print("TEXT-AWARE VIDEO COMPRESSION — CONFIGURATION")
    print("=" * 70)
    print(f"  Input video(s):       {', '.join(valid_inputs)}")
    print(f"  Output directory:     {args.outdir}")
    print()
    print("  ── Technique 2: Temporal Filter ──")
    print(f"  Comparison method:    {args.method.upper()}")
    print(f"  SSIM threshold:       {args.ssim}")
    print()
    print("  ── Technique 1: Spatial Compression ──")
    print(f"  CRAFT text threshold: {args.txt_thresh}")
    print(f"  CRAFT link threshold: {args.lnk_thresh}")
    print(f"  RefineNet:            {'Disabled' if args.no_refine else 'Enabled'}")
    print(f"  Text block quality:   {args.txt_quality}/100")
    print(f"  Background quality:   {args.bg_quality}/100")
    print()
    print("  ── Output Structure (per video) ──")
    for v in valid_inputs:
        vname = Path(v).stem
        print(f"  {args.outdir}/{vname}/")
        print(f"    ├── {vname}_compressed.mp4")
        print(f"    ├── {vname}_detected.mp4")
        print(f"    ├── {vname}_text_heatmap.mp4")
        print(f"    ├── {vname}_link_heatmap.mp4")
        print(f"    ├── {vname}_text_mask_overlay.mp4")
        print(f"    ├── {vname}_TextRegion.txt")
        print(f"    ├── analysis/ (report, stats, plots)")
        print(f"    └── dct_frames/ (per-frame DCT jpg + gif)")
    print("=" * 70)

    # ── Compression kwargs ────────────────────────────────────────────────────
    kwargs = dict(
        ssim_threshold=args.ssim,
        comparison_method=args.method,
        text_threshold=args.txt_thresh,
        link_threshold=args.lnk_thresh,
        text_quality=args.txt_quality,
        background_quality=args.bg_quality,
        use_refinenet=not args.no_refine,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    if len(valid_inputs) == 1:
        stats = compress_video(
            input_path=valid_inputs[0],
            base_output_dir=args.outdir,
            **kwargs,
        )
        video_name = Path(valid_inputs[0]).stem
        video_dir = os.path.join(args.outdir, video_name)
        _print_file_comparison(valid_inputs[0], video_dir, video_name, stats)
    else:
        results = compress_multiple_videos(
            video_list=valid_inputs,
            base_output_dir=args.outdir,
            **kwargs,
        )
        print("\n" + "=" * 70)
        print("ALL VIDEOS PROCESSED — SUMMARY")
        print("=" * 70)
        for vname, stats in results:
            tf = stats.get('temporal_filter_stats', {})
            video_dir = os.path.join(args.outdir, vname)
            print(f"\n  {vname}:")
            print(f"    Frames: {stats['total_frames']}")
            print(f"    Processed: {tf.get('frames_processed', '?')} | "
                  f"Skipped: {tf.get('frames_skipped', '?')} | "
                  f"Saved: {tf.get('computation_saved_pct', 0):.1f}%")
            print(f"    Text regions: {stats['total_text_regions']}")
            print(f"    DCT frames saved: {stats.get('dct_frames_saved', 0)}")
            print(f"    Output: {video_dir}/")
        print("=" * 70)


def _print_file_comparison(input_path, video_dir, video_name, stats):
    """Print file size comparison for a single video."""
    input_size = os.path.getsize(input_path) / (1024 * 1024)
    compressed_path = os.path.join(video_dir, f"{video_name}_compressed.mp4")
    if os.path.exists(compressed_path):
        output_size = os.path.getsize(compressed_path) / (1024 * 1024)
        ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0
    else:
        output_size = 0
        ratio = 0

    tf = stats.get('temporal_filter_stats', {})

    print("\n── FILE SIZE COMPARISON ──────────────────────────────────────────")
    print(f"  Input  file size:  {input_size:.1f} MB")
    print(f"  Output file size:  {output_size:.1f} MB")
    print(f"  Space saved:       {ratio:.1f}%")
    print()
    print(f"  Frames total:       {tf.get('total_frames_seen', '?')}")
    print(f"  Frames compressed:  {tf.get('frames_processed', '?')}  (Technique 1)")
    print(f"  Frames skipped:     {tf.get('frames_skipped', '?')}   (Technique 2)")
    print(f"  Computation saved:  {tf.get('computation_saved_pct', 0):.1f}%")
    print(f"  DCT frames saved:   {stats.get('dct_frames_saved', 0)}")
    print("─" * 70)


if __name__ == "__main__":
    main()
