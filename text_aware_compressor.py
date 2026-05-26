"""
text_aware_compressor.py
========================
Text-Aware Video Compression System
Implements the full architecture described in the research prompt:

  - Technique 2: Reference Frame Temporal Filtering (SSIM-based frame skipping)
  - Technique 1: Text-Aware Spatial Compression
      1. Preprocessing module
      2. Text detection (CRAFT + RefineNet)
      3. Binary text mask generation
      4. Adaptive quadtree partitioning
      5. Block variance analysis
      6. Block classification (Text / Background)
      7. DCT-based compression (differential quantization)
      8. Region simplification for uniform backgrounds
      9. Frame reconstruction (inverse DCT)

The user's DCT quadtree code (dct_quadtree.py) is imported and used as-is.
Existing CRAFT/RefineNet pipeline (cutils_mod.py) is reused via import.

Output Directory Structure (per video):
    output/
      <video_name>/
        <video_name>_compressed.mp4          — compressed output video
        <video_name>_detected.mp4            — bounding box overlay video
        <video_name>_text_heatmap.mp4        — CRAFT text score heatmap video
        <video_name>_link_heatmap.mp4        — CRAFT link score heatmap video
        <video_name>_text_mask_overlay.mp4   — text mask overlay video
        <video_name>_TextRegion.txt          — per-frame text region coordinates
        analysis/
          analysis_report.txt                — detailed analysis report
          statistics.json                    — machine-readable stats
          performance_analysis.png           — performance plots
        dct_frames/
          frame_0001_dct.jpg                 — DCT compressed frame image
          frame_0001_dct.gif                 — DCT quadtree GIF animation
          ...

Author: Text-Aware Compression System
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from scipy.fft import dctn, idctn
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Import user's DCT quadtree (zero changes to that file)
# ─────────────────────────────────────────────────────────────
from dct_quadtree import QuadTree, Quadrant, get_detail, average_colour, MAX_DEPTH, DETAIL_THRESHOLD

# ─────────────────────────────────────────────────────────────
# Import existing CRAFT pipeline (reused as-is)
# ─────────────────────────────────────────────────────────────
from cutils_mod import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_results
)


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 2 — Reference Frame Temporal Filtering
# ══════════════════════════════════════════════════════════════

class TemporalFilter:
    """
    Technique 2: Reduces redundant frame processing by comparing each
    incoming frame to a stored reference frame using SSIM.

    If the frames are visually similar (SSIM >= threshold), the frame is
    skipped and the reference is reused. Otherwise, the frame becomes the
    new reference and is forwarded to Technique 1 for spatial compression.

    Processing Steps (from the architecture prompt):
      1. Input Video
      2. Frame Extraction
      3. Reference Frame Initialization (first frame)
      4. Frame Difference Detection (SSIM comparison)
      5. Change Threshold Decision → skip or process
    """

    def __init__(self, ssim_threshold=0.95, comparison_method="ssim"):
        self.ssim_threshold = ssim_threshold
        self.comparison_method = comparison_method
        self.reference_frame = None
        self.frames_skipped = 0
        self.frames_processed = 0
        self.total_frames_seen = 0

    def _initialize_reference(self, frame):
        """Step 3: First frame becomes the initial reference frame."""
        self.reference_frame = frame.copy()

    def _compute_difference(self, frame):
        """
        Step 4: Frame Difference Detection.
        Supported methods: ssim, mse, pixel
        """
        ref_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.comparison_method == "ssim":
            similarity, _ = ssim(ref_gray, frame_gray, full=True)
            return similarity
        elif self.comparison_method == "mse":
            mse = np.mean((ref_gray.astype(np.float32) - frame_gray.astype(np.float32)) ** 2)
            return 1.0 / (1.0 + mse / 1000.0)
        elif self.comparison_method == "pixel":
            diff = np.mean(np.abs(ref_gray.astype(np.float32) - frame_gray.astype(np.float32)))
            return 1.0 - (diff / 255.0)
        else:
            raise ValueError(f"Unknown comparison method: {self.comparison_method}")

    def should_process(self, frame):
        """
        Step 5: Change Threshold Decision.
        Returns (should_process: bool, similarity_score: float).
        """
        self.total_frames_seen += 1

        if self.reference_frame is None:
            self._initialize_reference(frame)
            self.frames_processed += 1
            return True, 1.0

        similarity = self._compute_difference(frame)

        if similarity >= self.ssim_threshold:
            self.frames_skipped += 1
            return False, similarity
        else:
            self._initialize_reference(frame)
            self.frames_processed += 1
            return True, similarity

    def get_stats(self):
        skip_ratio = self.frames_skipped / max(self.total_frames_seen, 1)
        return {
            "total_frames_seen": self.total_frames_seen,
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "skip_ratio": skip_ratio,
            "computation_saved_pct": skip_ratio * 100,
        }


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 1 — STEP 1: Preprocessing Module
# ══════════════════════════════════════════════════════════════

class Preprocessor:
    """
    Technique 1, Step 1: Preprocessing Module.
    Operations: Frame resizing, normalization, optional noise reduction.
    """

    def __init__(self, target_size=None, apply_noise_reduction=False):
        self.target_size = target_size
        self.apply_noise_reduction = apply_noise_reduction

    def process(self, frame_bgr):
        original_size = (frame_bgr.shape[1], frame_bgr.shape[0])

        if self.target_size is not None:
            frame_bgr = cv2.resize(frame_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.apply_noise_reduction:
            frame_bgr = cv2.GaussianBlur(frame_bgr, (3, 3), 0)

        normalized = frame_bgr.astype(np.float32) / 255.0
        return frame_bgr, normalized, original_size


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 1 — STEPS 2–3: Text Detection + Mask Generation
# ══════════════════════════════════════════════════════════════

class TextDetector:
    """
    Technique 1, Steps 2–3: Text Detection (CRAFT + RefineNet) + Binary Mask.
    """

    def __init__(self, craft_net, refine_net, device="cpu",
                 text_threshold=0.85, link_threshold=0.6):
        self.craft_net = craft_net
        self.refine_net = refine_net
        self.device = device
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.use_gpu = (device in ("cuda", "mps"))

    def detect(self, frame_bgr):
        """Run CRAFT + RefineNet. Returns boxes and full prediction_result dict."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prediction_result = get_prediction(
            image=frame_rgb,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=0.4,
            cuda=self.use_gpu,
            long_size=1280,
        )
        return prediction_result["boxes"], prediction_result

    def create_text_mask(self, frame_shape, boxes):
        """Generate binary text mask from bounding boxes. 255=text, 0=bg."""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
            cv2.fillPoly(mask, [pts], 255)
        return mask


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 1 — STEPS 4–9: Adaptive Quadtree + DCT Compression
# ══════════════════════════════════════════════════════════════

class AdaptiveQuadtreeCompressor:
    """
    Technique 1, Steps 4–9: Adaptive Quadtree Partitioning + DCT Compression.
    """

    MIN_BLOCK_SIZE = 8
    TEXT_QUALITY = 90
    BACKGROUND_QUALITY = 30
    VARIANCE_THRESHOLD = 50.0

    def __init__(self, text_quality=None, background_quality=None,
                 variance_threshold=None, min_block_size=None):
        self.text_quality = text_quality or self.TEXT_QUALITY
        self.background_quality = background_quality or self.BACKGROUND_QUALITY
        self.variance_threshold = variance_threshold or self.VARIANCE_THRESHOLD
        self.min_block_size = min_block_size or self.MIN_BLOCK_SIZE
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {
            "total_blocks": 0, "text_blocks": 0, "background_blocks": 0,
            "uniform_blocks": 0, "simplified_blocks": 0,
        }

    def _partition(self, frame_gray, x, y, w, h, depth=0):
        block = frame_gray[y:h, x:w]
        bh, bw = h - y, w - x
        if bh <= 0 or bw <= 0:
            return []
        variance = np.var(block.astype(np.float32))
        if (bh <= self.min_block_size or bw <= self.min_block_size or
                depth >= MAX_DEPTH or variance <= self.VARIANCE_THRESHOLD):
            return [(x, y, w, h, depth, variance)]
        mid_x, mid_y = x + bw // 2, y + bh // 2
        leaves = []
        leaves += self._partition(frame_gray, x,     y,     mid_x, mid_y, depth + 1)
        leaves += self._partition(frame_gray, mid_x, y,     w,     mid_y, depth + 1)
        leaves += self._partition(frame_gray, x,     mid_y, mid_x, h,     depth + 1)
        leaves += self._partition(frame_gray, mid_x, mid_y, w,     h,     depth + 1)
        return leaves

    def _classify_block(self, text_mask, x, y, w, h):
        return np.any(text_mask[y:h, x:w] > 0)

    def _quality_to_step(self, quality):
        quality = max(1, min(100, quality))
        if quality < 50:
            step = 5000 / quality
        else:
            step = 200 - 2 * quality
        return max(1.0, step)

    def _dct_compress_block(self, block_float, quality):
        step = self._quality_to_step(quality)
        dct_coeffs = dctn(block_float, type=2, norm="ortho")
        quantized = np.round(dct_coeffs / step) * step
        reconstructed = idctn(quantized, type=2, norm="ortho")
        return np.clip(reconstructed, 0, 255)

    def _simplify_block(self, block_float):
        return np.full_like(block_float, np.mean(block_float))

    def compress_frame(self, frame_bgr, text_mask):
        """Steps 4–9: Partition → Classify → DCT → Reconstruct."""
        self._reset_stats()
        h, w = frame_bgr.shape[:2]
        compressed = np.zeros_like(frame_bgr, dtype=np.float32)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        leaf_blocks = self._partition(gray, 0, 0, w, h, depth=0)

        for (bx, by, bw, bh, depth, variance) in leaf_blocks:
            self.stats["total_blocks"] += 1
            is_uniform = variance <= self.VARIANCE_THRESHOLD
            is_text = self._classify_block(text_mask, bx, by, bw, bh)

            if is_text:
                self.stats["text_blocks"] += 1
                quality = self.text_quality
            else:
                self.stats["background_blocks"] += 1
                quality = self.background_quality

            if is_uniform and not is_text:
                self.stats["uniform_blocks"] += 1
                if (bh - by) >= 16 and (bw - bx) >= 16:
                    self.stats["simplified_blocks"] += 1
                    for c in range(3):
                        compressed[by:bh, bx:bw, c] = self._simplify_block(
                            frame_bgr[by:bh, bx:bw, c].astype(np.float32))
                    continue

            for c in range(3):
                compressed[by:bh, bx:bw, c] = self._dct_compress_block(
                    frame_bgr[by:bh, bx:bw, c].astype(np.float32), quality)

        compressed_frame = np.clip(compressed, 0, 255).astype(np.uint8)
        return compressed_frame, dict(self.stats)


# ══════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

def create_analysis_plots(stats, analysis_folder):
    """Create visualization plots for performance analysis."""
    frames = list(range(1, len(stats['frame_times']) + 1))
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Video Processing Analysis', fontsize=16, fontweight='bold')

    axes[0, 0].plot(frames, stats['frame_times'], color='blue', linewidth=0.5)
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Frame Processing Time')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(frames, stats['text_regions_per_frame'], color='green', linewidth=0.5)
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Number of Regions')
    axes[0, 1].set_title('Text Regions Detected per Frame')
    axes[0, 1].grid(True, alpha=0.3)

    component_names = ['Temporal\nFilter', 'Text\nDetection', 'DCT\nCompression']
    component_times = [
        np.mean(stats.get('temporal_filter_times', [0])),
        np.mean(stats.get('text_detection_times', [0])),
        np.mean(stats.get('dct_compression_times', [0])),
    ]
    axes[1, 0].bar(component_names, component_times,
                   color=['#1f77b4', '#2ca02c', '#d62728'])
    axes[1, 0].set_ylabel('Average Time (seconds)')
    axes[1, 0].set_title('Average Processing Time by Component')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].hist(stats['frame_times'], bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Processing Time Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    cumulative_time = np.cumsum(stats['frame_times'])
    axes[2, 0].plot(frames, cumulative_time, color='red', linewidth=1)
    axes[2, 0].set_xlabel('Frame Number')
    axes[2, 0].set_ylabel('Cumulative Time (seconds)')
    axes[2, 0].set_title('Cumulative Processing Time')
    axes[2, 0].grid(True, alpha=0.3)

    # Frames processed vs skipped pie chart
    tf = stats.get('temporal_filter_stats', {})
    processed = tf.get('frames_processed', len(frames))
    skipped = tf.get('frames_skipped', 0)
    if processed + skipped > 0:
        axes[2, 1].pie([processed, skipped],
                       labels=['Processed', 'Skipped'],
                       colors=['#2ca02c', '#ff7f0e'],
                       autopct='%1.1f%%', startangle=90)
        axes[2, 1].set_title('Technique 2: Frames Processed vs Skipped')
    else:
        axes[2, 1].text(0.5, 0.5, 'No data', ha='center', va='center')

    plt.tight_layout()
    plot_path = os.path.join(analysis_folder, 'performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path


def generate_analysis_report(stats, video_name, analysis_folder):
    """Generate detailed analysis report."""
    report_path = os.path.join(analysis_folder, 'analysis_report.txt')
    tf = stats.get('temporal_filter_stats', {})

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TEXT-AWARE VIDEO COMPRESSION ANALYSIS REPORT\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Frames in Video:    {stats['total_frames']}\n")
        f.write(f"Frames Processed (T1):    {tf.get('frames_processed', 'N/A')}\n")
        f.write(f"Frames Skipped (T2):      {tf.get('frames_skipped', 'N/A')}\n")
        f.write(f"Computation Saved:        {tf.get('computation_saved_pct', 0):.1f}%\n")
        f.write(f"Total Processing Time:    {stats['total_time']:.2f}s ({stats['total_time']/60:.1f} min)\n")
        f.write(f"Average Time per Frame:   {stats['avg_time_per_frame']:.4f}s\n")
        f.write(f"Average Processing FPS:   {stats['avg_fps']:.2f}\n")
        f.write(f"Video FPS:                {stats['video_fps']}\n")
        f.write(f"Total Text Regions:       {stats['total_text_regions']}\n")
        f.write(f"Average Regions/Frame:    {stats['avg_regions_per_frame']:.2f}\n")
        f.write(f"Frames with Text:         {stats['frames_with_text']}\n")
        f.write(f"Frames without Text:      {stats['frames_without_text']}\n")
        f.write(f"DCT Frames Saved:         {stats.get('dct_frames_saved', 0)}\n\n")

        f.write("TECHNIQUE 2 — TEMPORAL FILTER\n")
        f.write("-" * 80 + "\n")
        f.write(f"SSIM Threshold:           {stats.get('ssim_threshold', 'N/A')}\n")
        f.write(f"Comparison Method:        {stats.get('comparison_method', 'N/A')}\n")
        f.write(f"Skip Ratio:               {tf.get('skip_ratio', 0):.2%}\n\n")

        f.write("TECHNIQUE 1 — SPATIAL COMPRESSION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Text Blocks:        {stats.get('total_text_blocks', 0)}\n")
        f.write(f"Total Background Blocks:  {stats.get('total_bg_blocks', 0)}\n")
        f.write(f"Simplified Blocks:        {stats.get('total_simplified', 0)}\n")
        f.write(f"Text Quality Factor:      {stats.get('text_quality', 'N/A')}/100\n")
        f.write(f"Background Quality Factor: {stats.get('bg_quality', 'N/A')}/100\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Device:                   {stats.get('device', 'N/A')}\n")
        f.write(f"Resolution:               {stats.get('resolution', 'N/A')}\n")
        f.write(f"Text Threshold:           {stats.get('text_threshold', 'N/A')}\n")
        f.write(f"Link Threshold:           {stats.get('link_threshold', 'N/A')}\n")
        f.write(f"RefineNet:                {stats.get('use_refinenet', 'N/A')}\n")
        f.write("\n")

    return report_path


def save_statistics_json(stats, analysis_folder):
    """Save statistics as JSON for further analysis."""
    json_path = os.path.join(analysis_folder, 'statistics.json')
    json_stats = {}
    for key, value in stats.items():
        if isinstance(value, (list, np.ndarray)):
            json_stats[key] = [float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for v in value]
        elif isinstance(value, (np.floating, float)):
            json_stats[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            json_stats[key] = int(value)
        elif isinstance(value, dict):
            json_stats[key] = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in value.items()}
        else:
            json_stats[key] = value
    with open(json_path, 'w') as f:
        json.dump(json_stats, f, indent=4)
    return json_path


# ══════════════════════════════════════════════════════════════
# FULL PIPELINE — TextAwareVideoCompressor
# ══════════════════════════════════════════════════════════════

class TextAwareVideoCompressor:
    """
    Full System Orchestrator.

    Combines Technique 2 (temporal filtering) and Technique 1 (spatial compression)
    to process an input video and produce a compressed output video.

    Output directory structure for each video:
        output/<video_name>/
            <video_name>_compressed.mp4
            <video_name>_detected.mp4
            <video_name>_text_heatmap.mp4
            <video_name>_link_heatmap.mp4
            <video_name>_text_mask_overlay.mp4
            <video_name>_TextRegion.txt
            analysis/
                analysis_report.txt
                statistics.json
                performance_analysis.png
            dct_frames/
                frame_0001_dct.jpg
                frame_0001_dct.gif
                ...
    """

    def __init__(self,
                 ssim_threshold=0.95,
                 comparison_method="ssim",
                 text_threshold=0.85,
                 link_threshold=0.6,
                 text_quality=90,
                 background_quality=30,
                 target_size=None,
                 apply_noise_reduction=False,
                 use_refinenet=True):
        self.ssim_threshold = ssim_threshold
        self.comparison_method = comparison_method
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.text_quality = text_quality
        self.background_quality = background_quality
        self.target_size = target_size
        self.apply_noise_reduction = apply_noise_reduction
        self.use_refinenet = use_refinenet

        self._temporal_filter = None
        self._preprocessor = None
        self._text_detector = None
        self._compressor = None
        self._craft_net = None
        self._refine_net = None
        self._device = None

    def _check_device(self):
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_models(self):
        use_gpu = self._device in ("cuda", "mps")
        print(f"  Loading CRAFT model...")
        self._craft_net = load_craftnet_model(cuda=use_gpu)
        if self.use_refinenet:
            print(f"  Loading RefineNet model...")
            self._refine_net = load_refinenet_model(cuda=use_gpu)
        else:
            self._refine_net = None
            print(f"  RefineNet disabled.")

    def _initialize_modules(self):
        self._device = self._check_device()
        print(f"  Device: {self._device.upper()}")
        self._load_models()

        self._temporal_filter = TemporalFilter(
            ssim_threshold=self.ssim_threshold,
            comparison_method=self.comparison_method
        )
        self._preprocessor = Preprocessor(
            target_size=self.target_size,
            apply_noise_reduction=self.apply_noise_reduction
        )
        self._text_detector = TextDetector(
            craft_net=self._craft_net,
            refine_net=self._refine_net,
            device=self._device,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold
        )
        self._compressor = AdaptiveQuadtreeCompressor(
            text_quality=self.text_quality,
            background_quality=self.background_quality
        )

    def process(self, input_video_path, base_output_dir="output"):
        """
        Full pipeline: read input video → compress → write all outputs.

        Creates a per-video folder inside base_output_dir:
            base_output_dir/<video_name>/
                ├── <video_name>_compressed.mp4
                ├── <video_name>_detected.mp4
                ├── <video_name>_text_heatmap.mp4
                ├── <video_name>_link_heatmap.mp4
                ├── <video_name>_text_mask_overlay.mp4
                ├── <video_name>_TextRegion.txt
                ├── analysis/
                │   ├── analysis_report.txt
                │   ├── statistics.json
                │   └── performance_analysis.png
                └── dct_frames/
                    ├── frame_0001_dct.jpg
                    ├── frame_0001_dct.gif
                    └── ...

        Args:
            input_video_path (str): Path to the input video file.
            base_output_dir (str):  Base output directory (default: "output").

        Returns:
            dict: Full processing statistics.
        """
        # ── Video Name & Directory Setup ───────────────────────
        video_name = Path(input_video_path).stem
        video_output_dir = os.path.join(base_output_dir, video_name)
        analysis_dir     = os.path.join(video_output_dir, "analysis")
        dct_frames_dir   = os.path.join(video_output_dir, "dct_frames")

        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(dct_frames_dir, exist_ok=True)

        # Output file paths
        compressed_path   = os.path.join(video_output_dir, f"{video_name}_compressed.mp4")
        detected_path     = os.path.join(video_output_dir, f"{video_name}_detected.mp4")
        text_heatmap_path = os.path.join(video_output_dir, f"{video_name}_text_heatmap.mp4")
        link_heatmap_path = os.path.join(video_output_dir, f"{video_name}_link_heatmap.mp4")
        mask_overlay_path = os.path.join(video_output_dir, f"{video_name}_text_mask_overlay.mp4")
        text_region_path  = os.path.join(video_output_dir, f"{video_name}_TextRegion.txt")

        print("\n" + "=" * 70)
        print("TEXT-AWARE VIDEO COMPRESSION SYSTEM")
        print("=" * 70)
        print(f"Input:       {input_video_path}")
        print(f"Output dir:  {video_output_dir}")
        print()

        print("[INIT] Initializing modules...")
        self._initialize_modules()

        # ── Open Input Video ───────────────────────────────────
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS)
        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Resolution: {frame_w}×{frame_h}  |  FPS: {fps:.1f}  |  Frames: {total_frames}")

        # ── Read first frame to get heatmap dimensions ─────────
        ret, first_frame = cap.read()
        if not ret:
            raise IOError(f"Cannot read first frame from {input_video_path}")

        use_device = self._device in ("mps", "cuda")
        temp_pred = get_prediction(
            image=cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
            craft_net=self._craft_net,
            refine_net=self._refine_net if self.use_refinenet else None,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=0.4,
            cuda=use_device,
            long_size=1280
        )
        t1_hm = temp_pred["heatmaps"]["text_score_heatmap"]
        t2_hm = temp_pred["heatmaps"]["link_score_heatmap"]

        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ── Output Video Writers ───────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_compressed = cv2.VideoWriter(compressed_path,   fourcc, fps, (frame_w, frame_h))
        out_detected   = cv2.VideoWriter(detected_path,     fourcc, fps, (frame_w, frame_h))
        out_text_hm    = cv2.VideoWriter(text_heatmap_path, fourcc, fps, (len(t1_hm[0]), len(t1_hm)))
        out_link_hm    = cv2.VideoWriter(link_heatmap_path, fourcc, fps, (len(t2_hm[0]), len(t2_hm)))
        out_mask       = cv2.VideoWriter(mask_overlay_path, fourcc, fps, (frame_w, frame_h))

        # ── Text Region File ──────────────────────────────────
        f_txt = open(text_region_path, "w")
        f_txt.write("=" * 100 + "\n")
        f_txt.write(f"TEXT DETECTION RESULTS\n")
        f_txt.write(f"Video: {video_name}\n")
        f_txt.write(f"Resolution: {frame_w}x{frame_h}, FPS: {fps}\n")
        f_txt.write(f"Thresholds: text={self.text_threshold}, link={self.link_threshold}\n")
        f_txt.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_txt.write("=" * 100 + "\n\n")

        # ── Statistics Tracking ────────────────────────────────
        stats = {
            'frame_times': [],
            'text_regions_per_frame': [],
            'temporal_filter_times': [],
            'text_detection_times': [],
            'dct_compression_times': [],
            'total_frames': 0,
            'total_text_regions': 0,
            'frames_with_text': 0,
            'frames_without_text': 0,
            'total_text_blocks': 0,
            'total_bg_blocks': 0,
            'total_simplified': 0,
            'dct_frames_saved': 0,
            'video_fps': fps,
            'resolution': f"{frame_w}x{frame_h}",
            'device': self._device.upper(),
            'text_threshold': self.text_threshold,
            'link_threshold': self.link_threshold,
            'ssim_threshold': self.ssim_threshold,
            'comparison_method': self.comparison_method,
            'text_quality': self.text_quality,
            'bg_quality': self.background_quality,
            'use_refinenet': self.use_refinenet,
        }

        # Track state
        last_compressed_frame = None
        last_mask_overlay     = None
        last_detected_frame   = None
        process_time_start    = time.time()
        frame_idx             = 0
        processed_frame_count = 0  # Counter for frames actually processed by Technique 1

        print(f"\n[START] Processing frames...")
        print("-" * 70)

        # ══════════════════════════════════════════════════════
        # MAIN FRAME LOOP
        # ══════════════════════════════════════════════════════
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            stats['total_frames'] += 1
            t_frame_start = time.time()

            # ──────────────────────────────────────────────────
            # TECHNIQUE 2: Temporal Filter
            # ──────────────────────────────────────────────────
            t_temporal_start = time.time()
            should_proc, similarity = self._temporal_filter.should_process(frame)
            t_temporal = time.time() - t_temporal_start
            stats['temporal_filter_times'].append(t_temporal)

            if not should_proc:
                # Frame SKIPPED — reuse last outputs
                wr_comp = last_compressed_frame if last_compressed_frame is not None else frame
                wr_mask = last_mask_overlay if last_mask_overlay is not None else np.zeros_like(frame)
                wr_det  = last_detected_frame if last_detected_frame is not None else frame

                out_compressed.write(wr_comp)
                out_mask.write(wr_mask)
                out_detected.write(wr_det)
                # Write blank heatmaps for skipped frames
                out_text_hm.write(np.zeros_like(t1_hm))
                out_link_hm.write(np.zeros_like(t2_hm))

                frame_time = time.time() - t_frame_start
                stats['frame_times'].append(frame_time)
                stats['text_regions_per_frame'].append(0)
                stats['text_detection_times'].append(0)
                stats['dct_compression_times'].append(0)

                f_txt.write(f"Frame {frame_idx} (SKIPPED, similarity={similarity:.4f}):\n")
                f_txt.write("  Frame skipped by Technique 2 (temporal filter)\n\n")

                if frame_idx % 30 == 0:
                    elapsed = time.time() - process_time_start
                    print(f"  Frame {frame_idx:>5}/{total_frames}  SKIPPED  "
                          f"(similarity={similarity:.3f})  [{elapsed:.1f}s]")
                continue

            # ──────────────────────────────────────────────────
            # TECHNIQUE 1: Text-Aware Spatial Compression
            # ──────────────────────────────────────────────────
            processed_frame_count += 1

            # Step 1: Preprocessing
            proc_frame, _, orig_size = self._preprocessor.process(frame)

            # Steps 2–3: Text Detection + Binary Mask
            t_detect_start = time.time()
            boxes, pred_result = self._text_detector.detect(proc_frame)
            text_mask = self._text_detector.create_text_mask(proc_frame.shape, boxes)
            t_detect = time.time() - t_detect_start
            stats['text_detection_times'].append(t_detect)

            # If frame was resized in preprocessing, resize back
            if self.target_size is not None:
                proc_frame = cv2.resize(proc_frame, orig_size)
                text_mask  = cv2.resize(text_mask, orig_size, interpolation=cv2.INTER_NEAREST)

            # Steps 4–9: Adaptive Quadtree + DCT Compression
            t_dct_start = time.time()
            compressed_frame, block_stats = self._compressor.compress_frame(proc_frame, text_mask)
            t_dct = time.time() - t_dct_start
            stats['dct_compression_times'].append(t_dct)

            # ── Build output frames ───────────────────────────

            # 1) Bounding box overlay (detected video)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_rgb = frame_rgb.copy()
            for region in boxes:
                region_arr = np.array(region).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(detected_rgb, [region_arr], True, color=(0, 0, 255), thickness=2)
            detected_bgr = cv2.cvtColor(detected_rgb, cv2.COLOR_RGB2BGR)

            # 2) Text mask overlay (green tint on text regions)
            mask_overlay = compressed_frame.copy()
            text_pixels = text_mask > 0
            if np.any(text_pixels):
                mask_overlay[text_pixels] = (
                    mask_overlay[text_pixels] * 0.5 +
                    np.array([0, 200, 0], dtype=np.float32) * 0.5
                ).astype(np.uint8)

            # 3) Heatmaps from CRAFT prediction
            heatmaps = pred_result["heatmaps"]

            # ── Write all output videos ───────────────────────
            out_compressed.write(compressed_frame)
            out_detected.write(detected_bgr)
            out_mask.write(mask_overlay)
            out_text_hm.write(heatmaps["text_score_heatmap"])
            out_link_hm.write(cv2.cvtColor(heatmaps["link_score_heatmap"], cv2.COLOR_RGB2BGR))

            # Update references for skipped frames
            last_compressed_frame = compressed_frame
            last_mask_overlay     = mask_overlay
            last_detected_frame   = detected_bgr

            # ── Save DCT frame image + GIF ────────────────────
            # Use the user's QuadTree to generate a DCT quadtree image and GIF
            # for every processed frame
            try:
                # Convert compressed frame to PIL Image for QuadTree
                compressed_rgb = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(compressed_rgb)

                # Build quadtree using the user's QuadTree class (dct_quadtree.py)
                quadtree = QuadTree(pil_image)

                # Save DCT compressed frame image
                dct_depth = min(7, quadtree.max_depth)
                dct_image = quadtree.create_image(dct_depth, show_lines=False)
                dct_jpg_path = os.path.join(dct_frames_dir,
                                            f"frame_{processed_frame_count:04d}_dct.jpg")
                dct_image.save(dct_jpg_path)

                # Save DCT quadtree GIF animation
                dct_gif_path = os.path.join(dct_frames_dir,
                                            f"frame_{processed_frame_count:04d}_dct.gif")
                quadtree.create_gif(dct_gif_path, duration=500, show_lines=True)

                stats['dct_frames_saved'] += 1
            except Exception as e:
                print(f"    [WARN] DCT frame save failed for frame {frame_idx}: {e}")

            # ── Accumulate stats ──────────────────────────────
            n_regions = len(boxes)
            stats['total_text_regions'] += n_regions
            stats['text_regions_per_frame'].append(n_regions)
            stats['total_text_blocks']  += block_stats["text_blocks"]
            stats['total_bg_blocks']    += block_stats["background_blocks"]
            stats['total_simplified']   += block_stats["simplified_blocks"]

            if n_regions > 0:
                stats['frames_with_text'] += 1
            else:
                stats['frames_without_text'] += 1

            frame_time = time.time() - t_frame_start
            stats['frame_times'].append(frame_time)

            # ── Write to text region file ─────────────────────
            f_txt.write(f"Frame {frame_idx} (Time: {frame_time:.4f}s, "
                        f"FPS: {1/frame_time:.2f}, similarity={similarity:.4f}):\n")
            if n_regions > 0:
                f_txt.write(f"  Detected {n_regions} text region(s):\n")
                for i, region in enumerate(boxes):
                    region_flat = np.array(region).astype(np.int32).reshape((-1))
                    f_txt.write(f"    Region {i+1}: [{','.join(str(r) for r in region_flat)}]\n")
            else:
                f_txt.write("  No text regions detected\n")
            f_txt.write(f"  Blocks: text={block_stats['text_blocks']}, "
                        f"bg={block_stats['background_blocks']}, "
                        f"simplified={block_stats['simplified_blocks']}\n\n")

            # ── Progress log ──────────────────────────────────
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                elapsed = time.time() - process_time_start
                avg_fps = frame_idx / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_idx) / avg_fps if avg_fps > 0 else 0
                print(f"  Frame {frame_idx:>5}/{total_frames}  "
                      f"| txt_regions={n_regions:>3} "
                      f"| similarity={similarity:.3f} "
                      f"| {frame_time:.2f}s/frame "
                      f"| ETA={eta:.0f}s")

        # ── Release Resources ──────────────────────────────────
        cap.release()
        out_compressed.release()
        out_detected.release()
        out_text_hm.release()
        out_link_hm.release()
        out_mask.release()
        f_txt.close()

        total_time = time.time() - process_time_start
        temp_stats = self._temporal_filter.get_stats()

        # ── Finalize stats ─────────────────────────────────────
        stats['total_time'] = total_time
        stats['avg_time_per_frame'] = np.mean(stats['frame_times']) if stats['frame_times'] else 0
        stats['avg_fps'] = 1 / stats['avg_time_per_frame'] if stats['avg_time_per_frame'] > 0 else 0
        stats['avg_regions_per_frame'] = stats['total_text_regions'] / max(stats['total_frames'], 1)
        stats['temporal_filter_stats'] = temp_stats

        # ── Generate Analysis Outputs ──────────────────────────
        print(f"\n[ANALYSIS] Generating analysis reports in {analysis_dir}...")
        report_path = generate_analysis_report(stats, video_name, analysis_dir)
        json_path   = save_statistics_json(stats, analysis_dir)
        try:
            plot_path = create_analysis_plots(stats, analysis_dir)
            print(f"  Plots saved:    {plot_path}")
        except Exception as e:
            print(f"  [WARN] Plot generation failed: {e}")
        print(f"  Report saved:   {report_path}")
        print(f"  Stats saved:    {json_path}")

        # ── Print Summary ──────────────────────────────────────
        print("\n" + "=" * 70)
        print("COMPRESSION COMPLETE")
        print("=" * 70)
        print(f"  Total frames:          {total_frames}")
        print(f"  Frames processed (T1): {temp_stats['frames_processed']}")
        print(f"  Frames skipped (T2):   {temp_stats['frames_skipped']}")
        print(f"  Computation saved:     {temp_stats['computation_saved_pct']:.1f}%")
        print(f"  Total text regions:    {stats['total_text_regions']}")
        print(f"  DCT frames saved:      {stats['dct_frames_saved']}")
        print(f"  Total time:            {total_time:.1f}s")
        print()
        print(f"  OUTPUT DIRECTORY: {video_output_dir}/")
        print(f"  ├── {video_name}_compressed.mp4")
        print(f"  ├── {video_name}_detected.mp4")
        print(f"  ├── {video_name}_text_heatmap.mp4")
        print(f"  ├── {video_name}_link_heatmap.mp4")
        print(f"  ├── {video_name}_text_mask_overlay.mp4")
        print(f"  ├── {video_name}_TextRegion.txt")
        print(f"  ├── analysis/")
        print(f"  │   ├── analysis_report.txt")
        print(f"  │   ├── statistics.json")
        print(f"  │   └── performance_analysis.png")
        print(f"  └── dct_frames/")
        print(f"      └── ({stats['dct_frames_saved']} frames: _dct.jpg + _dct.gif)")
        print("=" * 70)

        return stats


# ══════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER
# ══════════════════════════════════════════════════════════════

def compress_video(input_path, base_output_dir="output", **kwargs):
    """
    Convenience function to compress a single video.

    Args:
        input_path (str):       Path to the input video.
        base_output_dir (str):  Base output directory (default: "output").
                                A subfolder <video_name>/ is created inside.
        **kwargs:               Any parameter accepted by TextAwareVideoCompressor.__init__

    Returns:
        dict: Processing statistics.
    """
    compressor = TextAwareVideoCompressor(**kwargs)
    return compressor.process(input_path, base_output_dir=base_output_dir)


def compress_multiple_videos(video_list, base_output_dir="output", **kwargs):
    """
    Compress multiple videos. Each gets its own subfolder in base_output_dir.

    Args:
        video_list (list):      List of video file paths.
        base_output_dir (str):  Base output directory.
        **kwargs:               Any parameter accepted by TextAwareVideoCompressor.__init__

    Returns:
        list: List of (video_name, stats_dict) tuples.
    """
    compressor = TextAwareVideoCompressor(**kwargs)

    # Initialize models once for all videos
    print("=" * 70)
    print("TEXT-AWARE VIDEO COMPRESSION — MULTI-VIDEO MODE")
    print(f"Videos to process: {len(video_list)}")
    print("=" * 70)

    results = []
    for idx, video_path in enumerate(video_list, 1):
        print(f"\n{'='*70}")
        print(f"VIDEO {idx}/{len(video_list)}: {Path(video_path).name}")
        print(f"{'='*70}")

        if not os.path.exists(video_path):
            print(f"  [ERROR] Video not found: {video_path} — SKIPPED")
            continue

        try:
            stats = compressor.process(video_path, base_output_dir=base_output_dir)
            results.append((Path(video_path).stem, stats))
        except Exception as e:
            print(f"  [ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Re-initialize temporal filter for next video
        compressor._temporal_filter = TemporalFilter(
            ssim_threshold=compressor.ssim_threshold,
            comparison_method=compressor.comparison_method
        )

    return results
