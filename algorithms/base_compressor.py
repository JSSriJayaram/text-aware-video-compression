"""
base_compressor.py
==================
Abstract base class that every compression algorithm must implement.
Ensures a standard interface so the benchmark runner can treat all
algorithms identically.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional
import os


@dataclass
class CompressResult:
    """Result object returned by every compressor's compress() method."""
    algorithm:          str   = ""
    device:             str   = ""          # "mps" | "cpu" | "videotoolbox"
    input_path:         str   = ""
    output_path:        str   = ""
    target_ssim:        float = 0.0

    # Size metrics
    original_size_mb:   float = 0.0
    compressed_size_mb: float = 0.0
    space_saved_pct:    float = 0.0
    compression_ratio:  float = 0.0

    # Quality metrics
    achieved_ssim:      float = 0.0
    achieved_psnr:      float = 0.0

    # Time metrics
    encode_time_s:      float = 0.0

    # Optional internals
    frames_total:       int   = 0
    frames_skipped:     int   = 0
    frames_processed:   int   = 0
    extra:              dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "algorithm":          self.algorithm,
            "device":             self.device,
            "input_path":         self.input_path,
            "output_path":        self.output_path,
            "target_ssim":        self.target_ssim,
            "original_size_mb":   self.original_size_mb,
            "compressed_size_mb": self.compressed_size_mb,
            "space_saved_pct":    self.space_saved_pct,
            "compression_ratio":  self.compression_ratio,
            "achieved_ssim":      self.achieved_ssim,
            "achieved_psnr":      self.achieved_psnr,
            "encode_time_s":      self.encode_time_s,
            "frames_total":       self.frames_total,
            "frames_skipped":     self.frames_skipped,
            "frames_processed":   self.frames_processed,
            **self.extra,
        }


class BaseCompressor(ABC):
    """
    Every compression algorithm inherits from this class and
    implements the compress() method.
    """

    NAME: str = "base"      # Override in subclass

    def __init__(self, target_ssim: float = 0.90):
        self.target_ssim = target_ssim

    @abstractmethod
    def compress(
        self,
        input_path:  str,
        output_path: str,
        target_ssim: float,
        device:      str = "cpu",
        max_frames:  Optional[int] = None,
    ) -> CompressResult:
        """
        Compress the video and return a CompressResult.

        Args:
            input_path:  Path to source video.
            output_path: Where to write the compressed video.
            target_ssim: Desired SSIM quality (0–1). Higher = better quality.
            device:      "mps" or "cpu" (or "videotoolbox" for HW-accelerated ffmpeg).
            max_frames:  If set, only compress this many frames (for quick tests).

        Returns:
            CompressResult with all measured metrics.
        """

    # ── Shared helpers ──────────────────────────────────────────

    @staticmethod
    def file_size_mb(path: str) -> float:
        if os.path.exists(path):
            return os.path.getsize(path) / (1024 * 1024)
        return 0.0

    @staticmethod
    def ensure_output_dir(output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
