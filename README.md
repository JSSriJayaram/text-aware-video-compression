# Text-Aware Video Compression System

A novel approach to video compression that prioritizes text preservation using a combination of CRAFT (Character Region Awareness for Text Detection) and adaptive DCT (Discrete Cosine Transform) Quadtree compression.

## Features

1. **Reference Frame Temporal Filtering (SSIM-based frame skipping)**: Reduces redundant frame processing by comparing each incoming frame to a stored reference frame using SSIM.
2. **Text-Aware Spatial Compression**:
    - **Text Detection**: Uses CRAFT and RefineNet to accurately identify text regions in frames.
    - **Adaptive Quadtree Partitioning**: Intelligently divides frames based on detail and text presence.
    - **DCT-based Compression**: Applies differential quantization, retaining high fidelity in text regions and simplifying uniform backgrounds.

## Project Structure

- `text_aware_compressor.py`: Core logic for the compression system, including temporal filtering, preprocessing, text detection, and DCT block classification.
- `dct_quadtree.py`: Implementation of the DCT quadtree algorithm.
- `cutils_mod.py`: Modified utilities for loading and running the CRAFT and RefineNet models.
- `run_compression.py`: Script to run the compression on a single video.
- `run_benchmark.py` & `benchmark_runner.py`: Scripts for benchmarking the compression algorithm against standard codecs (H.264, HEVC, VP9).
- `chart_generator.py` & `generate_final_report.py`: Utilities for visualizing performance and generating statistics.
- `optimizer.py`: COBYLA optimizer integration to find the best compression parameters.

## Output format

For each processed video, the system generates an output directory containing:
- Compressed video (`<video_name>_compressed.mp4`)
- Text mask overlay video, bounding box video, and heatmap visualizations
- Per-frame text region coordinates
- Detailed JSON statistics and analysis reports
- DCT frame images and quadtree GIFs

## Getting Started

### Prerequisites
Make sure you have Python installed along with the necessary dependencies:
- OpenCV (`cv2`)
- NumPy
- SciPy
- scikit-image
- PyTorch (for CRAFT)
- Matplotlib

### Running the Compression
To compress a video, you can use the `run_compression.py` script:
```bash
python run_compression.py
```
*(Add specific arguments as needed depending on your script's setup)*

## License
MIT License (or specify your license here)
