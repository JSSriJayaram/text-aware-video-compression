import cv2
import numpy as np
import os
import time
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import torch

from cutils_mod import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_results
)

def check_device():
    """
    Check available device (MPS for Mac M1/M2/M3/M4, CUDA for NVIDIA, or CPU)
    """
    if torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon GPU (MPS)"
        print(f"✓ Using: {device_name}")
    elif torch.cuda.is_available():
        device = "cuda"
        device_name = f"NVIDIA GPU: {torch.cuda.get_device_name(0)}"
        print(f"✓ Using: {device_name}")
    else:
        device = "cpu"
        device_name = "CPU (No GPU acceleration)"
        print(f"⚠ Using: {device_name}")
        print("  Note: Processing will be slow. Consider using MPS if on Mac M-series.")
    
    return device, device_name

def create_analysis_plots(stats, output_folder):
    """
    Create visualization plots for performance analysis
    """
    frames = list(range(1, len(stats['frame_times']) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Video Processing Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Frame Processing Time
    axes[0, 0].plot(frames, stats['frame_times'], color='blue', linewidth=0.5)
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Frame Processing Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Text Regions Detected per Frame
    axes[0, 1].plot(frames, stats['text_regions_per_frame'], color='green', linewidth=0.5)
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Number of Regions')
    axes[0, 1].set_title('Text Regions Detected per Frame')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average Component Times
    component_names = ['Resize', 'Preprocess', 'CraftNet', 'RefineNet', 'Postprocess']
    component_times = [
        np.mean(stats['resize_times']),
        np.mean(stats['preprocessing_times']),
        np.mean(stats['craftnet_times']),
        np.mean(stats['refinenet_times']),
        np.mean(stats['postprocess_times'])
    ]
    axes[1, 0].bar(component_names, component_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[1, 0].set_ylabel('Average Time (seconds)')
    axes[1, 0].set_title('Average Processing Time by Component')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Processing Time Distribution (Histogram)
    axes[1, 1].hist(stats['frame_times'], bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Processing Time Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Cumulative Processing Time
    cumulative_time = np.cumsum(stats['frame_times'])
    axes[2, 0].plot(frames, cumulative_time, color='red', linewidth=1)
    axes[2, 0].set_xlabel('Frame Number')
    axes[2, 0].set_ylabel('Cumulative Time (seconds)')
    axes[2, 0].set_title('Cumulative Processing Time')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: FPS Performance
    fps_values = [1/t if t > 0 else 0 for t in stats['frame_times']]
    axes[2, 1].plot(frames, fps_values, color='orange', linewidth=0.5)
    axes[2, 1].set_xlabel('Frame Number')
    axes[2, 1].set_ylabel('FPS')
    axes[2, 1].set_title('Processing FPS per Frame')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_analysis_report(stats, video_name, output_folder):
    """
    Generate detailed analysis report
    """
    report_path = os.path.join(output_folder, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"VIDEO PROCESSING ANALYSIS REPORT\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Frames Processed: {stats['total_frames']}\n")
        f.write(f"Total Processing Time: {stats['total_time']:.2f} seconds ({stats['total_time']/60:.1f} minutes)\n")
        f.write(f"Average Time per Frame: {stats['avg_time_per_frame']:.4f} seconds\n")
        f.write(f"Average Processing FPS: {stats['avg_fps']:.2f}\n")
        f.write(f"Video FPS: {stats['video_fps']}\n")
        f.write(f"Real-time Factor: {stats['realtime_factor']:.2f}x\n")
        if stats['realtime_factor'] >= 1.0:
            f.write(f"  → Can process REAL-TIME! ✓\n")
        else:
            f.write(f"  → Need {1/stats['realtime_factor']:.1f}x speedup for real-time\n")
        f.write(f"Total Text Regions Detected: {stats['total_text_regions']}\n")
        f.write(f"Average Regions per Frame: {stats['avg_regions_per_frame']:.2f}\n")
        f.write(f"Frames with Text: {stats['frames_with_text']} ({stats['frames_with_text']/stats['total_frames']*100:.1f}%)\n")
        f.write(f"Frames without Text: {stats['frames_without_text']} ({stats['frames_without_text']/stats['total_frames']*100:.1f}%)\n")
        f.write("\n")
        
        # Component Time Breakdown
        f.write("COMPONENT TIME BREAKDOWN (Average)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Resize Time: {np.mean(stats['resize_times']):.4f} seconds ({np.mean(stats['resize_times'])/stats['avg_time_per_frame']*100:.1f}%)\n")
        f.write(f"Preprocessing Time: {np.mean(stats['preprocessing_times']):.4f} seconds ({np.mean(stats['preprocessing_times'])/stats['avg_time_per_frame']*100:.1f}%)\n")
        f.write(f"CraftNet Time: {np.mean(stats['craftnet_times']):.4f} seconds ({np.mean(stats['craftnet_times'])/stats['avg_time_per_frame']*100:.1f}%)\n")
        f.write(f"RefineNet Time: {np.mean(stats['refinenet_times']):.4f} seconds ({np.mean(stats['refinenet_times'])/stats['avg_time_per_frame']*100:.1f}%)\n")
        f.write(f"Postprocess Time: {np.mean(stats['postprocess_times']):.4f} seconds ({np.mean(stats['postprocess_times'])/stats['avg_time_per_frame']*100:.1f}%)\n")
        f.write("\n")
        
        # Performance Statistics
        f.write("PERFORMANCE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Min Processing Time: {np.min(stats['frame_times']):.4f} seconds\n")
        f.write(f"Max Processing Time: {np.max(stats['frame_times']):.4f} seconds\n")
        f.write(f"Median Processing Time: {np.median(stats['frame_times']):.4f} seconds\n")
        f.write(f"Std Dev Processing Time: {np.std(stats['frame_times']):.4f} seconds\n")
        f.write(f"Min FPS: {np.min([1/t if t > 0 else 0 for t in stats['frame_times']]):.2f}\n")
        f.write(f"Max FPS: {np.max([1/t if t > 0 else 0 for t in stats['frame_times']]):.2f}\n")
        f.write("\n")
        
        # Text Detection Statistics
        f.write("TEXT DETECTION STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Min Regions per Frame: {np.min(stats['text_regions_per_frame'])}\n")
        f.write(f"Max Regions per Frame: {np.max(stats['text_regions_per_frame'])}\n")
        f.write(f"Median Regions per Frame: {np.median(stats['text_regions_per_frame']):.1f}\n")
        f.write(f"Std Dev Regions per Frame: {np.std(stats['text_regions_per_frame']):.2f}\n")
        if stats['avg_regions_per_frame'] > 30:
            f.write(f"\n⚠ WARNING: High detection rate ({stats['avg_regions_per_frame']:.1f} regions/frame)\n")
            f.write(f"  → Possible false positives. Consider increasing thresholds.\n")
        f.write("\n")
        
        # Memory and System Info
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Device: {stats['device_name']}\n")
        f.write(f"Video Resolution: {stats['resolution']}\n")
        f.write(f"Long Size Parameter: {stats['long_size']}\n")
        f.write(f"Text Threshold: {stats['text_threshold']}\n")
        f.write(f"Link Threshold: {stats['link_threshold']}\n")
        f.write("\n")
    
    return report_path

def save_statistics_json(stats, output_folder):
    """
    Save statistics as JSON for further analysis
    """
    json_path = os.path.join(output_folder, 'statistics.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_stats = {}
    for key, value in stats.items():
        if isinstance(value, (list, np.ndarray)):
            json_stats[key] = [float(v) if isinstance(v, (np.floating, float)) else int(v) for v in value]
        elif isinstance(value, (np.floating, float)):
            json_stats[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            json_stats[key] = int(value)
        else:
            json_stats[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(json_stats, f, indent=4)
    
    return json_path

def process_video(input_video_path, output_folder, analysis_folder, craft_net, refine_net, 
                  device="cpu", text_threshold=0.85, link_threshold=0.6, use_refine=True):
    """
    Process a single video and save all outputs with detailed analysis
    """
    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(analysis_folder, exist_ok=True)
    
    # Get video name without extension
    video_name = Path(input_video_path).stem
    
    # Define output paths
    output_path = os.path.join(output_folder, f"{video_name}_detected.mp4")
    link_heatmap_path = os.path.join(output_folder, f"{video_name}_link_heatmap.mp4")
    text_heatmap_path = os.path.join(output_folder, f"{video_name}_text_heatmap.mp4")
    text_region_path = os.path.join(output_folder, f"{video_name}_TextRegion.txt")
    
    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing: {video_name}")
    print(f"  Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")
    print(f"  Thresholds: text={text_threshold}, link={link_threshold}")
    print(f"  RefineNet: {'Enabled' if use_refine else 'Disabled (faster processing)'}")
    
    # Read first frame to get heatmap dimensions
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from {input_video_path}")
        cap.release()
        return None
    
    use_device = (device == "mps" or device == "cuda")
    
    temp = get_prediction(
        image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        craft_net=craft_net,
        refine_net=refine_net if use_refine else None,
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=0.4,
        cuda=use_device,
        long_size=1280
    )
    t1 = temp["heatmaps"]["text_score_heatmap"]
    t2 = temp["heatmaps"]["link_score_heatmap"]
    
    # Define codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    out1 = cv2.VideoWriter(link_heatmap_path, fourcc, fps, (len(t2[0]), len(t2)))
    out2 = cv2.VideoWriter(text_heatmap_path, fourcc, fps, (len(t1[0]), len(t1)))
    
    # Open text file for writing
    f = open(text_region_path, "w")
    f.write("=" * 100 + "\n")
    f.write(f"TEXT DETECTION RESULTS\n")
    f.write(f"Video: {video_name}\n")
    f.write(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}\n")
    f.write(f"Thresholds: text={text_threshold}, link={link_threshold}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 100 + "\n\n")
    
    # Initialize statistics tracking
    stats = {
        'frame_times': [],
        'resize_times': [],
        'preprocessing_times': [],
        'craftnet_times': [],
        'refinenet_times': [],
        'postprocess_times': [],
        'text_regions_per_frame': [],
        'total_frames': 0,
        'total_text_regions': 0,
        'frames_with_text': 0,
        'frames_without_text': 0,
        'video_fps': fps,
        'resolution': f"{frame_width}x{frame_height}",
        'device_name': f"{'MPS' if device == 'mps' else 'CUDA' if device == 'cuda' else 'CPU'}",
        'long_size': 1280,
        'text_threshold': text_threshold,
        'link_threshold': link_threshold
    }
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_index = 1
    start_time = time.time()
    
    # Process all frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        
        # Convert BGR to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=craft_net,
            refine_net=refine_net if use_refine else None,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=0.4,
            cuda=use_device,
            long_size=1280
        )
        
        # Export results (writes to video files)
        export_results(
            image=image,
            regions=prediction_result["boxes"],
            heatmaps=prediction_result["heatmaps"],
            out=out,
            out1=out1,
            out2=out2,
        )
        
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        
        # Collect statistics
        times = prediction_result["times"]
        stats['frame_times'].append(frame_processing_time)
        stats['resize_times'].append(times['resize_time'])
        stats['preprocessing_times'].append(times['preprocessing_time'])
        stats['craftnet_times'].append(times['craftnet_time'])
        stats['refinenet_times'].append(times['refinenet_time'])
        stats['postprocess_times'].append(times['postprocess_time'])
        
        regions = prediction_result["boxes"]
        num_regions = len(regions)
        stats['text_regions_per_frame'].append(num_regions)
        stats['total_text_regions'] += num_regions
        
        if num_regions > 0:
            stats['frames_with_text'] += 1
        else:
            stats['frames_without_text'] += 1
        
        # Write to text file
        f.write(f"Frame {frame_index} (Time: {frame_processing_time:.4f}s, FPS: {1/frame_processing_time:.2f}):\n")
        
        if num_regions > 0:
            f.write(f"  Detected {num_regions} text region(s):\n")
            for i, region in enumerate(regions):
                region = np.array(region).astype(np.int32).reshape((-1))
                strResult = "    Region " + str(i+1) + ": [" + ",".join([str(r) for r in region]) + "]\n"
                f.write(strResult)
        else:
            f.write("  No text regions detected\n")
        
        f.write("\n")
        frame_index += 1
        stats['total_frames'] += 1
        
        # Print progress
        if frame_index % 30 == 0 or frame_index == total_frames:
            elapsed = time.time() - start_time
            avg_fps = frame_index / elapsed
            eta = (total_frames - frame_index) / avg_fps if avg_fps > 0 else 0
            print(f"  Progress: {frame_index}/{total_frames} frames ({frame_index/total_frames*100:.1f}%) | "
                  f"Avg FPS: {avg_fps:.2f} | ETA: {eta:.1f}s")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    stats['total_time'] = total_time
    stats['avg_time_per_frame'] = np.mean(stats['frame_times'])
    stats['avg_fps'] = 1 / stats['avg_time_per_frame']
    stats['realtime_factor'] = (stats['total_frames'] / fps) / total_time if total_time > 0 else 0
    stats['avg_regions_per_frame'] = stats['total_text_regions'] / stats['total_frames']
    
    # Release resources
    cap.release()
    out.release()
    out1.release()
    out2.release()
    f.close()
    
    print(f"  Completed in {total_time:.2f}s (Avg: {stats['avg_fps']:.2f} FPS)")
    print(f"  Detected {stats['total_text_regions']} text regions across {stats['total_frames']} frames")
    
    # Generate analysis outputs
    print(f"  Generating analysis reports...")
    report_path = generate_analysis_report(stats, video_name, analysis_folder)
    json_path = save_statistics_json(stats, analysis_folder)
    plot_path = create_analysis_plots(stats, analysis_folder)
    
    print(f"  Analysis saved to: {analysis_folder}")
    
    return stats

def main():
    # Base directory for outputs
    base_output_dir = "/Users/padmaganesh/Desktop/cproject/secure camera_research/pythonversion/output"
    
    # List of input video files
    input_videos = [
        "VIDEO-2025-11-23-19-16-52.mp4",
        "km_20251123_720p_30f_20251123_163933.mp4",
        "TESTFILE3.mp4",
    ]
    
    # CONFIGURATION OPTIONS
    TEXT_THRESHOLD = 0.85  # Higher = fewer false positives (default: 0.7, recommended: 0.85)
    LINK_THRESHOLD = 0.6   # Higher = stricter word grouping (default: 0.4, recommended: 0.6)
    USE_REFINENET = True   # True = More accurate boundaries, False = ~33% faster
    
    print("=" * 80)
    print("TEXT DETECTION VIDEO PROCESSOR - M4 MAC OPTIMIZED")
    print("=" * 80)
    
    # Check and set device
    device, device_name = check_device()
    
    # Load models once
    print(f"\nLoading CRAFT and RefineNet models...")
    model_load_start = time.time()
    
    # Load models with appropriate device setting
    use_device = (device == "mps" or device == "cuda")
    craft_net = load_craftnet_model(cuda=use_device)
    refine_net = load_refinenet_model(cuda=use_device) if USE_REFINENET else None
    
    model_load_time = time.time() - model_load_start
    print(f"Models loaded in {model_load_time:.2f}s!")
    if not USE_REFINENET:
        print(f"RefineNet disabled for faster processing")
    
    # Track overall statistics
    overall_stats = []
    successful_videos = 0
    failed_videos = 0
    
    # Process each video
    for idx, input_video in enumerate(input_videos, start=1):
        print("\n" + "=" * 80)
        print(f"VIDEO {idx}/{len(input_videos)}")
        print("=" * 80)
        
        output_folder = os.path.join(base_output_dir, f"op{idx}")
        analysis_folder = os.path.join(base_output_dir, f"op{idx}", "analysis")
        
        if not os.path.exists(input_video):
            print(f"Warning: Video file not found: {input_video}")
            print(f"Skipping...")
            failed_videos += 1
            continue
        
        try:
            stats = process_video(
                input_video, 
                output_folder, 
                analysis_folder, 
                craft_net, 
                refine_net,
                device=device,
                text_threshold=TEXT_THRESHOLD,
                link_threshold=LINK_THRESHOLD,
                use_refine=USE_REFINENET
            )
            if stats:
                overall_stats.append({
                    'video': Path(input_video).stem,
                    'stats': stats
                })
                successful_videos += 1
        except Exception as e:
            print(f"Error processing {input_video}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_videos += 1
            continue
    
    # Generate overall summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Device Used: {device_name}")
    print(f"Total Videos: {len(input_videos)}")
    print(f"Successful: {successful_videos}")
    print(f"Failed: {failed_videos}")
    
    if overall_stats:
        print(f"\nOverall Statistics:")
        total_frames = sum([s['stats']['total_frames'] for s in overall_stats])
        total_time = sum([s['stats']['total_time'] for s in overall_stats])
        total_regions = sum([s['stats']['total_text_regions'] for s in overall_stats])
        avg_fps = total_frames / total_time if total_time > 0 else 0
        
        print(f"  Total Frames Processed: {total_frames}")
        print(f"  Total Processing Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"  Overall Average FPS: {avg_fps:.2f}")
        print(f"  Total Text Regions Detected: {total_regions}")
        
        # Save overall summary
        summary_path = os.path.join(base_output_dir, "overall_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("OVERALL PROCESSING SUMMARY\n")
            f.write(f"Device: {device_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Text Threshold: {TEXT_THRESHOLD}\n")
            f.write(f"  Link Threshold: {LINK_THRESHOLD}\n")
            f.write(f"  RefineNet: {'Enabled' if USE_REFINENET else 'Disabled'}\n\n")
            f.write(f"Total Videos Processed: {successful_videos}\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)\n")
            f.write(f"Average FPS: {avg_fps:.2f}\n")
            f.write(f"Total Text Regions: {total_regions}\n\n")
            
            for item in overall_stats:
                f.write(f"\n{item['video']}:\n")
                f.write(f"  Frames: {item['stats']['total_frames']}\n")
                f.write(f"  Time: {item['stats']['total_time']:.2f}s\n")
                f.write(f"  Avg FPS: {item['stats']['avg_fps']:.2f}\n")
                f.write(f"  Text Regions: {item['stats']['total_text_regions']}\n")
                f.write(f"  Avg Regions/Frame: {item['stats']['avg_regions_per_frame']:.2f}\n")
    
    print(f"\nAll outputs saved in: {base_output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()