#!/usr/bin/env python3
import os
import sys
import time
import argparse
from pathlib import Path
from openpose import OpenPoseInferencer
import numpy as np
import json

def print_with_flush(message):
    """Print message and flush stdout to ensure it appears in logs immediately."""
    print(message, flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenPose inferencer with Slurm")
    parser.add_argument("--input-dir", required=False, help="Directory containing images")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--file-list", required=True, help="Text file containing paths to images (one per line, absolute or relative to input-dir)")
    parser.add_argument("--model-path", default="../sub/pytorch-openpose/model/body_pose_model.pth", 
                        help="Path to OpenPose model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--min-keypoints", type=int, default=4, help="Minimum keypoints")
    parser.add_argument("--scale-factor", type=float, default=0.5, help="Image scale factor")
    
    # Slurm array job parameters
    parser.add_argument("--array-id", type=int, help="Slurm array task ID")
    parser.add_argument("--num-tasks", type=int, help="Total number of Slurm array tasks")
    
    return parser.parse_args()

def get_image_paths(input_dir, file_list=None):
    """
    Get list of image paths to process
    
    Args:
        input_dir (str): Directory containing images
        file_list (str, optional): Path to a text file containing paths to images
            - Each line should contain one image path
            - Paths can be absolute or relative to input_dir
    
    Returns:
        list: List of absolute paths to images
    """
    start_time = time.time()
    print_with_flush(f"Starting to get image paths at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if file_list:
        if not os.path.isfile(file_list):
            raise FileNotFoundError(f"File list not found: {file_list}")
        
        print_with_flush(f"Reading image paths from file: {file_list}")
        print_with_flush(f"File size: {os.path.getsize(file_list) / (1024*1024):.2f} MB")
        
        image_paths = []
        line_count = 0
        last_print_time = time.time()
        
        # Read file in chunks to improve performance
        with open(file_list, 'r') as f:
            for line in f:
                line_count += 1
                if line_count % 1000000 == 0 or time.time() - last_print_time > 30:
                    print_with_flush(f"Processed {line_count} lines")
                    last_print_time = time.time()
                    
                path = line.strip()
                if not path:  # Skip empty lines
                    continue
                
                # If the path is not absolute, make it relative to input_dir
                if input_dir and not os.path.isabs(path):
                    path = os.path.join(input_dir, path)
                
                # Skip file existence check since the list contains verified paths
                image_paths.append(path)
        
        elapsed = time.time() - start_time
        print_with_flush(f"Read {len(image_paths)} image paths from file list")
        print_with_flush(f"Time to read file list: {elapsed:.1f} seconds")
        return image_paths
    
    # Otherwise scan the directory
    print(f"Scanning directory for images: {input_dir}")
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    return image_paths

def copy_to_scratch_batch(batch_paths, scratch_dir):
    """Copy a batch of files to scratch space efficiently"""
    import shutil
    results = []
    
    for src_path in batch_paths:
        try:
            # Create deterministic filename to avoid collisions
            filename = os.path.basename(src_path)
            file_hash = str(abs(hash(src_path)))[-8:]
            unique_name = f"{file_hash}_{filename}"
            dst_path = os.path.join(scratch_dir, unique_name)
            
            # Copy with larger buffer size for better throughput
            with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                shutil.copyfileobj(src, dst, 1024*1024)  # 1MB buffer
                
            results.append((src_path, dst_path))
        except Exception as e:
            print_with_flush(f"Error copying {src_path}: {e}")
            results.append((src_path, src_path))
            
    return results

def main():
    print_with_flush(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_with_flush(f"Running on host: {os.uname().nodename}")
    print_with_flush(f"Process ID: {os.getpid()}")
    
    args = parse_args()
    print_with_flush(f"Arguments: {args}")
    
    # Setup device ID based on Slurm environment
    device_id = 0
    if "SLURM_LOCALID" in os.environ:
        device_id = int(os.environ["SLURM_LOCALID"])
        print_with_flush(f"Using SLURM_LOCALID for device_id: {device_id}")
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        print_with_flush(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        device_id = 0
    
    print_with_flush("About to get image paths...")
    
    # Get all image paths
    all_image_paths = get_image_paths(args.input_dir, args.file_list)
    print_with_flush(f"Found {len(all_image_paths)} images total")
    
    if len(all_image_paths) == 0:
        print("No images to process. Exiting.")
        return
    
    # Create a scratch directory for faster file I/O
    scratch_dir = os.path.join('/scratch', os.environ.get('USER', 'default'), f"openpose_{int(time.time())}")
    os.makedirs(scratch_dir, exist_ok=True)
    print_with_flush(f"Created scratch directory: {scratch_dir}")
    
    # For Slurm array jobs, divide the work
    if args.array_id is not None and args.num_tasks is not None:
        # Divide the dataset among array tasks
        chunk_size = int(np.ceil(len(all_image_paths) / args.num_tasks))
        start_idx = args.array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_image_paths))
        image_paths = all_image_paths[start_idx:end_idx]
        
        # Create task-specific output directory
        task_output_dir = os.path.join(args.output_dir, f"task_{args.array_id}")
        os.makedirs(task_output_dir, exist_ok=True)
        
        print_with_flush(f"Array task {args.array_id}: Processing {len(image_paths)} images ({start_idx} to {end_idx-1})")
    else:
        # Process all images in a single job
        image_paths = all_image_paths
        task_output_dir = args.output_dir
        os.makedirs(task_output_dir, exist_ok=True)
    
    # Process images in batches
    batch_size = args.batch_size  # Don't multiply by 4, use the actual batch size
    print_with_flush(f"Processing images in batches of {batch_size}")
    
    # Initialize inferencer once outside the loop
    inferencer = OpenPoseInferencer(
        model_path=args.model_path,
        device_id=device_id,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence,
        min_keypoints=args.min_keypoints,
        scale_factor=args.scale_factor
    )
    
    # Initialize a master results dictionary to accumulate all batch results
    all_results = {}
    pedestrians_found = 0
    total_pedestrians = 0
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_scratch_paths = []
        
        # Print progress
        print_with_flush(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        # Copy batch to scratch
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as executor:
            def copy_to_scratch(src_path):
                try:
                    # Create a unique filename in scratch
                    filename = os.path.basename(src_path)
                    # Use a simpler hash to avoid issues
                    file_hash = str(hash(src_path))[-8:]
                    unique_name = f"{file_hash}_{filename}"
                    dst_path = os.path.join(scratch_dir, unique_name)
                    
                    # Copy the file
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    return (src_path, dst_path)
                except Exception as e:
                    print_with_flush(f"Error copying {src_path}: {e}")
                    return (src_path, src_path)  # Fall back to original path
            
            # Copy images in parallel
            copy_results = list(executor.map(copy_to_scratch, batch_paths))
            
            # Build a map of original paths to scratch paths
            scratch_map = dict(copy_results)
            
            # Convert batch_paths to their corresponding scratch paths
            batch_scratch_paths = [scratch_map.get(path, path) for path in batch_paths]
        
        # Verify copied files
        scratch_file_count = sum(1 for p in batch_scratch_paths if p.startswith(scratch_dir))
        print_with_flush(f"Successfully copied {scratch_file_count}/{len(batch_paths)} files to scratch")
        
        # Create a batch-specific output directory
        batch_output_dir = os.path.join(task_output_dir, f"batch_{i//batch_size}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # Process current batch of images from scratch
        batch_results = inferencer.process_batch(batch_scratch_paths)
        
        # Save this batch's results separately (this avoids any possibility of overwriting)
        print_with_flush(f"Saving batch {i//batch_size} results to {batch_output_dir}")
        inferencer.save_detection_results(batch_results, batch_output_dir)
        
        # Also update our aggregated results
        all_results.update(batch_results)
        
        # Update statistics
        for result in batch_results.values():
            if result['is_pedestrian']:
                pedestrians_found += 1
            total_pedestrians += result['num_pedestrians']
        
        # Save intermediate checkpoint after each batch
        if (i + batch_size) % 1000 < batch_size or i + batch_size >= len(image_paths):
            checkpoint_dir = os.path.join(task_output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{i+batch_size}.json")
            print_with_flush(f"Saving checkpoint at {i+batch_size}/{len(image_paths)} images")
            
            # Save only the paths and pedestrian status for checkpointing
            checkpoint_data = {
                "pedestrian_images": [p for p, r in all_results.items() if r["is_pedestrian"]],
                "non_pedestrian_images": [p for p, r in all_results.items() if not r["is_pedestrian"]],
                "processed_count": len(all_results),
                "pedestrians_found": pedestrians_found,
                "total_pedestrians": total_pedestrians,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
        
        # Clean up scratch files for this batch to free space
        print_with_flush("Cleaning up scratch files from this batch")
        for path in batch_scratch_paths:
            if path.startswith(scratch_dir):
                try:
                    os.remove(path)
                except Exception as e:
                    print_with_flush(f"Warning: Could not remove scratch file {path}: {e}")
    
    # Save the final aggregated results
    print_with_flush(f"Saving final results for {len(all_results)} images")
    inferencer.save_detection_results(all_results, task_output_dir)
    
    # Save the combined summary in the parent directory
    summary_path = os.path.join(task_output_dir, "summary.json")
    summary = {
        "total_images": len(all_results),
        "images_with_pedestrians": len([r for r in all_results.values() if r["is_pedestrian"]]),
        "images_without_pedestrians": len(all_results) - len([r for r in all_results.values() if r["is_pedestrian"]]),
        "batches_processed": (len(image_paths) + batch_size - 1) // batch_size,
        "detection_parameters": {
            "confidence_threshold": args.confidence,
            "min_keypoints": args.min_keypoints,
            "scale_factor": args.scale_factor
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final cleanup of scratch directory
    try:
        import shutil
        shutil.rmtree(scratch_dir)
        print_with_flush(f"Removed scratch directory: {scratch_dir}")
    except Exception as e:
        print_with_flush(f"Error removing scratch directory: {e}")

if __name__ == "__main__":
    main()