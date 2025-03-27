#!/usr/bin/env python3
import os
import json
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Merge OpenPose detection results")
    parser.add_argument("--task-dir", required=True, help="Task output directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Merging results from {args.task_dir} to {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all batch directories
    batch_dirs = glob(os.path.join(args.task_dir, "batch_*"))
    print(f"Found {len(batch_dirs)} batch directories")
    
    # Merge pedestrian and non-pedestrian image lists
    pedestrian_images = set()
    non_pedestrian_images = set()
    
    for batch_dir in batch_dirs:
        ped_file = os.path.join(batch_dir, "pedestrian_images.txt")
        non_ped_file = os.path.join(batch_dir, "non_pedestrian_images.txt")
        
        if os.path.exists(ped_file):
            with open(ped_file, 'r') as f:
                pedestrian_images.update(line.strip() for line in f if line.strip())
        
        if os.path.exists(non_ped_file):
            with open(non_ped_file, 'r') as f:
                non_pedestrian_images.update(line.strip() for line in f if line.strip())
    
    # Write merged lists
    with open(os.path.join(args.output_dir, "pedestrian_images.txt"), 'w') as f:
        f.write('\n'.join(sorted(pedestrian_images)))
    
    with open(os.path.join(args.output_dir, "non_pedestrian_images.txt"), 'w') as f:
        f.write('\n'.join(sorted(non_pedestrian_images)))
    
    # Copy all keypoint files to the merged output
    os.makedirs(os.path.join(args.output_dir, "keypoints"), exist_ok=True)
    
    keypoint_count = 0
    for batch_dir in batch_dirs:
        keypoints_dir = os.path.join(batch_dir, "keypoints")
        if os.path.exists(keypoints_dir):
            keypoint_files = glob(os.path.join(keypoints_dir, "*.json"))
            for kf in keypoint_files:
                # Copy keypoint file
                with open(kf, 'r') as f:
                    data = json.load(f)
                
                out_file = os.path.join(args.output_dir, "keypoints", os.path.basename(kf))
                with open(out_file, 'w') as f:
                    json.dump(data, f)
                keypoint_count += 1
    
    # Create merged summary
    summary = {
        "total_images": len(pedestrian_images) + len(non_pedestrian_images),
        "images_with_pedestrians": len(pedestrian_images),
        "images_without_pedestrians": len(non_pedestrian_images),
        "batches_merged": len(batch_dirs),
        "keypoints_saved": keypoint_count,
        "merged_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Merged {len(batch_dirs)} batches: {len(pedestrian_images)} images with pedestrians, {len(non_pedestrian_images)} without")
    print(f"Saved {keypoint_count} keypoint files")

if __name__ == "__main__":
    import time
    main()