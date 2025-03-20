import argparse
import os
import sys
import time
from pathlib import Path
import logging

from yolo import YOLOInferencer

def load_image_list(file_list_path):
    """Load image list from file"""
    with open(file_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    return image_paths

def process_shard(args, image_paths):
    """Process a shard of images"""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize YOLO inferencer
    inferencer = YOLOInferencer(
        model_path=args.model_path,
        device_id=0,  # Always use first GPU in the allocated node
        batch_size=args.batch_size,
        confidence_threshold=args.confidence,
        class_ids=[int(cid) for cid in args.class_ids.split(',')],
        scale_factor=args.scale_factor,
        split_processing=args.split_processing,
        array_id=args.array_id
    )
    
    # Process images
    results = inferencer.process_image_list(image_paths, args.output_dir)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLO Pedestrian Detection on SLURM')
    parser.add_argument('--file-list', type=str, required=True, help='Path to file containing list of images')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for detections')
    parser.add_argument('--model-path', type=str, default='yolov8x.pt', help='Path to YOLO model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--class-ids', type=str, default='0', help='Comma-separated list of class IDs to detect')
    parser.add_argument('--scale-factor', type=float, default=0.5, help='Image scale factor')
    parser.add_argument('--array-id', type=int, required=True, help='SLURM array ID')
    parser.add_argument('--num-tasks', type=int, required=True, help='Total number of SLURM tasks')
    parser.add_argument('--split-processing', action='store_true', 
                  help='Process each image in halves for higher resolution')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, f"task_{args.array_id}.log"))
        ]
    )
    logger = logging.getLogger("YOLO-SLURM")
    
    # Load all image paths
    image_paths = load_image_list(args.file_list)
    
    logger.info(f"Loaded {len(image_paths)} image paths")
    
    # Divide images into shards based on array task ID
    num_images = len(image_paths)
    shard_size = num_images // args.num_tasks + (1 if num_images % args.num_tasks != 0 else 0)
    
    start_idx = args.array_id * shard_size
    end_idx = min(start_idx + shard_size, num_images)
    
    shard_paths = image_paths[start_idx:end_idx]
    
    logger.info(f"Task {args.array_id} processing shard {start_idx}:{end_idx} ({len(shard_paths)} images)")
    
    # Process shard
    start_time = time.time()
    results = process_shard(args, shard_paths)
    end_time = time.time()
    
    # Log statistics
    processing_time = end_time - start_time
    images_per_second = len(shard_paths) / processing_time if processing_time > 0 else 0
    
    logger.info(f"Task {args.array_id} completed. Processed {len(shard_paths)} images in {processing_time:.2f}s")
    logger.info(f"Processing speed: {images_per_second:.2f} images/s")
    
    pedestrian_count = sum(1 for r in results.values() if r["is_pedestrian"])
    logger.info(f"Found pedestrians in {pedestrian_count} images ({pedestrian_count/len(shard_paths)*100:.2f}%)")

if __name__ == "__main__":
    main()