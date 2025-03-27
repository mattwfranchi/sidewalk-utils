#!/usr/bin/env python3
import os
import glob
import argparse
import logging
from pathlib import Path
import time
from tqdm import tqdm
import multiprocessing

from aggregate_results import YOLOResultsAggregator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("YOLO-Processor")

def discover_task_directories(base_dir):
    """
    Discover all detection task directories in the given base directory.
    
    Args:
        base_dir: Base directory to search in
    
    Returns:
        List of task IDs found
    """
    base_path = Path(base_dir)
    task_ids = set()
    
    # Look for detection directories pattern
    detection_dirs = list(base_path.glob("**/detections_task*"))
    for dir_path in detection_dirs:
        try:
            task_id = int(dir_path.name.replace("detections_task", ""))
            task_ids.add(task_id)
            logger.info(f"Found detection directory: {dir_path}, task ID: {task_id}")
        except (ValueError, IndexError):
            logger.warning(f"Could not extract task ID from directory: {dir_path}")
    
    # Also look for summary files
    summary_files = list(base_path.glob("**/summary_task*.json"))
    for file_path in summary_files:
        try:
            task_id = int(file_path.name.replace("summary_task", "").replace(".json", ""))
            task_ids.add(task_id)
            logger.info(f"Found summary file: {file_path}, task ID: {task_id}")
        except (ValueError, IndexError):
            logger.warning(f"Could not extract task ID from file: {file_path}")
    
    # Get pedestrian image lists
    pedestrian_files = list(base_path.glob("**/pedestrian_images_task*.txt"))
    for file_path in pedestrian_files:
        try:
            task_id = int(file_path.name.replace("pedestrian_images_task", "").replace(".txt", ""))
            task_ids.add(task_id)
            logger.info(f"Found pedestrian image list: {file_path}, task ID: {task_id}")
        except (ValueError, IndexError):
            logger.warning(f"Could not extract task ID from file: {file_path}")
    
    return sorted(list(task_ids))

def find_common_parent_dir(base_dir, task_ids):
    """
    Find the common parent directory that contains results for all task IDs.
    
    Args:
        base_dir: Base directory to search in
        task_ids: List of task IDs to find
    
    Returns:
        Path to the common parent directory
    """
    base_path = Path(base_dir)
    all_dirs = []
    
    # For each task ID, find directories containing its results
    for task_id in task_ids:
        # Find directories containing detection files for this task
        detection_dirs = list(base_path.glob(f"**/detections_task{task_id}"))
        for dir_path in detection_dirs:
            # Add the parent directory
            all_dirs.append(dir_path.parent)
    
    # Count occurrences of each directory
    dir_counts = {}
    for dir_path in all_dirs:
        dir_counts[dir_path] = dir_counts.get(dir_path, 0) + 1
    
    # Find directory containing most task results
    best_dir = None
    max_count = 0
    for dir_path, count in dir_counts.items():
        if count > max_count:
            max_count = count
            best_dir = dir_path
    
    logger.info(f"Found common parent directory: {best_dir} containing {max_count} task results")
    return best_dir

def process_all_detections(
    base_dir, 
    output_path=None, 
    include_bboxes=True, 
    num_workers=None,
    batch_size=100000,
    chunk_size=1000,
    parallel_mode="thread",  # Default to thread mode which is more reliable
    task_subset=None,
    max_tasks=None
):
    """
    Process all detection results in the given directory structure.
    
    Args:
        base_dir: Base directory containing detection results
        output_path: Path for the output parquet file
        include_bboxes: Whether to include bounding box details
        num_workers: Number of parallel workers (default: CPU count - 1)
        batch_size: Maximum batch size for processing
        chunk_size: Number of files to process in each parallel chunk
        parallel_mode: Parallelization mode - "process" or "thread"
        task_subset: Process a specific subset of tasks (list of task IDs)
        max_tasks: Maximum number of tasks to process (useful for testing)
    
    Returns:
        Path to the generated parquet file
    """
    start_time = time.time()
    logger.info(f"Starting processing of all detection results in {base_dir}")
    
    # Discover all task IDs
    task_ids = discover_task_directories(base_dir)
    logger.info(f"Found {len(task_ids)} task IDs")
    
    if not task_ids:
        logger.error(f"No task directories found in {base_dir}")
        return None
    
    # Apply task subset filter if specified
    if task_subset:
        task_ids = [tid for tid in task_ids if tid in task_subset]
        logger.info(f"Filtered to {len(task_ids)} tasks from subset")
    
    # Apply max_tasks limit if specified
    if max_tasks and len(task_ids) > max_tasks:
        task_ids = task_ids[:max_tasks]
        logger.info(f"Limited to first {max_tasks} tasks")
    
    # Find the common parent directory
    results_dir = find_common_parent_dir(base_dir, task_ids)
    
    # Set output path if not provided
    if output_path is None:
        output_path = os.path.join(results_dir, "aggregated_yolo_results.parquet")
    
    # Determine sensible worker count if not specified
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {num_workers} workers based on system CPU count")
    
    # Create the aggregator
    aggregator = YOLOResultsAggregator(
        results_dir=results_dir,
        output_path=output_path,
        include_bboxes=include_bboxes,
        max_batch_size=batch_size,
        task_ids=task_ids,
        num_workers=num_workers,
        chunk_size=chunk_size,
        parallel_mode=parallel_mode
    )
    
    # Process all results
    logger.info(f"Starting aggregation with {num_workers} workers in {parallel_mode} mode")
    parquet_path = aggregator.aggregate()
    
    # Add metadata
    aggregator.add_metadata()
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Processing complete! Time elapsed: {elapsed_time:.2f} seconds")
    logger.info(f"📊 Results saved to: {parquet_path}")
    
    return parquet_path

def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="Process all YOLO detection results from a directory structure into a single parquet file"
    )
    parser.add_argument("base_dir", help="Base directory containing YOLO detection results")
    parser.add_argument("--output", "-o", help="Output parquet file path")
    parser.add_argument("--no-bboxes", action="store_true", help="Exclude bounding box details")
    parser.add_argument("--batch-size", type=int, default=100000, help="Maximum batch size for processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of files to process in each parallel chunk")
    parser.add_argument("--parallel-mode", choices=["process", "thread"], default="thread",
                      help="Parallelization mode - process for CPU-bound, thread for I/O-bound")
    parser.add_argument("--task-ids", type=int, nargs="+", help="Specific task IDs to process")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks to process")
    
    args = parser.parse_args()
    
    # Default to the Nexar dataset path if not specified
    base_dir = args.base_dir
    if not base_dir:
        base_dir = "data/detections/nexar2020_tuesthurs_subset/yolov8x/"
        logger.info(f"No base directory provided, using default: {base_dir}")
    
    process_all_detections(
        base_dir=base_dir,
        output_path=args.output,
        include_bboxes=not args.no_bboxes,
        batch_size=args.batch_size,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        parallel_mode=args.parallel_mode,
        task_subset=args.task_ids,
        max_tasks=args.max_tasks
    )

if __name__ == "__main__":
    main()