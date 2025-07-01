import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from shapely.geometry import box
import shapely.wkb as swkb
import base64
import concurrent.futures
import multiprocessing
import time
import pickle
import threading  # Add proper threading module import

from shapely_utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("YOLOAggregator")

class YOLOResultsAggregator:
    """
    Aggregates YOLO detection results from multiple sources into a parquet file.
    Focus on task-level parallelization for efficiency.
    """
    def __init__(
        self,
        results_dir: str,
        output_path: str = None,
        include_bboxes: bool = True,
        task_ids: List[int] = None,
        num_workers: int = None,
        parallel_mode: str = "thread",  # "process" or "thread"
    ):
        """
        Initialize the results aggregator.
        
        Args:
            results_dir: Directory containing YOLO detection results
            output_path: Path to save the parquet file (default: results_dir/aggregated_results.parquet)
            include_bboxes: Whether to include bounding box details in the parquet file
            task_ids: Specific task IDs to aggregate (default: all tasks in results_dir)
            num_workers: Number of parallel workers (default: CPU count)
            parallel_mode: Parallelization mode - "process" for CPU-bound, "thread" for I/O-bound
        """
        self.results_dir = Path(results_dir)
        self.output_path = output_path or str(self.results_dir / "aggregated_results.parquet")
        self.include_bboxes = include_bboxes
        self.task_ids = task_ids
        
        # Parallelization settings
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.parallel_mode = parallel_mode
        
        # Fix lock creation - only create a thread lock, or use Manager for process
        if parallel_mode == "process":
            # Create a manager for shared resources across processes
            self.manager = multiprocessing.Manager()
            self._write_lock = self.manager.Lock()
        else:  # "thread"
            self._write_lock = threading.Lock()  # Fixed: use proper threading.Lock
        
        logger.info(f"Initializing with {self.num_workers} workers in {parallel_mode} mode")
        
    def _get_task_ids(self) -> List[int]:
        """Discover available task IDs if not explicitly provided"""
        if self.task_ids is not None:
            return self.task_ids
            
        # Try to infer task IDs from summary files
        summary_pattern = str(self.results_dir / "summary_task*.json")
        summary_files = glob.glob(summary_pattern)
        
        task_ids = []
        for summary_file in summary_files:
            try:
                # Extract task ID from filename (summary_task{id}.json)
                filename = os.path.basename(summary_file)
                task_id = int(filename.replace("summary_task", "").replace(".json", ""))
                task_ids.append(task_id)
            except (ValueError, IndexError):
                logger.warning(f"Could not extract task ID from {summary_file}")
                
        if not task_ids:
            logger.warning("No task IDs found, will try to process all available data")
            
        return task_ids
    
    def _get_detection_files(self, task_id: int) -> List[str]:
        """Get list of detection JSON files for a specific task"""
        detection_dir = self.results_dir / f"detections_task{task_id}"
        if not detection_dir.exists():
            logger.warning(f"Detection directory not found: {detection_dir}")
            return []
            
        return glob.glob(str(detection_dir / "*.json"))
    
    def _get_image_lists(self, task_id: int) -> Dict[str, List[str]]:
        """Get lists of pedestrian and non-pedestrian images for a task"""
        pedestrian_file = self.results_dir / f"pedestrian_images_task{task_id}.txt"
        non_pedestrian_file = self.results_dir / f"non_pedestrian_images_task{task_id}.txt"
        
        result = {"pedestrian": [], "non_pedestrian": []}
        
        if pedestrian_file.exists():
            with open(pedestrian_file, 'r') as f:
                result["pedestrian"] = [line.strip() for line in f if line.strip()]
                
        if non_pedestrian_file.exists():
            with open(non_pedestrian_file, 'r') as f:
                result["non_pedestrian"] = [line.strip() for line in f if line.strip()]
                
        return result
    
    def _process_detection_file(self, file_path: str, image_name: str, task_id: int) -> Dict[str, Any]:
        """Process a single detection JSON file with Shapely geometries"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            base_record = {
                "image_name": image_name,
                "task_id": task_id,
                "is_pedestrian": data.get("is_pedestrian", False),
                "num_pedestrians": data.get("num_pedestrians", 0),
            }
            
            # Extract bounding box information if needed
            if self.include_bboxes:
                # Handle bboxes exactly as they are stored by yolo.py
                bboxes = data.get("bboxes", [])
                confidences = data.get("confidences", [])
                class_ids = data.get("class_ids", [])
                
                # Convert bboxes to Shapely rectangles and store as WKB
                shapely_boxes = []
                
                if bboxes and len(bboxes) > 0:
                    # Only process detections that match our class IDs (person class)
                    # This matches the filtering done in yolo.py
                    person_indices = []
                    if class_ids and confidences:
                        for i, (class_id, conf) in enumerate(zip(class_ids, confidences)):
                            # In yolo.py, class ID 0 is person, and we filter by confidence threshold
                            if class_id == 0:  # Person class
                                person_indices.append(i)
                    
                    # Use all boxes if we couldn't filter
                    if not person_indices:
                        person_indices = range(len(bboxes))
                    
                    for i in person_indices:
                        if i < len(bboxes):
                            bbox = bboxes[i]
                            # Create a Shapely box (minx, miny, maxx, maxy)
                            # YOLO format is [x1, y1, x2, y2]
                            if len(bbox) == 4:
                                rect = box(bbox[0], bbox[1], bbox[2], bbox[3])
                                # Convert to WKB and encode as base64 string for storage
                                wkb_data = swkb.dumps(rect)
                                shapely_boxes.append(base64.b64encode(wkb_data).decode('ascii'))
                
                base_record["shapely_boxes"] = shapely_boxes
                base_record["confidences"] = confidences if confidences else []
                base_record["class_ids"] = class_ids if class_ids else []
                
            return base_record
            
        except Exception as e:
            logger.error(f"Error processing detection file {file_path}: {e}")
            return {
                "image_name": image_name,
                "task_id": task_id,
                "is_pedestrian": False,
                "num_pedestrians": 0,
                "error": str(e)
            }
    
    def _create_non_pedestrian_record(self, image_name: str, task_id: int) -> Dict[str, Any]:
        """Create a record for a non-pedestrian image without a JSON file"""
        base_record = {
            "image_name": image_name,
            "task_id": task_id,
            "is_pedestrian": False,
            "num_pedestrians": 0,
        }
        
        if self.include_bboxes:
            base_record["shapely_boxes"] = []
            base_record["confidences"] = []
            base_record["class_ids"] = []
            
        return base_record
    
    def process_task(self, task_id: int) -> Tuple[str, int]:
        """Process a single task and return the path to temporary parquet file and record count"""
        task_start_time = time.time()
        logger.info(f"Processing task ID {task_id}")
        task_records = []
        
        # Get lists of pedestrian and non-pedestrian images
        image_lists = self._get_image_lists(task_id)
        pedestrian_images = set(image_lists["pedestrian"])
        non_pedestrian_images = set(image_lists["non_pedestrian"])
        
        # Get list of detection files
        detection_files = self._get_detection_files(task_id)
        
        # Process detection files
        logger.info(f"Processing {len(detection_files)} detection files for task {task_id}")
        
        # Process in smaller chunks to save memory
        chunk_size = 1000  # Process 1000 files at a time
        for i in range(0, len(detection_files), chunk_size):
            chunk_files = detection_files[i:i+chunk_size]
            for file_path in tqdm(chunk_files, desc=f"Task {task_id} detection files {i}-{min(i+chunk_size, len(detection_files))}", 
                                 position=task_id % 10):
                # Get image name without assuming .jpg extension
                base_name = os.path.basename(file_path)
                image_name = os.path.splitext(base_name)[0]
                record = self._process_detection_file(file_path, image_name, task_id)
                task_records.append(record)
        
        # Process non-pedestrian images
        logger.info(f"Processing {len(non_pedestrian_images)} non-pedestrian images for task {task_id}")
        
        # Process in chunks
        non_ped_list = list(non_pedestrian_images)
        for i in range(0, len(non_ped_list), chunk_size):
            chunk_images = non_ped_list[i:i+chunk_size]
            for image_name in tqdm(chunk_images, 
                                  desc=f"Task {task_id} non-pedestrian {i}-{min(i+chunk_size, len(non_ped_list))}", 
                                  position=task_id % 10):
                record = self._create_non_pedestrian_record(image_name, task_id)
                task_records.append(record)
        
        # Create a DataFrame for this task
        task_df = pd.DataFrame(task_records)
        record_count = len(task_df)
        
        # Save to a temporary parquet file for this task - ensure directory exists
        task_dir = os.path.dirname(self.output_path)
        os.makedirs(task_dir, exist_ok=True)
        task_output = os.path.join(task_dir, f"task{task_id}_temp.parquet")
        
        task_df.to_parquet(task_output, index=False)
        
        # Clear memory
        del task_records
        del task_df
        import gc
        gc.collect()
        
        task_elapsed = time.time() - task_start_time
        logger.info(f"Completed task {task_id} with {record_count} records in {task_elapsed:.2f} seconds")
        
        return task_output, record_count
    
    def aggregate(self) -> str:
        """
        Aggregate all detection results into a parquet file, using task-level parallelism.
        Each task is processed by a single worker.
        
        Returns:
            Path to the created parquet file
        """
        logger.info(f"Starting task-level parallel aggregation of YOLO detection results from {self.results_dir}")
        logger.info(f"Using {self.num_workers} workers with {self.parallel_mode} parallelism")
        
        start_time = time.time()
        task_ids = self._get_task_ids()
        
        if not task_ids:
            # If no specific task IDs found, try to process all available data
            # Look for any detection directories
            all_dirs = [d for d in os.listdir(self.results_dir) if os.path.isdir(os.path.join(self.results_dir, d))]
            detection_dirs = [d for d in all_dirs if d.startswith("detections_task")]
            
            task_ids = []
            for dir_name in detection_dirs:
                try:
                    task_id = int(dir_name.replace("detections_task", ""))
                    task_ids.append(task_id)
                except (ValueError, IndexError):
                    continue
        
        logger.info(f"Processing results for {len(task_ids)} task IDs")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Process in batches to limit memory usage
        batch_size = 10  # Process 10 tasks at a time
        total_records = 0
        all_task_outputs = []
        
        for i in range(0, len(task_ids), batch_size):
            batch_task_ids = task_ids[i:i+batch_size]
            logger.info(f"Processing batch of {len(batch_task_ids)} tasks ({i+1}-{min(i+batch_size, len(task_ids))}/{len(task_ids)})")
            
            # Execute batch of tasks in parallel - one task per worker
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.num_workers, len(batch_task_ids))) as executor:
                future_to_task = {executor.submit(self.process_task, task_id): task_id for task_id in batch_task_ids}
                
                # Collect results
                task_outputs = []
                batch_records = 0
                
                for future in tqdm(concurrent.futures.as_completed(future_to_task), 
                                  total=len(future_to_task), desc=f"Batch progress ({i+1}-{min(i+batch_size, len(task_ids))})"):
                    task_id = future_to_task[future]
                    try:
                        task_output, record_count = future.result()
                        task_outputs.append((task_output, record_count))
                        batch_records += record_count
                    except Exception as e:
                        logger.error(f"Task {task_id} failed: {e}")
            
            logger.info(f"Batch completed with {batch_records} records")
            total_records += batch_records
            all_task_outputs.extend(task_outputs)
            
            # Merge batch results incrementally if needed (e.g., after every 5 batches)
            if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= len(task_ids):
                self._merge_intermediate_outputs(all_task_outputs)
                # Clear the list of processed outputs after merging
                all_task_outputs = []
        
        elapsed_time = time.time() - start_time
        logger.info(f"Aggregation complete. Total {total_records} records written to {self.output_path}")
        logger.info(f"Processing time: {elapsed_time:.2f} seconds")
        
        return self.output_path
    
    def _merge_intermediate_outputs(self, task_outputs: List[Tuple[str, int]]) -> None:
        """
        Efficiently merge intermediate parquet files without loading everything into memory
        
        Args:
            task_outputs: List of tuples (file_path, record_count) to merge
        """
        if not task_outputs:
            return
            
        logger.info(f"Merging {len(task_outputs)} intermediate outputs")
        
        # Check if main output exists
        output_exists = os.path.exists(self.output_path)
        output_dir = os.path.dirname(self.output_path)
        
        # Create a temporary merge file
        import uuid
        temp_merge_path = os.path.join(output_dir, f"merge_{uuid.uuid4().hex}.parquet")
        
        # Use PyArrow for efficient merging
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        if output_exists:
            # If output already exists, we need to append to it
            try:
                # Read schema from existing file
                existing_schema = pq.read_schema(self.output_path)
                
                # Create a writer with the same schema
                writer = pq.ParquetWriter(temp_merge_path, existing_schema)
                
                # First, copy data from the existing file
                for batch in pq.ParquetFile(self.output_path).iter_batches():
                    writer.write_batch(batch)
                
                # Then add data from each task file
                for task_path, _ in task_outputs:
                    try:
                        for batch in pq.ParquetFile(task_path).iter_batches():
                            writer.write_batch(batch)
                    except Exception as e:
                        logger.error(f"Error reading {task_path}: {e}")
                
                writer.close()
                
                # Replace the original file with the merged file
                import shutil
                shutil.move(temp_merge_path, self.output_path)
                
            except Exception as e:
                logger.error(f"Error during incremental merge: {e}")
                # Fallback to pandas append if PyArrow approach fails
                try:
                    all_dfs = []
                    
                    # Read existing file
                    existing_df = pd.read_parquet(self.output_path)
                    all_dfs.append(existing_df)
                    
                    # Read each task file
                    for task_path, _ in task_outputs:
                        try:
                            df = pd.read_parquet(task_path)
                            all_dfs.append(df)
                        except Exception as e:
                            logger.error(f"Error reading {task_path}: {e}")
                    
                    # Concatenate and write
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    combined_df.to_parquet(self.output_path, index=False)
                    
                except Exception as e2:
                    logger.error(f"Pandas fallback also failed: {e2}")
        else:
            # First merge - just concatenate the files
            try:
                # Try using pyarrow for efficient merge
                tables = []
                for task_path, _ in task_outputs:
                    try:
                        tables.append(pq.read_table(task_path))
                    except Exception as e:
                        logger.error(f"Error reading {task_path}: {e}")
                
                if tables:
                    combined_table = pa.concat_tables(tables)
                    pq.write_table(combined_table, self.output_path)
            except Exception as e:
                logger.error(f"Error during first merge with PyArrow: {e}")
                # Fallback to pandas
                try:
                    all_dfs = []
                    for task_path, _ in task_outputs:
                        try:
                            df = pd.read_parquet(task_path)
                            all_dfs.append(df)
                        except Exception as e:
                            logger.error(f"Error reading {task_path}: {e}")
                    
                    if all_dfs:
                        combined_df = pd.concat(all_dfs, ignore_index=True)
                        combined_df.to_parquet(self.output_path, index=False)
                except Exception as e2:
                    logger.error(f"Pandas fallback also failed: {e2}")
        
        # Clean up temporary files
        for task_path, _ in task_outputs:
            try:
                os.remove(task_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {task_path}: {e}")
                
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info(f"Merge complete: Output saved to {self.output_path}")

    def add_metadata(self):
        """Add metadata from summary files to the parquet file"""
        task_ids = self._get_task_ids()
        metadata = {}
        
        for task_id in task_ids:
            summary_file = self.results_dir / f"summary_task{task_id}.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        task_metadata = json.load(f)
                    metadata[f"task_{task_id}"] = task_metadata
                except Exception as e:
                    logger.error(f"Error reading summary file {summary_file}: {e}")
        
        # Add aggregated metadata
        metadata["aggregation_info"] = {
            "task_ids": task_ids,
            "include_bboxes": self.include_bboxes,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Write metadata to a JSON file alongside the parquet file
        metadata_path = os.path.splitext(self.output_path)[0] + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Metadata written to {metadata_path}")


def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate YOLO detection results into a parquet file")
    parser.add_argument("results_dir", help="Directory containing YOLO detection results")
    parser.add_argument("--output", "-o", help="Output parquet file path")
    parser.add_argument("--no-bboxes", action="store_true", help="Exclude bounding box details")
    parser.add_argument("--task-ids", type=int, nargs="+", help="Specific task IDs to process")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--parallel-mode", choices=["process", "thread"], default="thread",
                        help="Parallelization mode - process for CPU-bound, thread for I/O-bound")
    
    args = parser.parse_args()
    
    aggregator = YOLOResultsAggregator(
        results_dir=args.results_dir,
        output_path=args.output,
        include_bboxes=not args.no_bboxes,
        task_ids=args.task_ids,
        num_workers=args.workers,
        parallel_mode=args.parallel_mode
    )
    
    aggregator.aggregate()
    aggregator.add_metadata()


if __name__ == "__main__":
    main()