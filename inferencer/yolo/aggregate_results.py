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
from functools import partial
import time
import queue
import pickle

from shapely_utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("YOLOAggregator")

class YOLOResultsAggregator:
    """
    Aggregates YOLO detection results from multiple sources into a parquet file.
    """
    def __init__(
        self,
        results_dir: str,
        output_path: str = None,
        include_bboxes: bool = True,
        max_batch_size: int = 100000,
        task_ids: List[int] = None,
        num_workers: int = None,
        chunk_size: int = 1000,
        parallel_mode: str = "process",  # "process" or "thread"
    ):
        """
        Initialize the results aggregator.
        
        Args:
            results_dir: Directory containing YOLO detection results
            output_path: Path to save the parquet file (default: results_dir/aggregated_results.parquet)
            include_bboxes: Whether to include bounding box details in the parquet file
            max_batch_size: Maximum number of rows to process at once (for memory management)
            task_ids: Specific task IDs to aggregate (default: all tasks in results_dir)
            num_workers: Number of parallel workers (default: CPU count)
            chunk_size: Number of files to process in each parallel chunk
            parallel_mode: Parallelization mode - "process" for CPU-bound, "thread" for I/O-bound
        """
        self.results_dir = Path(results_dir)
        self.output_path = output_path or str(self.results_dir / "aggregated_results.parquet")
        self.include_bboxes = include_bboxes
        self.max_batch_size = max_batch_size
        self.task_ids = task_ids
        
        # Parallellization settings
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.chunk_size = chunk_size
        self.parallel_mode = parallel_mode
        
        # Fix lock creation - only create a thread lock, or use Manager for process
        if parallel_mode == "process":
            # Create a manager for shared resources across processes
            self.manager = multiprocessing.Manager()
            self._write_lock = self.manager.Lock()
        else:  # "thread"
            self._write_lock = concurrent.futures.threading.Lock()
        
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
                # Handle multiple detection boxes
                bboxes = data.get("bboxes", [])
                confidences = data.get("confidences", [])
                class_ids = data.get("class_ids", [])
                
                # Convert bboxes to Shapely rectangles and store as WKB
                shapely_boxes = []
                
                if bboxes and len(bboxes) > 0:
                    for bbox in bboxes:
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
    
    def _write_parquet_file(self, df: pd.DataFrame, mode: str = "write"):
        """Write DataFrame to parquet file, with synchronization for parallel writing"""
        with self._write_lock:  # Ensure exclusive access when writing
            if mode == "write":
                df.to_parquet(self.output_path, index=False)
                logger.info(f"Wrote {len(df)} records to {self.output_path}")
            elif mode == "append":
                # PyArrow requires a '{i}' placeholder in the basename_template
                # for creating unique filenames when appending
                base_filename = os.path.basename(self.output_path)
                file_root, file_ext = os.path.splitext(base_filename)
                basename_template = f"{file_root}_part{{i}}{file_ext}"
                
                # Use temporary dataset path for writing parts
                temp_dir = os.path.join(os.path.dirname(self.output_path), f"{file_root}_parts")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Write the new part
                table = pa.Table.from_pandas(df)
                pq.write_to_dataset(
                    table, 
                    root_path=temp_dir,
                    partition_cols=[],
                    basename_template=basename_template
                )
                
                # Read and combine all parts
                try:
                    # Read the main file if it exists
                    if os.path.exists(self.output_path):
                        main_df = pd.read_parquet(self.output_path)
                        
                        # Read all part files
                        part_files = glob.glob(os.path.join(temp_dir, f"{file_root}_part*{file_ext}"))
                        for part_file in part_files:
                            part_df = pd.read_parquet(part_file)
                            main_df = pd.concat([main_df, part_df], ignore_index=True)
                        
                        # Write the combined dataframe back to the main file
                        main_df.to_parquet(self.output_path, index=False)
                    else:
                        # If main file doesn't exist, just rename the first part file
                        part_files = glob.glob(os.path.join(temp_dir, f"{file_root}_part*{file_ext}"))
                        if part_files:
                            import shutil
                            shutil.copy2(part_files[0], self.output_path)
                    
                    logger.info(f"Appended {len(df)} records to {self.output_path}")
                finally:
                    # Clean up temporary files
                    import shutil
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _process_detection_batch(self, detection_files: List[Tuple[str, str, int]]) -> List[Dict[str, Any]]:
        """Process a batch of detection files in parallel
        
        Args:
            detection_files: List of tuples (file_path, image_name, task_id)
            
        Returns:
            List of processed records
        """
        records = []
        
        for file_path, image_name, task_id in detection_files:
            try:
                record = self._process_detection_file(file_path, image_name, task_id)
                records.append(record)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                # Add an error record to maintain consistency
                records.append({
                    "image_name": image_name,
                    "task_id": task_id,
                    "is_pedestrian": False,
                    "num_pedestrians": 0,
                    "error": str(e)
                })
                
        return records
    
    def _process_non_pedestrian_batch(self, image_names: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
        """Process a batch of non-pedestrian images in parallel
        
        Args:
            image_names: List of tuples (image_name, task_id)
            
        Returns:
            List of processed records
        """
        records = []
        
        for image_name, task_id in image_names:
            try:
                record = self._create_non_pedestrian_record(image_name, task_id)
                records.append(record)
            except Exception as e:
                logger.error(f"Error processing non-pedestrian image {image_name}: {e}")
                records.append({
                    "image_name": image_name,
                    "task_id": task_id,
                    "is_pedestrian": False,
                    "num_pedestrians": 0,
                    "error": str(e)
                })
                
        return records
    
    def _chunk_list(self, input_list, chunk_size):
        """Split a list into chunks of specified size"""
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]
            
    def _get_executor(self):
        """Get the appropriate executor based on parallel_mode"""
        if self.parallel_mode == "process":
            # Check if we're running in a context that's safe for process pool
            try:
                # Try a minimal process test
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: "test")
                    result = future.result()
                # If we get here, process pool is safe
                return concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
            except (TypeError, AttributeError, pickle.PicklingError, ValueError) as e:
                # Fall back to thread mode if process pool fails
                logger.warning(f"Process pool failed: {e}. Falling back to thread mode.")
                return concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers * 2)
        else:  # "thread"
            return concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers)
    
    def aggregate(self) -> str:
        """
        Aggregate all detection results into a parquet file, using task-level parallelism.
        
        Returns:
            Path to the created parquet file
        """
        logger.info(f"Starting task-level parallel aggregation of YOLO detection results from {self.results_dir}")
        logger.info(f"Using {self.num_workers} workers with {self.parallel_mode} parallelism")
        
        start_time = time.time()
        task_ids = self._get_task_ids()
        
        if not task_ids:
            # If no specific task IDs found, try to process all available data
            logger.info("No specific task IDs found, processing all available data")
            
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
        
        # Create a new method to process an entire task
        def process_task(task_id):
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
            
            for file_path in tqdm(detection_files, desc=f"Task {task_id} detection files", position=task_id % 10):
                image_name = os.path.splitext(os.path.basename(file_path))[0] + ".jpg"
                record = self._process_detection_file(file_path, image_name, task_id)
                task_records.append(record)
            
            # Process non-pedestrian images
            logger.info(f"Processing {len(non_pedestrian_images)} non-pedestrian images for task {task_id}")
            
            for image_name in tqdm(non_pedestrian_images, desc=f"Task {task_id} non-pedestrian", position=task_id % 10):
                record = self._create_non_pedestrian_record(image_name, task_id)
                task_records.append(record)
            
            # Create a DataFrame for this task
            task_df = pd.DataFrame(task_records)
            
            # Save to a temporary parquet file for this task
            task_output = os.path.splitext(self.output_path)[0] + f"_task{task_id}.parquet"
            task_df.to_parquet(task_output, index=False)
            
            task_elapsed = time.time() - task_start_time
            logger.info(f"Completed task {task_id} with {len(task_records)} records in {task_elapsed:.2f} seconds")
            
            return task_output, len(task_records)
        
        # Execute tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_task, task_id): task_id for task_id in task_ids}
            
            # Collect results
            task_outputs = []
            total_records = 0
            
            for future in tqdm(concurrent.futures.as_completed(future_to_task), 
                              total=len(future_to_task), desc="Overall progress"):
                task_id = future_to_task[future]
                try:
                    task_output, record_count = future.result()
                    task_outputs.append(task_output)
                    total_records += record_count
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
        
        # Merge all task outputs
        logger.info(f"Merging {len(task_outputs)} task outputs into final parquet file")
        
        if task_outputs:
            # Read and combine all task parquet files
            all_dfs = []
            for task_output in task_outputs:
                try:
                    task_df = pd.read_parquet(task_output)
                    all_dfs.append(task_df)
                except Exception as e:
                    logger.error(f"Error reading {task_output}: {e}")
            
            # Concatenate all dataframes
            if all_dfs:
                final_df = pd.concat(all_dfs, ignore_index=True)
                final_df.to_parquet(self.output_path, index=False)
                
                # Clean up temporary files
                for task_output in task_outputs:
                    try:
                        os.remove(task_output)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {task_output}: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Aggregation complete. Total {total_records} records written to {self.output_path}")
        logger.info(f"Processing time: {elapsed_time:.2f} seconds")
        
        return self.output_path

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
    parser.add_argument("--batch-size", type=int, default=100000, help="Maximum batch size for processing")
    parser.add_argument("--task-ids", type=int, nargs="+", help="Specific task IDs to process")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of files to process in each parallel chunk")
    parser.add_argument("--parallel-mode", choices=["process", "thread"], default="process",
                        help="Parallelization mode - process for CPU-bound, thread for I/O-bound")
    
    args = parser.parse_args()
    
    aggregator = YOLOResultsAggregator(
        results_dir=args.results_dir,
        output_path=args.output,
        include_bboxes=not args.no_bboxes,
        max_batch_size=args.batch_size,
        task_ids=args.task_ids,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        parallel_mode=args.parallel_mode
    )
    
    aggregator.aggregate()
    aggregator.add_metadata()


if __name__ == "__main__":
    main()