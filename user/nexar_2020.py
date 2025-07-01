# sidewalk_utils 
# @mattwfranchi 

"""
Data input/output module for the sidewalk_utils project.
Provides dataset class and utilities for loading and processing Nexar 2020 dataset.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import os
import fire
import pandas as pd 
import sys
import traceback

from utils.logger import get_logger
from utils.timer import time_it
from user.base_nexar_dataset import BaseNexarDataset

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add pandarallel for parallel DataFrame operations
try:
    from pandarallel import pandarallel
    PANDARALLEL_AVAILABLE = True
except ImportError:
    PANDARALLEL_AVAILABLE = False

# ===== Constants =====
NEXAR_2020 = "/share/pierson/nexar_data/raw_data"
EXPORT_DIR = "/share/ju/sidewalk_utils/data/raw_export"

try:
    import pyarrow
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# Initialize pandarallel at module level if available
if PANDARALLEL_AVAILABLE:
    # Disable progress bars completely to avoid flooding logs
    pandarallel.initialize(progress_bar=False, verbose=0)


class Nexar2020Dataset(BaseNexarDataset):
    """Dataset class for the Nexar 2020 dataset."""

    def load_img_dir(self, img_dir):
        """Load images from a directory."""
        return list(Path(img_dir).glob("*.jpg"))

    @time_it(level="info", message="Loading images from Nexar 2020 dataset")
    def load_images(self, export=False):
        """Load images from the October-November 2020 Nexar dataset."""
        static_img_dirs_path = Path(NEXAR_2020) / "imgs" / "oct_15-nov-15"
        static_img_dirs = list(static_img_dirs_path.glob("*"))
        chunked_dirs = self._chunk_list(static_img_dirs, self.ncpus)
        futures = []
        with ProcessPoolExecutor(max_workers=self.ncpus) as executor:
            for chunk in chunked_dirs:
                futures.append(executor.submit(self._process_dir_chunk, chunk))
            for future in futures:
                self.imgs.extend(future.result())
        self.logger.success(f"Loaded {len(self.imgs)} images from Nexar 2020 dataset")
        if not self.md.empty:
            self.align_images_with_metadata()
        if export:
            self._export_image_paths()
        return self.imgs

    @time_it(level="info", message="Loading metadata from Nexar 2020 dataset")
    def load_metadata(self, export=False, format="parquet"):
        """Load metadata from the dataset."""
        metadata_path = Path(NEXAR_2020) / "metadata" / "oct_15-nov-15"
        self._load_metadata_parallel(metadata_path, export, format)


# Now let's modify the load_nexar_dataset function to use these new functions
def load_nexar_dataset(imgs=True, md=True, ncpus=8, 
                      start_date=None, end_date=None, dates=None, export=True, 
                      format="parquet", export_geoparquet=True, no_stats=False, 
                      validate_align=True, split_chunks=None, 
                      free_memory_after_export=True, memory_tracking=True):
    """Load a Nexar 2020 dataset, optionally filter by date, and show statistics.
    
    Args:
        imgs: Whether to load images
        md: Whether to load metadata
        ncpus: Number of CPUs to use for parallel processing
        start_date: Start date for range filtering (YYYY-MM-DD)
        end_date: End date for range filtering (YYYY-MM-DD)
        dates: List of specific dates to filter for (overrides start_date/end_date if provided)
        export: Whether to export the dataset
        format: Export format for metadata ('parquet' or 'csv')
        export_geoparquet: Whether to export geospatial metadata as GeoParquet
        no_stats: If True, don't show dataset statistics
        validate_align: Whether to validate image-metadata alignment
        split_chunks: Number of chunks to split the dataset into (None for no splitting)
        free_memory_after_export: Whether to free memory after exporting
        memory_tracking: Whether to track memory usage
    
    Returns:
        Loaded dataset or result of splitting
    """
    # If ncpus is not specified, use the number of available cores
    if ncpus is None:
        import multiprocessing
        ncpus = multiprocessing.cpu_count()
    
    logger = get_logger("load_dataset")
    logger.info(f"Loading Nexar 2020 dataset with {ncpus} CPUs...")
    
    # Create dataset with no automatic loading
    dataset = Nexar2020Dataset(load_imgs=False, load_md=False, ncpus=ncpus)
    
    # Use the dataset's log_memory_usage method
    if memory_tracking and PSUTIL_AVAILABLE:
        dataset.log_memory_usage("Initial state")
    
    # Load images if requested
    if imgs:
        dataset.load_images(export=False)  # Don't export yet
        if memory_tracking and PSUTIL_AVAILABLE:
            dataset.log_memory_usage("After loading images")
    
    # Load metadata if requested
    if md:
        dataset.load_metadata(export=False, format=format)  # Don't export yet
        if memory_tracking and PSUTIL_AVAILABLE:
            dataset.log_memory_usage("After loading metadata")
    
    # Validate alignment if requested and both images and metadata are loaded
    if validate_align and imgs and md and dataset.imgs and not dataset.md.empty:
        logger.info("Validating image-metadata alignment...")
        dataset.validate_alignment(fix_misalignment=True)
    
    # Apply date filtering if specified
    if md and not dataset.md.empty:
        if dates:  # Specific dates filtering takes precedence
            # Improved parsing of dates from command line
            logger.info(f"Received dates parameter: {dates}")
            parsed_dates = []
            
            # Handle string representation of list passed from command line
            if isinstance(dates, str):
                try:
                    import json
                    # Try parsing as JSON
                    try:
                        parsed_dates = json.loads(dates)
                        logger.info(f"Parsed dates from JSON: {parsed_dates}")
                    except json.JSONDecodeError:
                        # If not valid JSON, try parsing as comma-separated string
                        parsed_dates = [d.strip() for d in dates.strip('[]').split(',')]
                        logger.info(f"Parsed dates from comma-separated string: {parsed_dates}")
                except Exception as e:
                    logger.error(f"Error parsing dates parameter: {e}")
                    parsed_dates = [dates]  # Use as single date if all else fails
            else:
                parsed_dates = dates  # Already a list or iterable
                
            logger.info(f"Filtering dataset by {len(parsed_dates)} specific dates: {parsed_dates}")
            filtered_imgs, filtered_md = dataset.filter_by_specific_dates(
                parsed_dates, export=False, format=format
            )
            # Update dataset with filtered data
            dataset.imgs = filtered_imgs
            dataset.md = filtered_md
            logger.success(f"Dataset filtered to {len(filtered_imgs)} images and {len(filtered_md)} metadata rows")
        elif start_date or end_date:  # Range-based filtering
            logger.info(f"Filtering dataset by date range: {start_date} to {end_date}")
            filtered_imgs, filtered_md = dataset.filter_by_date(
                start_date, end_date, export=False, format=format
            )
            # Update dataset with filtered data
            dataset.imgs = filtered_imgs
            dataset.md = filtered_md
            logger.success(f"Dataset filtered to {len(filtered_imgs)} images and {len(filtered_md)} metadata rows")
        
        if memory_tracking and PSUTIL_AVAILABLE:
            dataset.log_memory_usage("After date filtering")
    
    # Split dataset into chunks if requested
    if split_chunks is not None and split_chunks > 1:
        # Show statistics before splitting (and potentially freeing memory)
        if not no_stats:
            logger.info("Showing original dataset statistics (before splitting)...")
            dataset.export_statistics()
            
        logger.info(f"Splitting dataset into {split_chunks} chunks...")
        results = dataset.split_into_chunks(
            num_chunks=split_chunks, 
            export=export, 
            format=format,
            prefix="chunk",
            free_memory=free_memory_after_export
        )
        
        if memory_tracking and PSUTIL_AVAILABLE and not free_memory_after_export:
            dataset.log_memory_usage("After dataset splitting")
        
        # Free memory after chunking if requested
        if free_memory_after_export:
            logger.info("Freeing original dataset from memory")
            del dataset.imgs
            del dataset.md
            import gc
            gc.collect()  # Force garbage collection
        
        return results
    
    # Show statistics BEFORE exporting and freeing memory
    if not no_stats:
        logger.info("Showing dataset statistics...")
        dataset.export_statistics()
    
    # Export after showing statistics
    if export:
        if imgs and dataset.imgs:
            dataset._export_image_paths(free_memory=free_memory_after_export)
        if md and not dataset.md.empty:
            # Standard metadata export
            dataset._export_metadata(format=format, free_memory=False)  # Don't free memory yet
            
            # Also export as GeoParquet if requested
            if export_geoparquet:
                dataset._export_geospatial_metadata(format="geoparquet", free_memory=free_memory_after_export)
            elif free_memory_after_export:
                # If we didn't export geoparquet but still want to free memory
                import gc
                dataset.md = pd.DataFrame()
                gc.collect()
        
        if memory_tracking and PSUTIL_AVAILABLE and not free_memory_after_export:
            dataset.log_memory_usage("After exporting data")
    
    logger.success(f"Dataset loaded: {len(dataset.imgs)} images and {len(dataset.md)} metadata rows")
    
    return dataset


if __name__ == "__main__":
    # Expose simplified CLI with just the load function
    fire.Fire(load_nexar_dataset)




