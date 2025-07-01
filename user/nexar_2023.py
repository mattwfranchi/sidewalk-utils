# deos2ac 
# @mattwfranchi 

"""
Data input/output module for the deos2ac project.
Provides dataset class and utilities for loading and processing Nexar 2023 dataset.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import os
import fire
import pandas as pd 
import sys
import traceback

from conjectural_inspector.utils.logger import get_logger
from conjectural_inspector.utils.timer import time_it
from conjectural_inspector.user.base_nexar_dataset import BaseNexarDataset

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
NEXAR_2023 = "/share/ju/nexar_data/2023"
EXPORT_DIR = "/share/ju/matt/conjectural_inspector/data/dashcam"

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


class Nexar2023Dataset(BaseNexarDataset): 
    """Dataset class for the Nexar 2023 dataset."""
    
    def load_img_dir(self, img_dir):
        """Load images from a directory.
        
        Args:
            img_dir: Path to the directory containing images
            
        Returns:
            List of image paths
        """
        return list(Path(img_dir).glob("*.jpg"))
    
    def _process_dir_chunk(self, dir_chunk):
        """Process a chunk of directories and return all image paths.
        
        Args:
            dir_chunk: List of directories to process
            
        Returns:
            List of image paths
        """
        all_images = []
        try:
            for img_dir in dir_chunk:
                try:
                    images = self.load_img_dir(img_dir)
                    all_images.extend(images)
                except Exception as e:
                    # Handle exceptions within individual directory processing
                    error_info = traceback.format_exc()
                    print(f"Error processing directory {img_dir}: {str(e)}\n{error_info}")
        except Exception as e:
            # Catch any other exceptions to prevent worker crashes
            error_info = traceback.format_exc()
            print(f"Unexpected error in worker process: {str(e)}\n{error_info}")
        
        return all_images
    
    def _process_md_chunk(self, file_chunk):
        """Process a chunk of metadata CSV files and return combined DataFrame.
        
        Args:
            file_chunk: List of CSV files to process
            
        Returns:
            DataFrame with combined metadata
        """
        dfs = []
        try:
            for file_path in file_chunk:
                try:
                    # OPTIMIZATION: Use a more efficient CSV reader with optimized dtypes
                    # Read the first few rows to infer types
                    sample = pd.read_csv(file_path, nrows=1000)
                    # Create dtype dictionary for optimized loading
                    dtypes = {col: 'category' if sample[col].dtype == 'object' else sample[col].dtype for col in sample}
                    # Read the full file with optimized dtypes
                    df = pd.read_csv(file_path, dtype=dtypes)
                    dfs.append(df)
                except Exception as e:
                    # Log the error but continue with other files
                    error_info = traceback.format_exc()
                    print(f"Error reading {file_path}: {str(e)}\n{error_info}")
        except Exception as e:
            # Catch any other exceptions to prevent worker crashes
            error_info = traceback.format_exc()
            print(f"Unexpected error in worker process: {str(e)}\n{error_info}")
        
        if dfs:
            # OPTIMIZATION: Use a more efficient concat method
            combined = pd.concat(dfs, ignore_index=True, copy=False)
            # Clear memory
            dfs.clear()
            return combined
        else:
            return pd.DataFrame()
    
    @time_it(level="info", message="Loading images from Nexar 2023 dataset")
    def load_images(self, export=False):
        """Load images from the Nexar 2023 dataset.
        
        The 2023 dataset structure is:
        /share/ju/nexar_data/2023/
        ├── 2023-08-22/
        │   ├── 604222321527357439/  (H3 hexagon ID)
        │   │   ├── frames/          # Images are in this subdirectory
        │   │   │   ├── *.jpg (images)
        │   │   │   └── ...
        │   │   └── metadata.csv
        │   └── ...
        └── ...
        
        Args:
            export: If True, save the list of image paths to a file in EXPORT_DIR
        """
        # Navigate to the 2023 dataset root
        dataset_root = Path(NEXAR_2023)
        if not dataset_root.exists():
            self.logger.error(f"Dataset root not found: {dataset_root}")
            return []
        
        # Get all date folders (e.g., 2023-08-22, 2023-08-23, etc.)
        date_folders = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('2023-')]
        date_folders.sort()  # Sort chronologically
        
        if not date_folders:
            self.logger.error(f"No date folders found in {dataset_root}")
            return []
        
        self.logger.info(f"Found {len(date_folders)} date folders")
        
        # Collect all image directories (the 'frames' subdirectories containing images)
        all_image_dirs = []
        
        for date_folder in date_folders:
            # Get all H3 hexagon folders in this date
            h3_folders = [d for d in date_folder.iterdir() if d.is_dir() and d.name.isdigit()]
            
            for h3_folder in h3_folders:
                # Look for the 'frames' subdirectory within each H3 folder
                frames_dir = h3_folder / "frames"
                if frames_dir.exists() and frames_dir.is_dir():
                    # Check if this folder contains images
                    images = list(frames_dir.glob("*.jpg"))
                    if images:
                        all_image_dirs.append(frames_dir)
                        self.logger.debug(f"Found image directory: {frames_dir} with {len(images)} images")
        
        self.logger.info(f"Found {len(all_image_dirs)} directories containing images")
        
        if not all_image_dirs:
            self.logger.warning("No image directories found")
            return []
        
        # Distribute directories evenly across CPUs
        chunked_dirs = self._chunk_list(all_image_dirs, self.ncpus)
        
        futures = []
        with ProcessPoolExecutor(max_workers=self.ncpus) as executor:
            # Submit one task per CPU, where each task processes a chunk of directories
            for chunk in chunked_dirs:
                futures.append(executor.submit(self._process_dir_chunk, chunk))
            
            # Wait for all futures to complete and collect results
            for future in futures:
                result = future.result()  # This will be a list of image paths
                self.logger.info(f"Loaded {len(result)} images from task")
                self.imgs.extend(result)
        
        self.logger.success(f"Loaded {len(self.imgs)} images from Nexar 2023 dataset")
        
        # Align with metadata if it's already loaded
        if not self.md.empty:
            self.logger.info("Metadata already loaded, aligning with images...")
            self.align_images_with_metadata()
        
        # Export image paths to file if requested
        if export:
            self._export_image_paths()
            
        return self.imgs
    
    @time_it(level="info", message="Loading metadata from Nexar 2023 dataset")
    def load_metadata(self, export=False, format="parquet"):
        """Load metadata from the Nexar 2023 dataset.
        
        The metadata is stored in CSV files within each H3 hexagon folder:
        /share/ju/nexar_data/2023/
        ├── 2023-08-22/
        │   ├── 604222321527357439/
        │   │   ├── frames/          # Images
        │   │   └── metadata.csv     # Metadata file
        │   └── ...
        └── ...
        
        Args:
            export: If True, save the metadata DataFrame to a file in EXPORT_DIR
            format: Export format ('parquet' or 'csv')
        """
        # Navigate to the 2023 dataset root
        dataset_root = Path(NEXAR_2023)
        if not dataset_root.exists():
            self.logger.error(f"Dataset root not found: {dataset_root}")
            return pd.DataFrame()
        
        # Get all date folders
        date_folders = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('2023-')]
        date_folders.sort()
        
        if not date_folders:
            self.logger.error(f"No date folders found in {dataset_root}")
            return pd.DataFrame()
        
        # Collect all metadata CSV files
        all_metadata_files = []
        
        for date_folder in date_folders:
            # Get all H3 hexagon folders in this date
            h3_folders = [d for d in date_folder.iterdir() if d.is_dir() and d.name.isdigit()]
            
            for h3_folder in h3_folders:
                # Look for metadata.csv directly in the H3 folder
                metadata_file = h3_folder / "metadata.csv"
                if metadata_file.exists():
                    all_metadata_files.append(metadata_file)
                    self.logger.debug(f"Found metadata file: {metadata_file}")
        
        self.logger.info(f"Found {len(all_metadata_files)} metadata files")
        
        if not all_metadata_files:
            self.logger.warning("No metadata files found")
            return pd.DataFrame()
        
        # Sort files by size for better load balancing
        all_metadata_files.sort(key=lambda x: x.stat().st_size)
        
        # Distribute files evenly across CPUs
        chunked_files = self._chunk_list(all_metadata_files, self.ncpus)
        
        futures = []
        with ProcessPoolExecutor(max_workers=self.ncpus) as executor:
            for chunk in chunked_files:
                futures.append(executor.submit(self._process_md_chunk, chunk))
            
            dfs = [future.result() for future in futures]
        
        # Process the results
        self._process_metadata_results(dfs, export, format)
        
        if export:
            self._export_metadata(format=format)
            
        return self.md
    
    def _process_metadata_results(self, dfs, export=False, format="parquet"):
        """Process metadata results from parallel loading.
        
        Args:
            dfs: List of DataFrames from parallel processing
            export: Whether to export the combined metadata
            format: Export format
        """
        if dfs:
            self.logger.info("Concatenating metadata DataFrames")
            # Filter out empty DataFrames
            non_empty_dfs = [df for df in dfs if not df.empty]
            
            if non_empty_dfs:
                self.md = pd.concat(non_empty_dfs, ignore_index=True, copy=False)
                
                # Process the metadata based on the 2023 structure
                # The 2023 metadata might have different column names than 2020
                if 'image_ref' in self.md.columns:
                    # Extract frame_id from image_ref (similar to 2020)
                    self.md['frame_id'] = self.md['image_ref'].apply(lambda x: Path(str(x)).stem)
                elif 'filename' in self.md.columns:
                    # Alternative column name for 2023
                    self.md['frame_id'] = self.md['filename'].apply(lambda x: Path(str(x)).stem)
                else:
                    # If no obvious image reference column, try to infer from other columns
                    self.logger.warning("No image_ref or filename column found. Attempting to infer frame_id...")
                    # This would need to be customized based on actual 2023 metadata structure
                
                # Convert timestamp if present
                if 'timestamp' in self.md.columns:
                    if not pd.api.types.is_datetime64_any_dtype(self.md['timestamp']):
                        # Try to convert from epoch milliseconds
                        try:
                            self.md['timestamp'] = pd.to_datetime(self.md['timestamp'], unit='ms')
                            self.md['timestamp'] = self.md['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                        except:
                            # If that fails, try regular datetime parsing
                            self.md['timestamp'] = pd.to_datetime(self.md['timestamp'])
                
                self.logger.success(f"Processed {len(self.md)} metadata rows")
                
                if export:
                    self._export_metadata(format=format)
            else:
                self.md = pd.DataFrame()
                self.logger.warning("All metadata DataFrames were empty")
        else:
            self.logger.warning("No metadata files were loaded")
            self.md = pd.DataFrame()


# ===== Helper Functions =====

def filter_dataset_by_date(dataset, start_date=None, end_date=None, export=False, format="parquet"):
    """Filter a dataset by date range."""
    logger = get_logger("filter_dataset")
    
    # Log initial memory state
    log_memory_usage(logger, "Before filtering")
    
    # Check if metadata is available
    if dataset.md.empty:
        logger.warning("No metadata available for filtering. Load metadata first.")
        return dataset.imgs, dataset.md
    
    # If no dates specified, return the original dataset
    if not start_date and not end_date:
        logger.info("No date range specified, returning original dataset")
        return dataset.imgs, dataset.md
    
    # Convert string dates to datetime
    start = pd.to_datetime(start_date) if start_date else pd.Timestamp.min
    end = pd.to_datetime(end_date) if end_date else pd.Timestamp.max
    
    logger.info(f"Filtering dataset by date range: {start.date()} to {end.date()}")
    
    # Ensure timestamp column is properly converted to datetime
    if 'timestamp' in dataset.md.columns:
        # Check if timestamp is already in datetime format
        if not pd.api.types.is_datetime64_any_dtype(dataset.md['timestamp']):
            logger.info("Converting timestamp column to datetime format")
            
            # Use parallel processing if available
            if PANDARALLEL_AVAILABLE:
                logger.info("Using parallel processing for timestamp conversion")
                # Convert from epoch milliseconds to datetime with timezone
                dataset.md['timestamp'] = pd.to_datetime(dataset.md['timestamp'], unit='ms')
                dataset.md['timestamp'] = dataset.md['timestamp'].parallel_apply(
                    lambda x: x.tz_localize('UTC').tz_convert('US/Eastern')
                )
            else:
                # Standard processing
                dataset.md['timestamp'] = pd.to_datetime(dataset.md['timestamp'], unit='ms')
                dataset.md['timestamp'] = dataset.md['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                
            logger.info(f"Timestamp sample: {dataset.md['timestamp'].iloc[0]} (converted from epoch time)")
    else:
        logger.error("No timestamp column found in metadata. Cannot filter by date.")
        return dataset.imgs, dataset.md
    
    # OPTIMIZATION: Filter metadata by date range (vectorized operation)
    # Pre-compute date column instead of repeatedly calling .dt.date in the filter
    logger.info("Optimized filtering: Pre-computing date column")
    
    # Create date column for faster filtering
    dataset.md['date'] = dataset.md['timestamp'].dt.date
    
    # Filter using the pre-computed column (much faster)
    filtered_md = dataset.md[(dataset.md['date'] >= start.date()) & 
                           (dataset.md['date'] <= end.date())]
    
    log_memory_usage(logger, "After metadata filtering")
    
    # OPTIMIZATION: Filter images by frame_id using a hash-based approach (much faster than list comprehension)
    filtered_imgs = []
    if dataset.imgs:
        # Create a hash set for O(1) lookups
        logger.info("Creating optimized hash set for frame_id matching")
        filtered_frame_ids = set(filtered_md['frame_id'])
        
        if len(dataset.imgs) > 10000:
            # OPTIMIZATION: For very large datasets, use batch processing to improve memory usage
            logger.info(f"Using optimized batch processing to filter {len(dataset.imgs)} images")
            batch_size = 50000  # Process images in batches of 50k
            filtered_imgs = []
            
            for i in range(0, len(dataset.imgs), batch_size):
                batch = dataset.imgs[i:i+batch_size]
                # Process this batch
                batch_results = [img for img in batch if Path(img).stem in filtered_frame_ids]
                filtered_imgs.extend(batch_results)
                
                if i % 100000 == 0 and i > 0:
                    logger.info(f"Processed {i} of {len(dataset.imgs)} images...")
        else:
            # Standard filtering for smaller lists using the hash set
            filtered_imgs = [img for img in dataset.imgs if Path(img).stem in filtered_frame_ids]
    
    log_memory_usage(logger, "After image filtering")
    
    logger.success(f"Filtered dataset contains {len(filtered_imgs)} images and {len(filtered_md)} metadata rows")
    
    # Export if requested
    if export:
        date_range = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
        
        if filtered_imgs:
            dataset._export_image_paths(subset=filtered_imgs, prefix=f"filtered_{date_range}")
        
        if not filtered_md.empty:
            dataset._export_metadata(subset=filtered_md, prefix=f"filtered_{date_range}", format=format)
    
    return filtered_imgs, filtered_md

def filter_dataset_by_specific_dates(dataset, dates, export=False, format="parquet"):
    """Filter a dataset to include only records from specific dates.
    
    Args:
        dataset: The dataset to filter
        dates: List of dates as strings (YYYY-MM-DD) or datetime objects
        export: Whether to export the filtered data
        format: Export format for metadata ('parquet' or 'csv')
        
    Returns:
        Tuple of (filtered_images, filtered_metadata)
    """
    logger = get_logger("filter_dataset_dates")
    
    # Log initial memory state
    log_memory_usage(logger, "Before filtering")
    
    # Check if metadata is available
    if dataset.md.empty:
        logger.warning("No metadata available for filtering. Load metadata first.")
        return dataset.imgs, dataset.md
    
    # If no dates specified, return the original dataset
    if not dates or len(dates) == 0:
        logger.info("No dates specified, returning original dataset")
        return dataset.imgs, dataset.md
    
    # Convert string dates to datetime objects
    parsed_dates = []
    for date_str in dates:
        try:
            # Handle both date objects and strings
            if isinstance(date_str, (datetime, pd.Timestamp)):
                parsed_dates.append(pd.Timestamp(date_str).date())
            else:
                # Try different formats
                try:
                    # Try standard ISO format first
                    parsed_dates.append(pd.to_datetime(date_str).date())
                except:
                    # Try other common formats if ISO fails
                    formats = ['%Y-%m-%d', '%m-%d-%Y', '%m/%d/%Y', '%d-%m-%Y']
                    for fmt in formats:
                        try:
                            parsed_dates.append(pd.to_datetime(date_str, format=fmt).date())
                            break
                        except:
                            continue
                    else:
                        logger.warning(f"Couldn't parse date: {date_str}, skipping")
        except Exception as e:
            logger.warning(f"Error parsing date {date_str}: {e}")
    
    if not parsed_dates:
        logger.warning("No valid dates found in the provided list")
        return dataset.imgs, dataset.md
    
    logger.info(f"Filtering dataset for {len(parsed_dates)} specific dates: {', '.join(str(d) for d in parsed_dates)}")
    
    # Ensure timestamp column is properly converted to datetime
    if 'timestamp' in dataset.md.columns:
        # Check if timestamp is already in datetime format
        if not pd.api.types.is_datetime64_any_dtype(dataset.md['timestamp']):
            logger.info("Converting timestamp column to datetime format")
            
            # Use parallel processing if available
            if PANDARALLEL_AVAILABLE:
                logger.info("Using parallel processing for timestamp conversion")
                # Convert from epoch milliseconds to datetime with timezone
                dataset.md['timestamp'] = pd.to_datetime(dataset.md['timestamp'], unit='ms')
                dataset.md['timestamp'] = dataset.md['timestamp'].parallel_apply(
                    lambda x: x.tz_localize('UTC').tz_convert('US/Eastern')
                )
            else:
                # Standard processing
                dataset.md['timestamp'] = pd.to_datetime(dataset.md['timestamp'], unit='ms')
                dataset.md['timestamp'] = dataset.md['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                
            logger.info(f"Timestamp sample: {dataset.md['timestamp'].iloc[0]} (converted from epoch time)")
    else:
        logger.error("No timestamp column found in metadata. Cannot filter by date.")
        return dataset.imgs, dataset.md
    
    # Filter metadata to include only records from the specified dates
    # This converts timestamp to date and checks if it's in our list of parsed dates
    filtered_md = dataset.md[dataset.md['timestamp'].dt.date.isin(parsed_dates)]
    
    log_memory_usage(logger, "After metadata filtering")
    
    # Filter images by frame_id using parallel operations if available
    filtered_imgs = []
    if dataset.imgs:
        filtered_frame_ids = set(filtered_md['frame_id'])
        
        # Use parallel processing for large image lists
        if len(dataset.imgs) > 10000 and PANDARALLEL_AVAILABLE:
            logger.info(f"Using parallel processing to filter {len(dataset.imgs)} images")
            # Create a DataFrame for parallel filtering
            img_df = pd.DataFrame({'path': dataset.imgs})
            img_df['frame_id'] = img_df['path'].parallel_apply(lambda x: Path(x).stem)
            img_df['keep'] = img_df['frame_id'].parallel_apply(lambda x: x in filtered_frame_ids)
            filtered_imgs = [Path(p) for p in img_df[img_df['keep']]['path']]
        else:
            # Standard filtering for smaller lists
            filtered_imgs = [img for img in dataset.imgs if Path(img).stem in filtered_frame_ids]
    
    log_memory_usage(logger, "After image filtering")
    
    logger.success(f"Filtered dataset contains {len(filtered_imgs)} images and {len(filtered_md)} metadata rows")
    
    # Export if requested
    if export:
        date_str = "specific_dates"
        if len(parsed_dates) <= 3:
            # Use abbreviated format for small number of dates
            date_str = "_".join([d.strftime('%Y%m%d') for d in parsed_dates])
        
        if filtered_imgs:
            dataset._export_image_paths(subset=filtered_imgs, prefix=f"filtered_{date_str}")
        
        if not filtered_md.empty:
            dataset._export_metadata(subset=filtered_md, prefix=f"filtered_{date_str}", format=format)
    
    return filtered_imgs, filtered_md

def export_dataset_statistics(dataset, output_file=None):
    """Generate and export statistics about a dataset.
    
    Args:
        dataset: The dataset to analyze
        output_file: Path to output file, or None to print to console
        
    Returns:
        Dictionary containing statistics
    """
    logger = get_logger("dataset_stats")
    
    stats = {
        "dataset_type": dataset.__class__.__name__,
        "images": {},
        "metadata": {}
    }
    
    # Image statistics
    if dataset.imgs:
        stats["images"]["count"] = len(dataset.imgs)
        
        # Directory statistics
        dirs = {}
        for img in dataset.imgs:
            parent = str(Path(img).parent)
            if parent in dirs:
                dirs[parent] += 1
            else:
                dirs[parent] = 1
        
        stats["images"]["directories"] = {
            "count": len(dirs),
            "avg_images_per_dir": sum(dirs.values()) / len(dirs) if dirs else 0
        }
    else:
        stats["images"]["count"] = 0
    
    # Metadata statistics
    if not dataset.md.empty:
        stats["metadata"]["count"] = len(dataset.md)
        
        # Date range
        min_date = dataset.md['timestamp'].min()
        max_date = dataset.md['timestamp'].max()
        stats["metadata"]["date_range"] = {
            "start": min_date.strftime("%Y-%m-%d"),
            "end": max_date.strftime("%Y-%m-%d")
        }
        
        # Statistics by day
        day_counts = dataset.md.groupby(dataset.md['timestamp'].dt.date).size()
        stats["metadata"]["by_day"] = {
            "num_days": len(day_counts),
            "avg_rows_per_day": float(day_counts.mean()),
            "min_rows": {
                "count": int(day_counts.min()),
                "date": day_counts.idxmin().strftime("%Y-%m-%d")
            },
            "max_rows": {
                "count": int(day_counts.max()),
                "date": day_counts.idxmax().strftime("%Y-%m-%d")
            }
        }
    else:
        stats["metadata"]["count"] = 0
    
    # Format output
    output_text = []
    output_text.append(f"\n===== {stats['dataset_type']} Statistics =====")
    
    if stats["images"]["count"] > 0:
        output_text.append("\nImages:")
        output_text.append(f"  Total count: {stats['images']['count']}")
        output_text.append(f"  Number of directories: {stats['images']['directories']['count']}")
        output_text.append(f"  Average images per directory: {stats['images']['directories']['avg_images_per_dir']:.1f}")
    else:
        output_text.append("\nImages: None loaded")
    
    if stats["metadata"]["count"] > 0:
        output_text.append("\nMetadata:")
        output_text.append(f"  Total rows: {stats['metadata']['count']}")
        output_text.append(f"  Date range: {stats['metadata']['date_range']['start']} to {stats['metadata']['date_range']['end']}")
        output_text.append(f"  Number of days: {stats['metadata']['by_day']['num_days']}")
        output_text.append(f"  Average rows per day: {stats['metadata']['by_day']['avg_rows_per_day']:.1f}")
        output_text.append(f"  Min rows on a day: {stats['metadata']['by_day']['min_rows']['count']} ({stats['metadata']['by_day']['min_rows']['date']})")
        output_text.append(f"  Max rows on a day: {stats['metadata']['by_day']['max_rows']['count']} ({stats['metadata']['by_day']['max_rows']['date']})")
    else:
        output_text.append("\nMetadata: None loaded")
    
    output_text.append("\n=========================================")
    
    # Format as string
    output_str = "\n".join(output_text)
    
    # Output to file or console
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_str)
        logger.success(f"Statistics exported to {output_file}")
    else:
        print(output_str)
    
    return stats


def split_dataset_into_chunks(dataset, num_chunks=2, export=True, format="parquet", prefix="chunk", free_memory=True):
    """Split a dataset into evenly-sized chunks for distributed processing.
    
    This function is useful for dividing a dataset across multiple GPUs or nodes.
    The function preserves alignment between images and metadata in each chunk.
    
    Args:
        dataset: The dataset to split
        num_chunks: Number of chunks to create
        export: Whether to export the chunks to files
        format: Export format for metadata ('parquet' or 'csv')
        prefix: Prefix for exported files
        free_memory: Whether to delete chunks from memory after export
    
    Returns:
        List of (chunk_images, chunk_metadata) tuples, or just export paths if free_memory=True
    """
    logger = get_logger("split_dataset")
    
    # First ensure that images and metadata are aligned
    validate_image_metadata_alignment(dataset)
    
    # Calculate chunk sizes
    total_items = len(dataset.imgs)  # Should be the same as len(dataset.md) after validation
    
    if total_items == 0:
        logger.warning("Dataset is empty, cannot split")
        return []
    
    # Adjust num_chunks if we have fewer items than requested chunks
    if total_items < num_chunks:
        logger.warning(f"Fewer items ({total_items}) than requested chunks ({num_chunks}). Adjusting to {total_items} chunks.")
        num_chunks = total_items
    
    # Calculate base chunk size and remainder
    base_chunk_size = total_items // num_chunks
    remainder = total_items % num_chunks
    
    logger.info(f"Splitting dataset with {total_items} items into {num_chunks} chunks")
    
    # Create chunks
    chunks = []
    export_paths = []
    start_idx = 0
    
    for i in range(num_chunks):
        # Add one extra item to early chunks if we have a remainder
        chunk_size = base_chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        
        # Extract chunk data
        chunk_imgs = dataset.imgs[start_idx:end_idx]
        chunk_md = dataset.md.iloc[start_idx:end_idx].copy()
        
        logger.info(f"Chunk {i+1}: {len(chunk_imgs)} items (indices {start_idx}-{end_idx-1})")
        
        # Export if requested
        if export:
            chunk_prefix = f"{prefix}_{i+1}_of_{num_chunks}"
            
            # Create a temporary dataset for export
            temp_dataset = type(dataset)(load_imgs=False, load_md=False, ncpus=dataset.ncpus)
            temp_dataset.imgs = chunk_imgs
            temp_dataset.md = chunk_md
            
            # Export
            img_path = temp_dataset._export_image_paths(prefix=chunk_prefix)
            md_path = temp_dataset._export_metadata(format=format, prefix=chunk_prefix)
            
            export_paths.append((img_path, md_path))
            logger.success(f"Exported chunk {i+1} to {img_path} and {md_path}")
            
            # Free memory after export if requested
            if free_memory:
                # Delete temporary dataset to free memory
                del temp_dataset
                # Don't store chunks in memory
                continue
        
        # Add to results (only if we're not freeing memory)
        chunks.append((chunk_imgs, chunk_md))
        
        # Update start index for next chunk
        start_idx = end_idx
    
    logger.success(f"Successfully split dataset into {num_chunks} chunks")
    
    # Return export paths if we freed memory, otherwise return chunks
    if export and free_memory:
        import gc
        gc.collect()  # Force garbage collection
        return export_paths
    else:
        return chunks


def validate_image_metadata_alignment(dataset, fix_misalignment=True):
    """Validate and optionally fix the alignment between images and metadata.
    
    Args:
        dataset: The dataset to validate
        fix_misalignment: Whether to fix misalignments by dropping unmatched entries
        
    Returns:
        Boolean indicating whether the dataset is aligned (after fixing if requested)
    """
    logger = get_logger("validate_alignment")
    
    if not dataset.imgs or dataset.md.empty:
        logger.warning("Cannot validate: missing either images or metadata")
        return False
    
    # Extract frame_ids from images and metadata
    img_frame_ids = [Path(img).stem for img in dataset.imgs]
    md_frame_ids = dataset.md['frame_id'].tolist()
    
    # Check if lengths match
    if len(img_frame_ids) != len(md_frame_ids):
        logger.warning(f"Mismatched length: {len(img_frame_ids)} images vs {len(md_frame_ids)} metadata rows")
        is_aligned = False
    else:
        # Check if corresponding frame_ids match
        mismatches = sum(1 for i, (img_id, md_id) in enumerate(zip(img_frame_ids, md_frame_ids)) if img_id != md_id)
        if mismatches > 0:
            logger.warning(f"Found {mismatches} frame_id mismatches between images and metadata")
            is_aligned = False
        else:
            logger.success(f"Perfect alignment: {len(img_frame_ids)} images match exactly with metadata rows")
            is_aligned = True
    
    # Fix misalignment if requested and needed
    if not is_aligned and fix_misalignment:
        logger.info("Fixing misalignment by ensuring perfect 1:1 mapping between images and metadata")
        # Force re-alignment of images and metadata
        dataset.align_images_with_metadata()
        
        # Verify the fix worked
        if len(dataset.imgs) == len(dataset.md):
            fixed_img_ids = [Path(img).stem for img in dataset.imgs]
            fixed_md_ids = dataset.md['frame_id'].tolist()
            mismatches = sum(1 for i, (img_id, md_id) in enumerate(zip(fixed_img_ids, fixed_md_ids)) if img_id != md_id)
            
            if mismatches == 0:
                logger.success(f"Successfully fixed alignment: {len(dataset.imgs)} images match exactly with metadata rows")
                is_aligned = True
            else:
                logger.error(f"Failed to fix alignment: still have {mismatches} mismatches")
        else:
            logger.error(f"Failed to fix alignment: still have length mismatch ({len(dataset.imgs)} images vs {len(dataset.md)} metadata rows)")
    
    return is_aligned


def get_memory_usage(detailed=False):
    """Get current memory usage information.
    
    Args:
        detailed: If True, return detailed memory statistics
        
    Returns:
        Dictionary with memory usage information
    """
    if not PSUTIL_AVAILABLE:
        return {"available": False, "message": "psutil not installed"}
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Basic memory usage
    memory_usage = {
        "available": True,
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
    }
    
    # Add system-wide memory info
    system_memory = psutil.virtual_memory()
    memory_usage["system"] = {
        "total_gb": system_memory.total / (1024**3),
        "available_gb": system_memory.available / (1024**3),
        "percent_used": system_memory.percent
    }
    
    if detailed:
        # Add more detailed memory info
        memory_usage["detailed"] = {
            "uss_mb": getattr(memory_info, 'uss', 0) / (1024 * 1024),  # Unique Set Size if available
            "swap_mb": getattr(memory_info, 'swap', 0) / (1024 * 1024),  # Swap memory if available
        }
    
    return memory_usage

def log_memory_usage(logger, operation_name=""):
    """Log the current memory usage.
    
    Args:
        logger: Logger to use
        operation_name: Name of the operation being performed
    """
    if not PSUTIL_AVAILABLE:
        return
    
    memory = get_memory_usage()
    prefix = f"[{operation_name}] " if operation_name else ""
    
    logger.info(f"{prefix}Memory usage: {memory['rss_mb']:.1f} MB (RSS), "
                f"System: {memory['system']['percent_used']}% used, "
                f"{memory['system']['available_gb']:.1f} GB available")


def load_nexar_dataset(imgs=True, md=True, ncpus=8, 
                      start_date=None, end_date=None, dates=None, export=True, 
                      format="parquet", export_geoparquet=True, no_stats=False, 
                      validate_align=True, split_chunks=None, 
                      free_memory_after_export=True, memory_tracking=True):
    """Load a Nexar 2023 dataset, optionally filter by date, and show statistics.
    
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
    logger.info(f"Loading Nexar 2023 dataset with {ncpus} CPUs...")
    
    if memory_tracking and PSUTIL_AVAILABLE:
        log_memory_usage(logger, "Initial state")
    
    # Create dataset with no automatic loading
    dataset = Nexar2023Dataset(load_imgs=False, load_md=False, ncpus=ncpus)
    
    # Load images if requested
    if imgs:
        dataset.load_images(export=False)  # Don't export yet
        if memory_tracking and PSUTIL_AVAILABLE:
            log_memory_usage(logger, "After loading images")
    
    # Load metadata if requested
    if md:
        dataset.load_metadata(export=False, format=format)  # Don't export yet
        if memory_tracking and PSUTIL_AVAILABLE:
            log_memory_usage(logger, "After loading metadata")
    
    # Validate alignment if requested and both images and metadata are loaded
    if validate_align and imgs and md and dataset.imgs and not dataset.md.empty:
        logger.info("Validating image-metadata alignment...")
        validate_image_metadata_alignment(dataset, fix_misalignment=True)
    
    # Apply date filtering if specified
    if md and not dataset.md.empty:
        if dates:  # Specific dates filtering takes precedence
            logger.info(f"Filtering dataset by {len(dates)} specific dates")
            filtered_imgs, filtered_md = filter_dataset_by_specific_dates(
                dataset, dates, export=False, format=format
            )
            # Update dataset with filtered data
            dataset.imgs = filtered_imgs
            dataset.md = filtered_md
            logger.success(f"Dataset filtered to {len(filtered_imgs)} images and {len(filtered_md)} metadata rows")
        elif start_date or end_date:  # Range-based filtering
            logger.info(f"Filtering dataset by date range: {start_date} to {end_date}")
            filtered_imgs, filtered_md = filter_dataset_by_date(
                dataset, start_date, end_date, export=False, format=format
            )
            # Update dataset with filtered data
            dataset.imgs = filtered_imgs
            dataset.md = filtered_md
            logger.success(f"Dataset filtered to {len(filtered_imgs)} images and {len(filtered_md)} metadata rows")
        
        if memory_tracking and PSUTIL_AVAILABLE:
            log_memory_usage(logger, "After date filtering")
    
    # Split dataset into chunks if requested
    if split_chunks is not None and split_chunks > 1:
        # Show statistics before splitting (and potentially freeing memory)
        if not no_stats:
            logger.info("Showing original dataset statistics (before splitting)...")
            export_dataset_statistics(dataset)
            
        logger.info(f"Splitting dataset into {split_chunks} chunks...")
        results = split_dataset_into_chunks(
            dataset, 
            num_chunks=split_chunks, 
            export=export, 
            format=format,
            free_memory=free_memory_after_export
        )
        
        if memory_tracking and PSUTIL_AVAILABLE and not free_memory_after_export:
            log_memory_usage(logger, "After dataset splitting")
        
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
        export_dataset_statistics(dataset)
    
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
            log_memory_usage(logger, "After exporting data")
    
    logger.success(f"Dataset loaded: {len(dataset.imgs)} images and {len(dataset.md)} metadata rows")
    
    return dataset


if __name__ == "__main__":
    # Expose simplified CLI with just the load function
    fire.Fire(load_nexar_dataset)




