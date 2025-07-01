# deos2ac 
# @mattwfranchi 

"""
OPTIMIZED Data input/output module for the deos2ac project.
Provides highly optimized dataset class for loading and processing Nexar 2023 dataset.
"""

import os
import sys
import time
import gc
import tempfile
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Set, Tuple, Optional
import fire
import pandas as pd
import numpy as np
import traceback

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

# PyArrow for efficient parquet operations
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# ===== Constants =====
NEXAR_2023 = "/share/ju/nexar_data/2023"
EXPORT_DIR = "/share/ju/matt/conjectural-inspector/conjectural_inspector/data/dashcam"

# Initialize pandarallel at module level if available
if PANDARALLEL_AVAILABLE:
    pandarallel.initialize(progress_bar=False, verbose=0)

# Import after setting up dependencies
from conjectural_inspector.utils.logger import get_logger
from conjectural_inspector.utils.timer import time_it


# ===== Module-level helper functions for multiprocessing =====

def process_date_folder(date_folder: Path) -> List[Path]:
    """Process a date folder to find image directories (module-level for multiprocessing)."""
    image_dirs = []
    try:
        # Get all H3 hexagon folders in this date
        h3_folders = [d for d in date_folder.iterdir() 
                     if d.is_dir() and d.name.isdigit()]
        
        for h3_folder in h3_folders:
            frames_dir = h3_folder / "frames"
            if frames_dir.exists() and frames_dir.is_dir():
                # Check if this folder contains images (quick check)
                try:
                    # Use a more efficient check - just look for first jpg file
                    if next(frames_dir.glob("*.jpg"), None) is not None:
                        image_dirs.append(frames_dir)
                except (StopIteration, PermissionError, OSError):
                    # Skip if we can't access the directory
                    continue
    except Exception as e:
        # Use print for multiprocessing compatibility
        print(f"Error processing date folder {date_folder}: {e}")
    
    return image_dirs


def process_date_folder_metadata(date_folder: Path) -> List[Path]:
    """Process a date folder to find metadata files (module-level for multiprocessing)."""
    metadata_files = []
    try:
        h3_folders = [d for d in date_folder.iterdir() 
                     if d.is_dir() and d.name.isdigit()]
        
        for h3_folder in h3_folders:
            metadata_file = h3_folder / "metadata.csv"
            if metadata_file.exists():
                metadata_files.append(metadata_file)
    except Exception as e:
        # Use print for multiprocessing compatibility
        print(f"Error processing date folder {date_folder}: {e}")
    
    return metadata_files


def load_images_from_directory(img_dir: Path) -> List[Path]:
    """Load images from a single directory efficiently (module-level for multiprocessing)."""
    try:
        # Use list comprehension for better performance
        images = [f for f in img_dir.glob("*.jpg")]
        return images
    except Exception as e:
        # Use print for multiprocessing compatibility
        print(f"Error loading images from {img_dir}: {e}")
        return []


def process_image_chunk(dir_chunk: List[Path]) -> List[Path]:
    """Process a chunk of directories and return all image paths (module-level for multiprocessing)."""
    all_images = []
    try:
        for img_dir in dir_chunk:
            try:
                images = load_images_from_directory(img_dir)
                all_images.extend(images)
            except Exception as e:
                # Use print for multiprocessing compatibility
                print(f"Error processing directory {img_dir}: {e}")
    except Exception as e:
        # Use print for multiprocessing compatibility
        print(f"Unexpected error in worker process: {e}")
    
    return all_images


def load_metadata_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Load a single metadata file efficiently (module-level for multiprocessing)."""
    try:
        # Use optimized CSV reading with chunking for large files
        file_size = file_path.stat().st_size
        chunk_size = 100000  # Read 100k rows at a time for large files
        
        if file_size > 100 * 1024 * 1024:  # Files larger than 100MB
            # Read in chunks to avoid memory issues
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                chunks.clear()  # Free memory
                return df
            else:
                return None
        else:
            # For smaller files, read directly
            return pd.read_csv(file_path, low_memory=False)
            
    except Exception as e:
        # Use print for multiprocessing compatibility
        print(f"Error reading {file_path}: {e}")
        return None


def process_metadata_chunk(file_chunk: List[Path]) -> pd.DataFrame:
    """Process a chunk of metadata CSV files and return combined DataFrame (module-level for multiprocessing)."""
    dfs = []
    try:
        for file_path in file_chunk:
            df = load_metadata_file(file_path)
            if df is not None and not df.empty:
                dfs.append(df)
    except Exception as e:
        # Use print for multiprocessing compatibility
        print(f"Unexpected error in metadata worker process: {e}")
    
    if dfs:
        # Use efficient concatenation
        combined = pd.concat(dfs, ignore_index=True, copy=False)
        dfs.clear()  # Free memory
        return combined
    else:
        return pd.DataFrame()


def chunk_list(lst: List, num_chunks: int) -> List[List]:
    """Split a list into roughly equal chunks (module-level for multiprocessing)."""
    if not lst:
        return []
    
    # If we have fewer items than chunks, just return individual items
    if len(lst) <= num_chunks:
        return [[item] for item in lst]
    
    # For metadata files, we want to balance by file size, not just count
    # Check if the first item has a stat() method (it's a Path)
    if hasattr(lst[0], 'stat'):
        # Sort by file size (largest first) for better load balancing
        lst = sorted(lst, key=lambda x: x.stat().st_size, reverse=True)
    
    # Use a more sophisticated chunking algorithm
    chunks = [[] for _ in range(num_chunks)]
    chunk_sizes = [0] * num_chunks
    
    # Distribute items to the chunk with the smallest current size
    for item in lst:
        # Find the chunk with the smallest current size
        min_chunk_idx = chunk_sizes.index(min(chunk_sizes))
        
        # Add item to that chunk
        chunks[min_chunk_idx].append(item)
        
        # Update the chunk size (for files, use file size; for others, use count)
        if hasattr(item, 'stat'):
            chunk_sizes[min_chunk_idx] += item.stat().st_size
        else:
            chunk_sizes[min_chunk_idx] += 1
    
    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk]
    
    return chunks


class OptimizedNexar2023Dataset:
    """Highly optimized dataset class for the Nexar 2023 dataset."""
    
    def __init__(self, ncpus: int = 8, memory_limit_gb: float = 50.0):
        """Initialize the optimized dataset.
        
        Args:
            ncpus: Number of CPU cores to use for parallel processing
            memory_limit_gb: Memory limit in GB to prevent OOM
        """
        self.ncpus = ncpus
        self.memory_limit_gb = memory_limit_gb
        self.logger = get_logger("nexar2023_optimized")
        
        # Initialize empty containers
        self.imgs: List[Path] = []
        self.md: pd.DataFrame = pd.DataFrame()
        
        # Create export directory
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        # Memory tracking
        self._last_memory_check = time.time()
        self._memory_check_interval = 30  # Check memory every 30 seconds
        
    def _check_memory_usage(self, operation: str = ""):
        """Check memory usage and log if needed."""
        if not PSUTIL_AVAILABLE:
            return
            
        current_time = time.time()
        if current_time - self._last_memory_check < self._memory_check_interval:
            return
            
        self._last_memory_check = current_time
        
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_gb = memory_info.rss / (1024**3)
        
        system_memory = psutil.virtual_memory()
        system_used_gb = system_memory.used / (1024**3)
        system_total_gb = system_memory.total / (1024**3)
        
        self.logger.info(f"[{operation}] Memory: {rss_gb:.1f}GB (RSS), "
                        f"System: {system_used_gb:.1f}GB/{system_total_gb:.1f}GB "
                        f"({system_memory.percent:.1f}%)")
        
        # Force garbage collection if memory usage is high
        if rss_gb > self.memory_limit_gb * 0.8:
            self.logger.warning(f"High memory usage detected ({rss_gb:.1f}GB), forcing garbage collection")
            gc.collect()
    
    def _discover_image_directories(self) -> List[Path]:
        """Discover all image directories efficiently using parallel processing."""
        self.logger.info("Discovering image directories...")
        self._check_memory_usage("discovery_start")
        
        dataset_root = Path(NEXAR_2023)
        if not dataset_root.exists():
            self.logger.error(f"Dataset root not found: {dataset_root}")
            return []
        
        # Get all date folders
        date_folders = [d for d in dataset_root.iterdir() 
                       if d.is_dir() and d.name.startswith('2023-')]
        date_folders.sort()
        
        self.logger.info(f"Found {len(date_folders)} date folders")
        
        # Process in parallel using module-level function with timeout
        all_image_dirs = []
        
        # Use a smaller number of workers for discovery to avoid overwhelming the filesystem
        discovery_workers = min(4, self.ncpus)
        self.logger.info(f"Using {discovery_workers} workers for directory discovery")
        
        try:
            with ProcessPoolExecutor(max_workers=discovery_workers) as executor:
                # Submit all tasks
                future_to_folder = {executor.submit(process_date_folder, folder): folder 
                                  for folder in date_folders}
                
                # Collect results with timeout and progress logging
                completed = 0
                for future in future_to_folder:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per folder
                        all_image_dirs.extend(result)
                        completed += 1
                        
                        # Log progress every 10 folders
                        if completed % 10 == 0:
                            self.logger.info(f"Directory discovery progress: {completed}/{len(date_folders)} folders processed")
                            
                    except Exception as e:
                        folder = future_to_folder[future]
                        self.logger.error(f"Error processing folder {folder}: {e}")
                        completed += 1
                        # Continue with other folders
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error in parallel discovery: {e}")
            # Fallback to sequential processing
            self.logger.info("Falling back to sequential directory discovery...")
            for folder in date_folders:
                try:
                    result = process_date_folder(folder)
                    all_image_dirs.extend(result)
                except Exception as e:
                    self.logger.error(f"Error processing folder {folder}: {e}")
                    continue
        
        self.logger.info(f"Found {len(all_image_dirs)} directories containing images")
        self._check_memory_usage("discovery_complete")
        
        return all_image_dirs
    
    @time_it(level="info", message="Loading images from Nexar 2023 dataset")
    def load_images(self, export: bool = False) -> List[Path]:
        """Load images from the Nexar 2023 dataset with optimized memory usage."""
        self.logger.info(f"Loading Nexar 2023 dataset with {self.ncpus} CPUs...")
        self._check_memory_usage("load_images_start")
        
        # Discover image directories
        all_image_dirs = self._discover_image_directories()
        
        if not all_image_dirs:
            self.logger.warning("No image directories found")
            return []
        
        # Distribute directories evenly across CPUs
        chunked_dirs = chunk_list(all_image_dirs, self.ncpus)
        
        # Log chunk distribution for monitoring
        self.logger.info(f"Total image directories: {len(all_image_dirs)}")
        
        for i, chunk in enumerate(chunked_dirs):
            self.logger.info(f"Image chunk {i+1}: {len(chunk)} directories ({len(chunk)/len(all_image_dirs)*100:.1f}%)")
        
        # Process in parallel with memory monitoring
        futures = []
        with ProcessPoolExecutor(max_workers=self.ncpus) as executor:
            for chunk in chunked_dirs:
                futures.append(executor.submit(process_image_chunk, chunk))
            
            # Collect results with progress tracking
            total_loaded = 0
            for i, future in enumerate(futures):
                result = future.result()
                total_loaded += len(result)
                self.imgs.extend(result)
                
                # Log progress and check memory
                self.logger.info(f"Task {i+1}/{len(futures)}: Loaded {len(result)} images (Total: {total_loaded})")
                self._check_memory_usage(f"task_{i+1}")
        
        self.logger.success(f"Loaded {len(self.imgs)} images from Nexar 2023 dataset")
        self._check_memory_usage("load_images_complete")
        
        # Export if requested
        if export:
            self._export_image_paths()
        
        return self.imgs
    
    def _discover_metadata_files(self) -> List[Path]:
        """Discover all metadata files efficiently."""
        self.logger.info("Discovering metadata files...")
        self._check_memory_usage("metadata_discovery_start")
        
        dataset_root = Path(NEXAR_2023)
        if not dataset_root.exists():
            self.logger.error(f"Dataset root not found: {dataset_root}")
            return []
        
        # Get all date folders
        date_folders = [d for d in dataset_root.iterdir() 
                       if d.is_dir() and d.name.startswith('2023-')]
        date_folders.sort()
        
        # Process in parallel using module-level function with timeout
        all_metadata_files = []
        
        # Use a smaller number of workers for discovery to avoid overwhelming the filesystem
        discovery_workers = min(4, self.ncpus)
        self.logger.info(f"Using {discovery_workers} workers for metadata discovery")
        
        try:
            with ProcessPoolExecutor(max_workers=discovery_workers) as executor:
                # Submit all tasks
                future_to_folder = {executor.submit(process_date_folder_metadata, folder): folder 
                                  for folder in date_folders}
                
                # Collect results with timeout and progress logging
                completed = 0
                for future in future_to_folder:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per folder
                        all_metadata_files.extend(result)
                        completed += 1
                        
                        # Log progress every 10 folders
                        if completed % 10 == 0:
                            self.logger.info(f"Metadata discovery progress: {completed}/{len(date_folders)} folders processed")
                            
                    except Exception as e:
                        folder = future_to_folder[future]
                        self.logger.error(f"Error processing folder {folder}: {e}")
                        completed += 1
                        # Continue with other folders
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error in parallel metadata discovery: {e}")
            # Fallback to sequential processing
            self.logger.info("Falling back to sequential metadata discovery...")
            for folder in date_folders:
                try:
                    result = process_date_folder_metadata(folder)
                    all_metadata_files.extend(result)
                except Exception as e:
                    self.logger.error(f"Error processing folder {folder}: {e}")
                    continue
        
        self.logger.info(f"Found {len(all_metadata_files)} metadata files")
        self._check_memory_usage("metadata_discovery_complete")
        
        return all_metadata_files
    
    @time_it(level="info", message="Loading metadata from Nexar 2023 dataset")
    def load_metadata(self, export: bool = False, format: str = "parquet") -> pd.DataFrame:
        """Load metadata from the Nexar 2023 dataset with optimized memory usage."""
        self.logger.info(f"Loading metadata with {self.ncpus} CPUs...")
        self._check_memory_usage("load_metadata_start")
        
        # Discover metadata files
        all_metadata_files = self._discover_metadata_files()
        
        if not all_metadata_files:
            self.logger.warning("No metadata files found")
            return pd.DataFrame()
        
        # Sort files by size for better load balancing
        all_metadata_files.sort(key=lambda x: x.stat().st_size)
        
        # Distribute files evenly across CPUs
        chunked_files = chunk_list(all_metadata_files, self.ncpus)
        
        # Log chunk distribution for monitoring
        total_size = sum(f.stat().st_size for f in all_metadata_files)
        self.logger.info(f"Total metadata size: {total_size / (1024**3):.2f} GB")
        
        for i, chunk in enumerate(chunked_files):
            chunk_size = sum(f.stat().st_size for f in chunk)
            chunk_size_gb = chunk_size / (1024**3)
            self.logger.info(f"Chunk {i+1}: {len(chunk)} files, {chunk_size_gb:.2f} GB ({chunk_size/total_size*100:.1f}%)")
        
        # Process in parallel with memory monitoring
        futures = []
        with ProcessPoolExecutor(max_workers=self.ncpus) as executor:
            for chunk in chunked_files:
                futures.append(executor.submit(process_metadata_chunk, chunk))
            
            # Collect results with progress tracking
            dfs = []
            for i, future in enumerate(futures):
                result = future.result()
                if not result.empty:
                    dfs.append(result)
                    self.logger.info(f"Task {i+1}/{len(futures)}: Loaded {len(result)} metadata rows")
                self._check_memory_usage(f"metadata_task_{i+1}")
        
        # Combine results efficiently
        if dfs:
            self.logger.info("Combining metadata DataFrames...")
            self.md = pd.concat(dfs, ignore_index=True, copy=False)
            dfs.clear()  # Free memory
            
            # Process the metadata
            self._process_metadata()
            
            self.logger.success(f"Processed {len(self.md)} metadata rows")
        else:
            self.md = pd.DataFrame()
            self.logger.warning("No metadata was loaded")
        
        self._check_memory_usage("load_metadata_complete")
        
        # Export if requested
        if export:
            self._export_metadata(format=format)
        
        return self.md
    
    def _process_metadata(self):
        """Process the loaded metadata efficiently."""
        if self.md.empty:
            return
        
        self.logger.info("Processing metadata...")
        
        # Extract frame_id - 2023 format has frame_id directly
        if 'frame_id' in self.md.columns:
            self.logger.info("Frame_id column already present in metadata")
        else:
            self.logger.error("No frame_id column found in metadata")
            return
        
        # Convert timestamp if present (2023 format uses 'captured_at')
        if 'captured_at' in self.md.columns:
            self.logger.info("Converting captured_at timestamp column...")
            if not pd.api.types.is_datetime64_any_dtype(self.md['captured_at']):
                # Convert from epoch milliseconds to datetime with timezone
                self.md['timestamp'] = pd.to_datetime(self.md['captured_at'], unit='ms')
                self.md['timestamp'] = self.md['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        elif 'timestamp' in self.md.columns:
            self.logger.info("Converting timestamp column...")
            if not pd.api.types.is_datetime64_any_dtype(self.md['timestamp']):
                # Convert from epoch milliseconds to datetime with timezone
                self.md['timestamp'] = pd.to_datetime(self.md['timestamp'], unit='ms')
                self.md['timestamp'] = self.md['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            self.logger.warning("No timestamp column found (neither 'captured_at' nor 'timestamp')")
        
        # Extract GPS coordinates for geospatial export (2023 format uses gps_info.longitude/latitude)
        if 'gps_info.longitude' in self.md.columns and 'gps_info.latitude' in self.md.columns:
            self.logger.info("Extracting GPS coordinates from gps_info columns...")
            self.md['lng'] = self.md['gps_info.longitude']
            self.md['lat'] = self.md['gps_info.latitude']
        elif 'longitude' in self.md.columns and 'latitude' in self.md.columns:
            self.logger.info("Extracting GPS coordinates from longitude/latitude columns...")
            self.md['lng'] = self.md['longitude']
            self.md['lat'] = self.md['latitude']
        else:
            self.logger.warning("No GPS coordinate columns found for geospatial export")
        
        self.logger.info("Metadata processing complete")
    
    def _export_image_paths(self, subset: Optional[List[Path]] = None, prefix: str = "") -> Optional[Path]:
        """Export image paths to a file efficiently."""
        imgs_to_export = subset if subset is not None else self.imgs
        
        if not imgs_to_export:
            self.logger.warning("No images to export")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_str = f"{prefix}_" if prefix else ""
        out_filepath = Path(EXPORT_DIR) / f"{prefix_str}nexar2023_images_{len(imgs_to_export)}_{timestamp}.txt"
        
        try:
            self.logger.info(f"Exporting {len(imgs_to_export)} image paths to {out_filepath}")
            
            # Write in chunks to avoid memory issues
            chunk_size = 10000
            with open(out_filepath, 'w') as f:
                for i in range(0, len(imgs_to_export), chunk_size):
                    chunk = imgs_to_export[i:i + chunk_size]
                    for img_path in chunk:
                        f.write(f"{str(img_path).replace(os.sep, '/')}\n")
            
            self.logger.success(f"Successfully exported image paths to {out_filepath}")
            return out_filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting image paths: {e}")
            return None
    
    def _export_metadata(self, subset: Optional[pd.DataFrame] = None, prefix: str = "", format: str = "parquet", free_memory: bool = False) -> Optional[Path]:
        """Export metadata DataFrame to a file efficiently."""
        md_to_export = subset if subset is not None else self.md
        
        if md_to_export.empty:
            self.logger.warning("No metadata to export")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_str = f"{prefix}_" if prefix else ""
        
        extension = ".parquet" if format.lower() == "parquet" else ".csv"
        out_filepath = Path(EXPORT_DIR) / f"{prefix_str}nexar2023_metadata_{len(md_to_export)}_{timestamp}{extension}"
        
        try:
            self.logger.info(f"Exporting {len(md_to_export)} metadata rows to {out_filepath}")
            
            if format.lower() == "parquet" and PARQUET_AVAILABLE:
                # Use PyArrow for efficient parquet export
                table = pa.Table.from_pandas(md_to_export)
                pq.write_table(table, out_filepath)
            else:
                # Fallback to CSV
                md_to_export.to_csv(out_filepath, index=False)
            
            self.logger.success(f"Successfully exported metadata to {out_filepath}")
            
            # Free memory if requested
            if free_memory:
                self.logger.info("Freeing metadata from memory")
                self.md = pd.DataFrame()
                import gc
                gc.collect()
            
            return out_filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting metadata: {e}")
            return None
    
    def _export_geospatial_metadata(self, subset: Optional[pd.DataFrame] = None, prefix: str = "", format: str = "geoparquet", 
                                    free_memory: bool = False, crs: str = "EPSG:4326") -> Optional[Path]:
        """Export metadata with GPS coordinates as a GeoParquet file.
        
        Args:
            subset: Specific subset of metadata to export, or None for all
            prefix: Prefix for the exported filename
            format: Export format ('geoparquet' or 'geojson')
            free_memory: Whether to delete the metadata from memory after export
            crs: Coordinate reference system (default: EPSG:4326/WGS84)
            
        Returns:
            Path to the exported file
        """
        # Ensure we have parquet support
        if not PARQUET_AVAILABLE:
            self.logger.error("PyArrow not installed. Cannot export GeoParquet.")
            return None
        
        # Create export directory if it doesn't exist
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        # Use subset or all metadata
        md_to_export = subset if subset is not None else self.md
        
        if md_to_export.empty:
            self.logger.warning("No metadata to export")
            return None
            
        # Check if we have coordinates. This needs to be abstracted to work for 2020 or 2023. Right now, would only work for 2020. 
        if 'lat' not in md_to_export.columns or 'lng' not in md_to_export.columns:
            self.logger.error("Metadata doesn't contain lat/lng columns needed for geospatial export")
            return None
        
        # Set filename for the export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_str = f"{prefix}_" if prefix else ""
        out_filepath = Path(EXPORT_DIR) / f"{prefix_str}nexar2023_geo_{len(md_to_export)}_{timestamp}.parquet"
        
        try:
            # Create a copy of the DataFrame
            geo_md = md_to_export.copy()
            
            # Skip rows with invalid coordinates
            geo_md = geo_md.dropna(subset=['lat', 'lng'])
            
            # Create WKT or GeoJSON geometry
            try:
                import geopandas as gpd
                from shapely.geometry import Point
                
                # Create Point geometry from lat and lng
                self.logger.info("Converting coordinates to geometry")
                geometry = [Point(lon, lat) for lon, lat in zip(geo_md['lng'], geo_md['lat'])]
                
                # Convert to GeoDataFrame
                geo_md = gpd.GeoDataFrame(geo_md, geometry=geometry, crs="EPSG:4326")
                
                # Export as GeoParquet
                self.logger.info(f"Exporting {len(geo_md)} geospatial metadata rows to {out_filepath}")
                geo_md.to_parquet(out_filepath)
                
            except ImportError:
                self.logger.error("GeoPandas not installed. Cannot export GeoParquet.")
                return None
                
            self.logger.success(f"Successfully exported geospatial metadata to {out_filepath}")
            
            # Free memory if requested
            if free_memory:
                self.logger.info("Freeing metadata from memory")
                self.md = pd.DataFrame()
                import gc
                gc.collect()
        except Exception as e:
            error_info = traceback.format_exc()
            self.logger.error(f"Error exporting geospatial metadata: {str(e)}\n{error_info}")
            return None
            
        return out_filepath
    
    def align_images_with_metadata(self) -> Tuple[List[Path], pd.DataFrame]:
        """Ensure perfect alignment between images and metadata efficiently."""
        if not self.imgs or self.md.empty:
            self.logger.warning("Cannot align: missing either images or metadata")
            return [], pd.DataFrame()
        
        self.logger.info(f"Aligning {len(self.imgs)} images with {len(self.md)} metadata rows")
        self._check_memory_usage("alignment_start")
        
        # Ensure frame_id is present in metadata
        if 'frame_id' not in self.md.columns:
            self.logger.error("No frame_id column found in metadata")
            return [], pd.DataFrame()
        
        # Create frame_id sets for efficient lookup
        self.logger.info("Creating frame_id index...")
        img_frame_ids = set(Path(img).stem for img in self.imgs)
        md_frame_ids = set(self.md['frame_id'])
        
        # Find common frame_ids
        common_frame_ids = img_frame_ids & md_frame_ids
        
        self.logger.info(f"Found {len(common_frame_ids)} common frame_ids")
        
        if not common_frame_ids:
            self.logger.error("No matching frame_ids found between images and metadata")
            return [], pd.DataFrame()
        
        # Filter metadata to common frame_ids
        self.logger.info("Filtering metadata...")
        self.md = self.md[self.md['frame_id'].isin(common_frame_ids)].copy()
        
        # Filter images to common frame_ids (using temporary file for memory efficiency)
        self.logger.info("Filtering images...")
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            with open(temp_path, 'w') as f:
                for img in self.imgs:
                    if Path(img).stem in common_frame_ids:
                        f.write(f"{img}\n")
            
            # Read back filtered images
            with open(temp_path, 'r') as f:
                filtered_imgs = [Path(line.strip()) for line in f]
            
            # Sort both to ensure alignment
            self.logger.info("Sorting for alignment...")
            self.md = self.md.sort_values('frame_id').reset_index(drop=True)
            filtered_imgs.sort(key=lambda x: x.stem)
            
            # Update dataset
            self.imgs = filtered_imgs
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        self.logger.success(f"Successfully aligned {len(self.imgs)} images with metadata")
        self._check_memory_usage("alignment_complete")
        
        return self.imgs, self.md
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics efficiently."""
        stats = {
            "dataset_type": "OptimizedNexar2023Dataset",
            "images": {"count": len(self.imgs)},
            "metadata": {"count": len(self.md)}
        }
        
        if not self.md.empty and 'timestamp' in self.md.columns:
            stats["metadata"]["date_range"] = {
                "start": self.md['timestamp'].min().strftime("%Y-%m-%d"),
                "end": self.md['timestamp'].max().strftime("%Y-%m-%d")
            }
        
        return stats
    
    def filter_by_date(self, start_date=None, end_date=None, export=False, format="parquet"):
        """Filter the dataset by date range."""
        self.logger.info("Filtering dataset by date range...")
        self._check_memory_usage("Before filtering")
        
        if self.md.empty:
            self.logger.warning("No metadata available for filtering. Load metadata first.")
            return self.imgs, self.md
        
        if not start_date and not end_date:
            self.logger.info("No date range specified, returning original dataset")
            return self.imgs, self.md
        
        start = pd.to_datetime(start_date) if start_date else pd.Timestamp.min
        end = pd.to_datetime(end_date) if end_date else pd.Timestamp.max
        
        self.logger.info(f"Filtering dataset by date range: {start.date()} to {end.date()}")
        
        # Ensure timestamp column is properly converted to datetime
        if 'timestamp' in self.md.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.md['timestamp']):
                self.logger.info("Converting timestamp column to datetime format")
                self.md['timestamp'] = pd.to_datetime(self.md['timestamp'], unit='ms')
                self.md['timestamp'] = self.md['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            self.logger.error("No timestamp column found in metadata. Cannot filter by date.")
            return self.imgs, self.md
        
        # Create date column for faster filtering
        self.md['date'] = self.md['timestamp'].dt.date
        
        # Filter using the pre-computed column (much faster)
        filtered_md = self.md[(self.md['date'] >= start.date()) & 
                           (self.md['date'] <= end.date())]
        
        self._check_memory_usage("After metadata filtering")
        
        # Filter images by frame_id using a hash-based approach
        filtered_imgs = []
        if self.imgs:
            # Create a hash set for O(1) lookups
            self.logger.info("Creating optimized hash set for frame_id matching")
            filtered_frame_ids = set(filtered_md['frame_id'])
            
            if len(self.imgs) > 10000:
                # Use batch processing to improve memory usage
                self.logger.info(f"Using optimized batch processing to filter {len(self.imgs)} images")
                batch_size = 50000  # Process images in batches of 50k
                filtered_imgs = []
                
                for i in range(0, len(self.imgs), batch_size):
                    batch = self.imgs[i:i+batch_size]
                    # Process this batch
                    batch_results = [img for img in batch if Path(img).stem in filtered_frame_ids]
                    filtered_imgs.extend(batch_results)
                    
                    if i % 100000 == 0 and i > 0:
                        self.logger.info(f"Processed {i} of {len(self.imgs)} images...")
            else:
                # Standard filtering for smaller lists using the hash set
                filtered_imgs = [img for img in self.imgs if Path(img).stem in filtered_frame_ids]
        
        self._check_memory_usage("After image filtering")
        
        self.logger.success(f"Filtered dataset contains {len(filtered_imgs)} images and {len(filtered_md)} metadata rows")
        
        # Export if requested
        if export:
            date_range = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
            
            if filtered_imgs:
                self._export_image_paths(subset=filtered_imgs, prefix=f"filtered_{date_range}")
            
            if not filtered_md.empty:
                self._export_metadata(subset=filtered_md, prefix=f"filtered_{date_range}", format=format)
        
        return filtered_imgs, filtered_md

    def filter_by_specific_dates(self, dates, export=False, format="parquet"):
        """Filter the dataset to include only records from specific dates."""
        self.logger.info("Filtering dataset by specific dates...")
        self._check_memory_usage("Before filtering")
        
        if self.md.empty:
            self.logger.warning("No metadata available for filtering. Load metadata first.")
            return self.imgs, self.md
        
        if not dates:
            self.logger.info("No dates specified, returning original dataset")
            return self.imgs, self.md
        
        # Parse dates
        self.logger.info(f"Parsing dates: {dates}")
        parsed_dates = []
        for date_str in dates:
            try:
                # Handle both date objects and strings
                if isinstance(date_str, (datetime, pd.Timestamp)):
                    parsed_dates.append(pd.Timestamp(date_str).date())
                else:
                    # Try different formats with explicit format matching
                    date_formats = ['%Y-%m-%d', '%m-%d-%Y', '%m/%d/%Y', '%d-%m-%Y']
                    for fmt in date_formats:
                        try:
                            parsed_date = pd.to_datetime(date_str, format=fmt).date()
                            parsed_dates.append(parsed_date)
                            self.logger.info(f"Parsed date '{date_str}' as {parsed_date} using format {fmt}")
                            break
                        except:
                            continue
                    else:
                        # Try a flexible parsing as last resort
                        try:
                            parsed_date = pd.to_datetime(date_str).date()
                            parsed_dates.append(parsed_date)
                            self.logger.info(f"Parsed date '{date_str}' as {parsed_date} using flexible parsing")
                        except:
                            self.logger.warning(f"Couldn't parse date: {date_str}, skipping")
            except Exception as e:
                self.logger.warning(f"Error parsing date {date_str}: {e}")
        
        if not parsed_dates:
            self.logger.warning("No valid dates found in the provided list")
            return self.imgs, self.md
        
        self.logger.info(f"Successfully parsed {len(parsed_dates)} dates: {', '.join(str(d) for d in parsed_dates)}")
        
        # Ensure timestamp column is properly converted to datetime
        if 'timestamp' in self.md.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.md['timestamp']):
                self.logger.info("Converting timestamp column to datetime format")
                self.md['timestamp'] = pd.to_datetime(self.md['timestamp'], unit='ms')
                self.md['timestamp'] = self.md['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            self.logger.error("No timestamp column found in metadata. Cannot filter by date.")
            return self.imgs, self.md
        
        # Create date column for faster filtering if it doesn't exist
        if 'date' not in self.md.columns:
            self.logger.info("Creating date column for filtering")
            self.md['date'] = self.md['timestamp'].dt.date
        
        # Filter metadata to include only records from the specified dates
        filtered_md = self.md[self.md['date'].isin(parsed_dates)]
        
        self._check_memory_usage("After metadata filtering")
        
        # Filter images by frame_id
        filtered_imgs = []
        if self.imgs:
            filtered_frame_ids = set(filtered_md['frame_id'])
            self.logger.info(f"Filtering {len(self.imgs)} images to match {len(filtered_frame_ids)} frame_ids")
            
            # Use batch processing for large image lists
            batch_size = 1000000
            if len(self.imgs) > batch_size:
                filtered_imgs = []
                total_processed = 0
                total_matches = 0
                
                for i in range(0, len(self.imgs), batch_size):
                    end_idx = min(i + batch_size, len(self.imgs))
                    batch = self.imgs[i:end_idx]
                    batch_filtered = [img for img in batch if Path(img).stem in filtered_frame_ids]
                    filtered_imgs.extend(batch_filtered)
                    
                    total_processed += len(batch)
                    total_matches += len(batch_filtered)
                    
                    if i % (5 * batch_size) == 0 and i > 0:
                        self.logger.info(f"Processed {total_processed} of {len(self.imgs)} images, found {total_matches} matches so far")
                        # Force garbage collection to free memory
                        import gc
                        gc.collect()
            else:
                filtered_imgs = [img for img in self.imgs if Path(img).stem in filtered_frame_ids]
        
        self._check_memory_usage("After image filtering")
        self.logger.success(f"Filtered dataset contains {len(filtered_imgs)} images and {len(filtered_md)} metadata rows")
        
        # Export if requested
        if export and (filtered_imgs or not filtered_md.empty):
            date_str = "specific_dates"
            if len(parsed_dates) <= 3:
                # Use abbreviated format for small number of dates
                date_str = "_".join([d.strftime('%Y%m%d') for d in parsed_dates])
            
            if filtered_imgs:
                self._export_image_paths(subset=filtered_imgs, prefix=f"filtered_{date_str}")
            
            if not filtered_md.empty:
                self._export_metadata(subset=filtered_md, prefix=f"filtered_{date_str}", format=format)
        
        return filtered_imgs, filtered_md


def load_nexar_dataset_optimized(
    imgs: bool = True, 
    md: bool = True, 
    ncpus: int = 8, 
    memory_limit_gb: float = 50.0,
    export: bool = True, 
    format: str = "parquet",
    align: bool = True,
    export_geoparquet: bool = True,
    start_date: str = None,
    end_date: str = None,
    dates: List[str] = None
) -> OptimizedNexar2023Dataset:
    """Load a Nexar 2023 dataset with optimized performance and memory usage.
    
    Args:
        imgs: Whether to load images
        md: Whether to load metadata
        ncpus: Number of CPUs to use for parallel processing
        memory_limit_gb: Memory limit in GB to prevent OOM
        export: Whether to export the dataset
        format: Export format for metadata ('parquet' or 'csv')
        align: Whether to align images with metadata
        export_geoparquet: Whether to export geospatial metadata as GeoParquet
        start_date: Start date for range filtering (YYYY-MM-DD)
        end_date: End date for range filtering (YYYY-MM-DD)
        dates: List of specific dates to filter for (overrides start_date/end_date if provided)
    
    Returns:
        Loaded dataset
    """
    logger = get_logger("load_dataset_optimized")
    logger.info(f"Loading Nexar 2023 dataset with {ncpus} CPUs and {memory_limit_gb}GB memory limit...")
    
    # Create dataset
    dataset = OptimizedNexar2023Dataset(ncpus=ncpus, memory_limit_gb=memory_limit_gb)
    
    # Load images if requested
    if imgs:
        dataset.load_images(export=False)
    
    # Load metadata if requested
    if md:
        dataset.load_metadata(export=False, format=format)
    
    # Align if requested and both are loaded
    if align and imgs and md and dataset.imgs and not dataset.md.empty:
        dataset.align_images_with_metadata()
    
    # Apply date filtering if specified
    if md and not dataset.md.empty:
        if dates:  # Specific dates filtering takes precedence
            logger.info(f"Filtering dataset by {len(dates)} specific dates")
            filtered_imgs, filtered_md = dataset.filter_by_specific_dates(
                dates, export=False, format=format
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
    
    # Show statistics
    stats = dataset.get_statistics()
    logger.info(f"Dataset statistics: {stats}")
    
    # Export if requested
    if export:
        if imgs and dataset.imgs:
            dataset._export_image_paths()
        if md and not dataset.md.empty:
            # Standard metadata export
            dataset._export_metadata(format=format, free_memory=False)  # Don't free memory yet
            
            # Also export as GeoParquet if requested
            if export_geoparquet:
                dataset._export_geospatial_metadata(format="geoparquet", free_memory=False)
            else:
                # If we didn't export geoparquet but still want to free memory
                import gc
                dataset.md = pd.DataFrame()
                gc.collect()
    
    logger.success(f"Dataset loaded: {len(dataset.imgs)} images and {len(dataset.md)} metadata rows")
    
    return dataset


if __name__ == "__main__":
    # Expose simplified CLI
    fire.Fire(load_nexar_dataset_optimized) 