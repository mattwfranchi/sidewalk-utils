import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import traceback
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

from utils.logger import get_logger

# Check for PyArrow for parquet support
try:
    import pyarrow
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
    # Check if geoparquet is available
    try:
        import pyarrow.parquet.geoparquet as gpq
    except ImportError:
        pass
except ImportError:
    PARQUET_AVAILABLE = False

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
    pandarallel.initialize(progress_bar=False, verbose=0)
except ImportError:
    PANDARALLEL_AVAILABLE = False

# Export directory
EXPORT_DIR = "/share/ju/sidewalk_utils/data/raw_export"

class BaseNexarDataset(ABC):
    """Base class for Nexar datasets."""
    
    def __init__(self, year="2020", load_imgs=True, load_md=True, ncpus=8):
        """Initialize the dataset.
        
        Args:
            year: Dataset year ('2020' or '2023')
            load_imgs: Whether to load images on init
            load_md: Whether to load metadata on init
            ncpus: Number of CPU cores to use for parallel processing
        """
        self.year = year
        self.imgs = []
        self.md = pd.DataFrame()
        self.ncpus = ncpus
        self.logger = get_logger(f"nexar{year}")
        
        # Create export directory if it doesn't exist
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        # Load data if requested
        if load_imgs:
            self.load_images()
            
        if load_md:
            self.load_metadata()
    
    @abstractmethod
    def load_images(self, export=False):
        """Load images from the dataset."""
        pass
    
    @abstractmethod
    def load_metadata(self, export=False):
        """Load metadata from the dataset."""
        pass
    
    def filter_by_date(self, start_date=None, end_date=None, export=False, format="parquet"):
        """Filter the dataset by date range."""
        logger = self.logger
        self.log_memory_usage("Before filtering")
        if self.md.empty:
            logger.warning("No metadata available for filtering. Load metadata first.")
            return self.imgs, self.md
        if not start_date and not end_date:
            logger.info("No date range specified, returning original dataset")
            return self.imgs, self.md
        start = pd.to_datetime(start_date) if start_date else pd.Timestamp.min
        end = pd.to_datetime(end_date) if end_date else pd.Timestamp.max
        logger.info(f"Filtering dataset by date range: {start.date()} to {end.date()}")
        if 'timestamp' in self.md.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.md['timestamp']):
                self.md['timestamp'] = pd.to_datetime(self.md['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            logger.error("No timestamp column found in metadata. Cannot filter by date.")
            return self.imgs, self.md
        self.md['date'] = self.md['timestamp'].dt.date
        filtered_md = self.md[(self.md['date'] >= start.date()) & (self.md['date'] <= end.date())]
        filtered_imgs = [img for img in self.imgs if Path(img).stem in set(filtered_md['frame_id'])]
        self.log_memory_usage("After filtering")
        if export:
            self._export_image_paths(subset=filtered_imgs, prefix=f"filtered_{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}")
            self._export_metadata(subset=filtered_md, format=format)
        return filtered_imgs, filtered_md

    def filter_by_specific_dates(self, dates, export=False, format="parquet"):
        """Filter the dataset to include only records from specific dates."""
        logger = self.logger
        self.log_memory_usage("Before filtering")
        if self.md.empty:
            logger.warning("No metadata available for filtering. Load metadata first.")
            return self.imgs, self.md
        if not dates:
            logger.info("No dates specified, returning original dataset")
            return self.imgs, self.md
        
        # Improved date parsing with better logging
        logger.info(f"Parsing dates: {dates}")
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
                            logger.info(f"Parsed date '{date_str}' as {parsed_date} using format {fmt}")
                            break
                        except:
                            continue
                    else:
                        # Try a flexible parsing as last resort
                        try:
                            parsed_date = pd.to_datetime(date_str).date()
                            parsed_dates.append(parsed_date)
                            logger.info(f"Parsed date '{date_str}' as {parsed_date} using flexible parsing")
                        except:
                            logger.warning(f"Couldn't parse date: {date_str}, skipping")
            except Exception as e:
                logger.warning(f"Error parsing date {date_str}: {e}")
        
        if not parsed_dates:
            logger.warning("No valid dates found in the provided list")
            return self.imgs, self.md
        
        logger.info(f"Successfully parsed {len(parsed_dates)} dates: {', '.join(str(d) for d in parsed_dates)}")
        
        # Ensure timestamp column is properly converted to datetime
        if 'timestamp' in self.md.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.md['timestamp']):
                logger.info("Converting timestamp column to datetime format")
                self.md['timestamp'] = pd.to_datetime(self.md['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            logger.error("No timestamp column found in metadata. Cannot filter by date.")
            return self.imgs, self.md
        
        # Process filtering in batches for better memory efficiency
        logger.info(f"Filtering {len(self.md)} metadata rows for the specified dates")
        
        # Create date column for faster filtering if it doesn't exist
        if 'date' not in self.md.columns:
            logger.info("Creating date column for filtering")
            self.md['date'] = self.md['timestamp'].dt.date
        
        # Filter metadata in batches to reduce memory usage
        batch_size = 1000000  # Process 1M rows at a time for large datasets
        if len(self.md) > batch_size:
            logger.info(f"Using batch processing for filtering {len(self.md)} rows")
            filtered_dfs = []
            total_processed = 0
            total_matches = 0
            
            for i in range(0, len(self.md), batch_size):
                end_idx = min(i + batch_size, len(self.md))
                batch = self.md.iloc[i:end_idx]
                # Convert batch dates to string representation for faster comparison
                batch_filtered = batch[batch['date'].isin(parsed_dates)]
                
                if not batch_filtered.empty:
                    filtered_dfs.append(batch_filtered)
                    total_matches += len(batch_filtered)
                
                total_processed += len(batch)
                if i % (5 * batch_size) == 0 and i > 0:
                    logger.info(f"Processed {total_processed} of {len(self.md)} rows, found {total_matches} matches so far")
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
            
            logger.info(f"Combining {len(filtered_dfs)} filtered batches")
            if filtered_dfs:
                filtered_md = pd.concat(filtered_dfs, ignore_index=True)
                filtered_dfs.clear()  # Free memory
            else:
                filtered_md = pd.DataFrame()
        else:
            # For smaller datasets, we can filter all at once
            filtered_md = self.md[self.md['date'].isin(parsed_dates)]
        
        logger.info(f"Found {len(filtered_md)} metadata rows matching the specified dates")
        
        # Optimize image filtering for large datasets
        if self.imgs:
            filtered_frame_ids = set(filtered_md['frame_id'])
            logger.info(f"Filtering {len(self.imgs)} images to match {len(filtered_frame_ids)} frame_ids")
            
            # For large image lists, use batched processing to reduce memory pressure
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
                        logger.info(f"Processed {total_processed} of {len(self.imgs)} images, found {total_matches} matches so far")
                        # Force garbage collection to free memory
                        import gc
                        gc.collect()
            else:
                filtered_imgs = [img for img in self.imgs if Path(img).stem in filtered_frame_ids]
        else:
            filtered_imgs = []
        
        self.log_memory_usage("After filtering")
        logger.success(f"Filtered dataset contains {len(filtered_imgs)} images and {len(filtered_md)} metadata rows")
        
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

    def export_statistics(self, output_file=None):
        """Generate and export statistics about the dataset."""
        stats = {
            "dataset_type": self.__class__.__name__,
            "images": {"count": len(self.imgs)},
            "metadata": {"count": len(self.md)}
        }
        if not self.md.empty:
            stats["metadata"]["date_range"] = {
                "start": self.md['timestamp'].min().strftime("%Y-%m-%d"),
                "end": self.md['timestamp'].max().strftime("%Y-%m-%d")
            }
        if output_file:
            with open(output_file, 'w') as f:
                f.write(str(stats))
        else:
            print(stats)
        return stats

    def split_into_chunks(self, num_chunks=2, export=True, format="parquet", prefix="chunk", free_memory=True):
        """Split the dataset into evenly-sized chunks."""
        total_items = len(self.imgs)
        if total_items == 0:
            self.logger.warning("Dataset is empty, cannot split")
            return []
        base_chunk_size = total_items // num_chunks
        remainder = total_items % num_chunks
        chunks = []
        for i in range(num_chunks):
            chunk_size = base_chunk_size + (1 if i < remainder else 0)
            chunk_imgs = self.imgs[i * chunk_size:(i + 1) * chunk_size]
            chunk_md = self.md.iloc[i * chunk_size:(i + 1) * chunk_size]
            if export:
                self._export_image_paths(subset=chunk_imgs, prefix=f"{prefix}_{i+1}")
                self._export_metadata(subset=chunk_md, prefix=f"{prefix}_{i+1}", format=format)
            chunks.append((chunk_imgs, chunk_md))
        return chunks

    def validate_alignment(self, fix_misalignment=True):
        """Validate and optionally fix alignment between images and metadata."""
        img_frame_ids = [Path(img).stem for img in self.imgs]
        md_frame_ids = self.md['frame_id'].tolist()
        if len(img_frame_ids) != len(md_frame_ids):
            self.logger.warning("Mismatched lengths between images and metadata")
            if fix_misalignment:
                self.align_images_with_metadata()
        return len(self.imgs) == len(self.md)

    def log_memory_usage(self, operation_name=""):
        """Log the current memory usage."""
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil is not available. Memory usage cannot be logged.")
            return
        memory = self.get_memory_usage()
        self.logger.info(f"[{operation_name}] Memory usage: {memory['rss_mb']:.1f} MB (RSS)")

    def get_memory_usage(self, detailed=False):
        """Get current memory usage information."""
        if not PSUTIL_AVAILABLE:
            return {"available": False, "message": "psutil not installed"}
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024)
        }
    
    def _chunk_list(self, lst, num_chunks):
        """Split a list into roughly equal chunks.
        
        Args:
            lst: The list to split
            num_chunks: Number of chunks to create
            
        Returns:
            List of chunks (each chunk is a list)
        """
        if not lst:
            return []
            
        avg_chunk_size = max(1, len(lst) // num_chunks)
        chunks = []
        
        for i in range(0, len(lst), avg_chunk_size):
            chunks.append(lst[i:i + avg_chunk_size])
        
        return chunks
    
    def _export_image_paths(self, subset=None, prefix="", format="txt", free_memory=False):
        """Export image paths to a file in EXPORT_DIR.
        
        Args:
            subset: Specific subset of images to export, or None for all
            prefix: Prefix for the exported filename
            format: Export format ('txt', 'csv', or 'json')
            free_memory: Whether to delete the images from memory after export
        
        Returns:
            Path to the exported file
        """
        # Create export directory if it doesn't exist
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        # Use subset or all images
        imgs_to_export = subset if subset is not None else self.imgs
        
        if not imgs_to_export:
            self.logger.warning("No images to export")
            return None
            
        # Set filename for the export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_str = f"{prefix}_" if prefix else ""
        out_filepath = Path(EXPORT_DIR) / f"{prefix_str}nexar{self.year}_images_{len(imgs_to_export)}_{timestamp}.txt"
        
        try:
            # Write image paths to file
            self.logger.info(f"Exporting {len(imgs_to_export)} image paths to {out_filepath}")
            with open(out_filepath, 'w') as f:
                for img_path in imgs_to_export:
                    # Ensure paths are strings with forward slashes
                    f.write(f"{str(img_path).replace(os.sep, '/')}\n")
                    
            self.logger.success(f"Successfully exported image paths to {out_filepath}")
            
            # Free memory if requested
            if free_memory:
                self.logger.info("Freeing image list from memory")
                self.imgs = []
                import gc
                gc.collect()
        except Exception as e:
            error_info = traceback.format_exc()
            self.logger.error(f"Error exporting image paths: {str(e)}\n{error_info}")
            return None
            
        return out_filepath
    
    def _export_metadata(self, subset=None, prefix="", format="parquet", free_memory=False):
        """Export metadata DataFrame to a file in EXPORT_DIR.
        
        Args:
            subset: Specific subset of metadata to export, or None for all
            prefix: Prefix for the exported filename
            format: Export format ('parquet' or 'csv')
            free_memory: Whether to delete the metadata from memory after export
            
        Returns:
            Path to the exported file
        """
        # Create export directory if it doesn't exist
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        # Use subset or all metadata
        md_to_export = subset if subset is not None else self.md
        
        if md_to_export.empty:
            self.logger.warning("No metadata to export")
            return None
            
        # Set filename for the export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_str = f"{prefix}_" if prefix else ""
        
        # Add extension based on format
        extension = ".parquet" if format.lower() == "parquet" else ".csv"
        out_filepath = Path(EXPORT_DIR) / f"{prefix_str}nexar{self.year}_metadata_{len(md_to_export)}_{timestamp}{extension}"
        
        try:
            self.logger.info(f"Exporting {len(md_to_export)} metadata rows to {out_filepath}")
            
            # Export based on format
            if format.lower() == "parquet":
                if not PARQUET_AVAILABLE:
                    self.logger.warning("PyArrow not installed. Falling back to CSV export.")
                    format = "csv"
                    out_filepath = out_filepath.with_suffix(".csv")
                else:
                    # Use PyArrow for Parquet export
                    table = pyarrow.Table.from_pandas(md_to_export)
                    pq.write_table(table, out_filepath)
            
            # If not parquet or fallback to CSV
            if format.lower() != "parquet" or not PARQUET_AVAILABLE:
                md_to_export.to_csv(out_filepath, index=False)
                    
            self.logger.success(f"Successfully exported metadata to {out_filepath}")
            
            # Free memory if requested
            if free_memory:
                self.logger.info("Freeing metadata from memory")
                self.md = pd.DataFrame()
                import gc
                gc.collect()
        except Exception as e:
            error_info = traceback.format_exc()
            self.logger.error(f"Error exporting metadata: {str(e)}\n{error_info}")
            return None
            
        return out_filepath
    
    def _export_geospatial_metadata(self, subset=None, prefix="", format="geoparquet", 
                                    free_memory=False, crs="EPSG:4326"):
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
        out_filepath = Path(EXPORT_DIR) / f"{prefix_str}nexar{self.year}_geo_{len(md_to_export)}_{timestamp}.parquet"
        
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
    
    def align_images_with_metadata(self):
        """Ensure perfect alignment between images and metadata rows.
        
        This method aligns the dataset's images and metadata based on frame_id,
        ensuring that for every image there is a corresponding metadata row
        and vice versa. Uses memory-efficient batch processing.
        
        Returns:
            Tuple of (aligned_imgs, aligned_md)
        """
        if not self.imgs or self.md.empty:
            self.logger.warning("Cannot align: missing either images or metadata")
            return [], pd.DataFrame()
        
        self.logger.info(f"Aligning {len(self.imgs)} images with {len(self.md)} metadata rows")
        
        # First, ensure that frame_id is present in metadata
        if 'frame_id' not in self.md.columns:
            self.logger.info("Computing frame_id from image_ref")
            # Extract frame_id from image_ref
            self.md['frame_id'] = self.md['image_ref'].apply(lambda x: Path(str(x)).stem)
        
        # MEMORY OPTIMIZATION: Process alignment in batches
        start_time = time.time()
        
        # Create temporary file to store aligned image paths
        import tempfile
        temp_img_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
        temp_img_path = temp_img_file.name
        temp_img_file.close()  # Close but don't delete
        
        # Create temporary file to store frame_ids (for later metadata filtering)
        temp_frameids_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
        temp_frameids_path = temp_frameids_file.name
        temp_frameids_file.close()  # Close but don't delete

        self.logger.info(f"Using temporary files for memory-efficient processing")
        
        # Process images in batches to build frame_id index without keeping everything in memory
        self.logger.info("Building frame_id index in batches")
        batch_size = 500000  # Process 500k images at a time
        img_frame_id_set = set()
        total_aligned = 0
        
        # First pass: collect frame_ids from images
        for i in range(0, len(self.imgs), batch_size):
            batch_end = min(i + batch_size, len(self.imgs))
            batch = self.imgs[i:batch_end]
            
            # Process this batch
            for img_path in batch:
                frame_id = Path(img_path).stem
                img_frame_id_set.add(frame_id)
                
            if i % 1000000 == 0 and i > 0:
                self.logger.info(f"Processed {i} of {len(self.imgs)} images for frame_id extraction")
                # Force garbage collection to free memory
                import gc
                gc.collect()
        
        # Filter metadata to match available images (in batches)
        self.logger.info(f"Filtering metadata to match {len(img_frame_id_set)} image frame_ids")
        
        # MEMORY OPTIMIZATION: Filter metadata in batches
        md_batch_size = 1000000  # Process 1M rows at a time
        total_matched_md = 0
        matched_dfs = []
        
        for i in range(0, len(self.md), md_batch_size):
            batch_end = min(i + md_batch_size, len(self.md))
            md_batch = self.md.iloc[i:batch_end]
            
            # Filter this batch for matching frame_ids
            matched_md_batch = md_batch[md_batch['frame_id'].isin(img_frame_id_set)]
            if not matched_md_batch.empty:
                matched_dfs.append(matched_md_batch)
                total_matched_md += len(matched_md_batch)
                
                # Write matching frame_ids to temp file for later use
                with open(temp_frameids_path, 'a') as f:
                    for frame_id in matched_md_batch['frame_id']:
                        f.write(f"{frame_id}\n")
            
            if i % 5000000 == 0 and i > 0:
                self.logger.info(f"Processed {i} of {len(self.md)} metadata rows, matched {total_matched_md} so far")
                # Intermediate garbage collection
                import gc
                gc.collect()
        
        # Combine filtered metadata batches
        self.logger.info(f"Combining {len(matched_dfs)} filtered metadata batches")
        if matched_dfs:
            filtered_md = pd.concat(matched_dfs, ignore_index=True)
            # Clear memory
            matched_dfs.clear()
        else:
            filtered_md = pd.DataFrame()
        
        # Force garbage collection before next phase
        import gc
        gc.collect()
        
        if len(filtered_md) == 0:
            self.logger.error("No metadata matches the available images! Check path structures.")
            # Clean up temp files
            os.unlink(temp_img_path)
            os.unlink(temp_frameids_path)
            return [], pd.DataFrame()
        
        # Get the set of frame_ids that have matching metadata
        self.logger.info("Loading frame_ids from temporary file")
        md_frame_ids = set()
        with open(temp_frameids_path, 'r') as f:
            for line in f:
                md_frame_ids.add(line.strip())
        
        # Filter images to match metadata
        self.logger.info(f"Filtering {len(self.imgs)} images to match {len(md_frame_ids)} metadata frame_ids")
        total_matched_img = 0
        
        # Process images in batches
        with open(temp_img_path, 'w') as f:
            for i in range(0, len(self.imgs), batch_size):
                batch_end = min(i + batch_size, len(self.imgs))
                batch = self.imgs[i:batch_end]
                
                # Filter this batch
                for img_path in batch:
                    frame_id = Path(img_path).stem
                    if frame_id in md_frame_ids:
                        f.write(f"{img_path}\n")
                        total_matched_img += 1
                
                if i % 1000000 == 0 and i > 0:
                    self.logger.info(f"Processed {i} of {len(self.imgs)} images, matched {total_matched_img} so far")
                    # Intermediate garbage collection
                    import gc
                    gc.collect()
        
        # Load the aligned image paths
        self.logger.info(f"Loading {total_matched_img} aligned image paths from temporary file")
        aligned_imgs = []
        with open(temp_img_path, 'r') as f:
            for line in f:
                aligned_imgs.append(Path(line.strip()))
        
        # Update dataset attributes
        self.imgs = aligned_imgs
        self.md = filtered_md
        
        # Clean up temporary files
        os.unlink(temp_img_path)
        os.unlink(temp_frameids_path)
        
        elapsed = time.time() - start_time
        
        # Final verification
        if len(self.imgs) != len(self.md):
            self.logger.warning(f"After alignment: {len(self.imgs)} images and {len(self.md)} metadata rows")
            
            # ADDITIONAL MEMORY-EFFICIENT ALIGNMENT STEP
            # If counts don't match, perform an additional step to ensure perfect alignment
            if len(self.imgs) > 0 and len(self.md) > 0:
                self.logger.info("Performing final alignment step")
                
                # Create a mapping from frame_id to index
                img_frame_ids = [Path(img).stem for img in self.imgs]
                img_map = {frame_id: i for i, frame_id in enumerate(img_frame_ids)}
                
                # Get matching metadata rows in same order as images
                md_frame_ids = self.md['frame_id'].tolist()
                md_indices = []
                img_indices = []
                
                # Find matching pairs
                for i, md_frame_id in enumerate(md_frame_ids):
                    if md_frame_id in img_map:
                        md_indices.append(i)
                        img_indices.append(img_map[md_frame_id])
                
                # Filter to keep only matching rows
                self.md = self.md.iloc[md_indices].reset_index(drop=True)
                self.imgs = [self.imgs[i] for i in img_indices]
                
                self.logger.success(f"Final alignment complete: {len(self.imgs)} images and {len(self.md)} metadata rows")
        else:
            self.logger.success(f"Successfully aligned {len(self.imgs)} images with metadata in {elapsed:.2f} seconds")
        
        return self.imgs, self.md

    def _process_dir_chunk(self, dir_chunk):
        """Process a chunk of directories and return all image paths."""
        all_images = []
        try:
            for img_dir in dir_chunk:
                try:
                    images = self.load_img_dir(img_dir)
                    all_images.extend(images)
                except Exception as e:
                    error_info = traceback.format_exc()
                    self.logger.error(f"Error processing directory {img_dir}: {str(e)}\n{error_info}")
        except Exception as e:
            error_info = traceback.format_exc()
            self.logger.error(f"Unexpected error in worker process: {str(e)}\n{error_info}")
        return all_images

    def _process_md_chunk(self, file_chunk):
        """Process a chunk of metadata CSV files and return combined DataFrame."""
        dfs = []
        try:
            for file_path in file_chunk:
                try:
                    sample = pd.read_csv(file_path, nrows=1000)
                    dtypes = {col: 'category' if sample[col].dtype == 'object' else sample[col].dtype for col in sample}
                    df = pd.read_csv(file_path, dtype=dtypes)
                    dfs.append(df)
                except Exception as e:
                    error_info = traceback.format_exc()
                    self.logger.error(f"Error reading {file_path}: {str(e)}\n{error_info}")
        except Exception as e:
            error_info = traceback.format_exc()
            self.logger.error(f"Unexpected error in worker process: {str(e)}\n{error_info}")
        return pd.concat(dfs, ignore_index=True, copy=False) if dfs else pd.DataFrame()

    def _process_metadata_results(self, dfs, export=False, format="parquet"):
        """Common processing for metadata results."""
        if dfs:
            self.logger.info("Concatenating metadata DataFrames")
            # Fix the FutureWarning by handling empty DataFrames explicitly
            if any(not df.empty for df in dfs):
                self.md = pd.concat(dfs, ignore_index=True, copy=False)
                self.md['frame_id'] = self.md['image_ref'].apply(lambda x: Path(x).stem)
                self.md['timestamp'] = pd.to_datetime(self.md['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                self.logger.success(f"Processed {len(self.md)} metadata rows")
                if export:
                    self._export_metadata(format=format)
            else:
                self.md = pd.DataFrame()
                self.logger.warning("All metadata DataFrames were empty")
        else:
            self.logger.warning("No metadata files were loaded")
            self.md = pd.DataFrame()

    def _load_metadata_parallel(self, metadata_path, export=False, format="parquet"):
        """Load metadata using parallel processing."""
        metadata_files = list(Path(metadata_path).glob("*.csv"))
        metadata_files.sort(key=lambda x: x.stat().st_size)
        chunked_files = self._chunk_list(metadata_files, self.ncpus)
        futures = []
        with ProcessPoolExecutor(max_workers=self.ncpus) as executor:
            for chunk in chunked_files:
                futures.append(executor.submit(self._process_md_chunk, chunk))
            dfs = [future.result() for future in futures]
        self._process_metadata_results(dfs, export, format)