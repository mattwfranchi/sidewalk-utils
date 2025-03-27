import os
import geopandas as gpd
import warnings
import numpy as np
from utils.logger import get_logger
from data.nyc.c import PROJ_FT

class GeoDataProcessor:
    """Base class for processing geospatial data with common utilities."""

    def __init__(self, name=None):
        """Initialize the processor with its own logger."""
        module_name = name or __name__
        self.logger = get_logger(module_name)

    def read_geodataframe(self, file_path, crs=None, name="dataset"):
        """
        Read geodataframe from file with error handling
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Input file not found: {file_path}")
                return None
                
            self.logger.info(f"Loading {name} from {file_path}")
            
            # Add support for parquet files
            if file_path.lower().endswith('.parquet'):
                self.logger.info(f"Detected Parquet format, using read_parquet()")
                df = gpd.read_parquet(file_path)
            else:
                df = gpd.read_file(file_path)
            
            if df.empty:
                self.logger.error(f"Input file contains no data")
                return None
                
            self.logger.info(f"Loaded {len(df)} features")
            
            # Convert to specified CRS if provided
            if crs is not None:
                try:
                    if df.crs is None:
                        self.logger.warning(f"Input file has no CRS specified, assuming {crs}")
                        df.set_crs(crs, inplace=True)
                    elif df.crs != crs:
                        self.logger.info(f"Converting from {df.crs} to {crs}")
                        df = df.to_crs(crs)
                except Exception as e:
                    self.logger.error(f"Failed to transform CRS: {e}")
                    return None
                    
            return df
        except Exception as e:
            self.logger.error(f"Failed to read input file: {e}")
            return None
    
    def clip_to_neighborhood(self, data_gdf, nta_gdf, neighborhood_name):
        """
        Clip geodataframe to a specific neighborhood
        
        Parameters:
        -----------
        data_gdf : GeoDataFrame
            Data to clip
        nta_gdf : GeoDataFrame
            Neighborhoods GeoDataFrame
        neighborhood_name : str
            Name of neighborhood to clip to
            
        Returns:
        --------
        GeoDataFrame
            Clipped data or None if operation fails
        """
        try:
            self.logger.info(f"Clipping data to {neighborhood_name} neighborhood")
            
            # Check input validity
            if data_gdf is None or nta_gdf is None:
                self.logger.error("Input geodataframes cannot be None")
                return None
                
            if 'NTAName' not in nta_gdf.columns:
                self.logger.error("NTAName column not found in neighborhood data")
                return None
                
            # Get the specific neighborhood boundary
            target_nta = nta_gdf[nta_gdf.NTAName == neighborhood_name]
            
            if target_nta.empty:
                self.logger.error(f"Neighborhood '{neighborhood_name}' not found in NTA data")
                return None
                
            # Perform spatial join
            try:
                self.logger.info(f"Performing spatial join operation")
                clipped_data = gpd.sjoin(data_gdf, target_nta, predicate='within')
                self.logger.info(f"Clipped data contains {len(clipped_data)} features")
                return clipped_data
            except Exception as e:
                self.logger.error(f"Spatial join operation failed: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in clip_to_neighborhood: {e}")
            return None
    
    def simplify_geometries(self, gdf, tolerance=10):
        """
        Simplify geometries with error handling
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input geodataframe
        tolerance : float
            Simplification tolerance
            
        Returns:
        --------
        GeoDataFrame
            Geodataframe with simplified geometries
        """
        try:
            if gdf is None or gdf.empty:
                self.logger.error("Cannot simplify empty geodataframe")
                return None
                
            self.logger.info(f"Simplifying geometries with tolerance {tolerance}")
            
            # Create a copy to avoid modifying the original
            simplified = gdf.copy()
            
            # Check geometry validity
            invalid_count = simplified[~simplified.geometry.is_valid].shape[0]
            if invalid_count > 0:
                self.logger.warning(f"Found {invalid_count} invalid geometries, attempting to fix")
                simplified.geometry = simplified.geometry.buffer(0)
                
            # Apply simplification
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                simplified.geometry = simplified.geometry.simplify(tolerance)
                
            # Validate results
            error_count = simplified[~simplified.geometry.is_valid].shape[0]
            if error_count > 0:
                self.logger.warning(f"{error_count} geometries are invalid after simplification")
                
            empty_count = simplified[simplified.geometry.is_empty].shape[0]
            if empty_count > 0:
                self.logger.warning(f"{empty_count} geometries are empty after simplification")
                simplified = simplified[~simplified.geometry.is_empty]
                
            self.logger.info(f"Simplification complete, {len(simplified)} valid features remain")
            return simplified
            
        except Exception as e:
            self.logger.error(f"Error in simplify_geometries: {e}")
            return None

    def save_geoparquet(self, gdf, output_path):
        """
        Save geodataframe to GeoParquet file with GeoArrow encoding
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Geodataframe to save
        output_path : str
            Path to save file
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            if gdf is None or gdf.empty:
                self.logger.error("Cannot save empty geodataframe")
                return False
                
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            self.logger.info(f"Saving {len(gdf)} features to {output_path}")
            
            # Save data as GeoParquet with GeoArrow encoding
            try:
                gdf.to_parquet(
                    output_path,
                    compression='snappy',
                    geometry_encoding='geoarrow',  # Using GeoArrow instead of WKB
                    write_covering_bbox=True,
                    schema_version='1.1.0'  # Explicitly use GeoParquet 1.1.0 schema
                )
                self.logger.info("Data successfully written using GeoArrow encoding")
                return True
            except ImportError:
                self.logger.error("Failed to save as GeoParquet: pyarrow package is required")
                return False
            except Exception as e:
                self.logger.error(f"Failed to use GeoArrow encoding: {e}, falling back to WKB")
                try:
                    gdf.to_parquet(
                        output_path,
                        compression='snappy',
                        geometry_encoding='WKB',
                        write_covering_bbox=True
                    )
                    self.logger.info("Data successfully written using WKB encoding (fallback)")
                    return True
                except Exception as e2:
                    self.logger.error(f"Failed to save with fallback method: {e2}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error saving geodataframe: {e}")
            return False
            
    def ensure_crs(self, gdf, target_crs=PROJ_FT):
        """
        Ensure geodataframe is in the specified CRS
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input geodataframe
        target_crs : str, optional
            Target CRS, defaults to NYC projected feet
            
        Returns:
        --------
        GeoDataFrame
            Transformed geodataframe or None if transformation fails
        """
        try:
            if gdf is None or gdf.empty:
                self.logger.error("Cannot transform empty geodataframe")
                return None
                
            self.logger.info(f"Ensuring CRS is {target_crs}")
            
            if gdf.crs == target_crs:
                self.logger.info("CRS already matches target")
                return gdf
                
            try:
                result = gdf.to_crs(target_crs)
                self.logger.info(f"Successfully transformed from {gdf.crs} to {target_crs}")
                return result
            except Exception as e:
                self.logger.error(f"CRS transformation failed: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in ensure_crs: {e}")
            return None

    def clamp_column_values(self, df, column_name, lower_percentile=0.1, upper_percentile=99.9, 
                          inplace=False, track_changes=True):
        """
        Clamp values in a column to be between specified percentiles to remove outliers
        
        Parameters:
        -----------
        df : DataFrame or GeoDataFrame
            Input dataframe
        column_name : str
            Name of the column to clamp
        lower_percentile : float
            Lower percentile bound (0-100)
        upper_percentile : float
            Upper percentile bound (0-100)
        inplace : bool
            Whether to modify the dataframe in-place
        track_changes : bool
            Whether to log statistics about the changes
            
        Returns:
        --------
        DataFrame or GeoDataFrame
            Dataframe with clamped values
        """
        try:
            if df is None or df.empty:
                self.logger.error("Cannot clamp values in empty dataframe")
                return df
                
            if column_name not in df.columns:
                self.logger.error(f"Column '{column_name}' not found in dataframe")
                return df
                
            # Make a copy if not in-place
            result = df if inplace else df.copy()
            
            # Get original values for comparison if needed
            if track_changes:
                original_values = result[column_name].copy()
            
            # Calculate percentile bounds
            lower_bound = np.nanpercentile(result[column_name], lower_percentile)
            upper_bound = np.nanpercentile(result[column_name], upper_percentile)
            
            self.logger.info(f"Clamping '{column_name}' values between {lower_bound:.2f} ({lower_percentile}th percentile) "
                          f"and {upper_bound:.2f} ({upper_percentile}th percentile)")
            
            # Count values outside the bounds before clamping
            too_low_count = (result[column_name] < lower_bound).sum()
            too_high_count = (result[column_name] > upper_bound).sum()
            total_outside = too_low_count + too_high_count
            
            if total_outside > 0:
                percent_outside = (total_outside / len(result)) * 100
                self.logger.info(f"Found {total_outside} values ({percent_outside:.2f}%) outside the percentile bounds: "
                              f"{too_low_count} below, {too_high_count} above")
                
            # Apply clamping
            result[column_name] = result[column_name].clip(lower=lower_bound, upper=upper_bound)
            
            # Log statistics about the changes if requested
            if track_changes and total_outside > 0:
                # Calculate change statistics
                diffs = result[column_name] - original_values
                nonzero_diffs = diffs[diffs != 0]
                
                if not nonzero_diffs.empty:
                    abs_diffs = nonzero_diffs.abs()
                    self.logger.info(f"Change statistics for clamped values:")
                    self.logger.info(f"  - Mean absolute change: {abs_diffs.mean():.2f}")
                    self.logger.info(f"  - Max absolute change: {abs_diffs.max():.2f}")
                    self.logger.info(f"  - Min absolute change: {abs_diffs.min():.2f}")
                
                # Log the new column statistics
                new_stats = result[column_name].describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                self.logger.info(f"New statistics after clamping:")
                self.logger.info(f"  - Min: {new_stats['min']:.2f}")
                self.logger.info(f"  - Max: {new_stats['max']:.2f}")
                self.logger.info(f"  - Mean: {new_stats['mean']:.2f}")
                self.logger.info(f"  - Median: {new_stats['50%']:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in clamp_column_values: {e}")
            return df