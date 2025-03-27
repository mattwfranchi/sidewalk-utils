import os
import geopandas as gpd
from utils.logger import get_logger

class GeoMapperBase:
    """Base class for geospatial mapping tools with common utilities."""

    def __init__(self, name=None):
        """Initialize the mapper with its own logger."""
        module_name = name or __name__
        self.logger = get_logger(module_name)

    def read_geodataframe(self, file_path, crs=None, name="dataset"):
        """
        Read geodataframe from file with error handling
        
        Parameters:
        -----------
        file_path : str
            Path to the geodataframe file
        crs : str, optional
            Coordinate reference system to use
        name : str
            Name of the dataset for logging
            
        Returns:
        --------
        gpd.GeoDataFrame or None
            The loaded geodataframe or None if loading fails
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Input file not found: {file_path}")
                return None
                
            self.logger.info(f"Loading {name} from {file_path}")
            
            # Add file extension detection:
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

    def ensure_common_crs(self, left_df, right_df, target_crs=None):
        """
        Ensure both geodataframes use the same coordinate reference system
        
        Parameters:
        -----------
        left_df : gpd.GeoDataFrame
            First geodataframe
        right_df : gpd.GeoDataFrame
            Second geodataframe
        target_crs : str, optional
            Target CRS to convert both dataframes to. If None, use left_df's CRS
            
        Returns:
        --------
        tuple
            (left_df, right_df) with aligned CRS, or (None, None) if conversion fails
        """
        try:
            if left_df is None or right_df is None:
                self.logger.error("One or both input dataframes are None")
                return None, None
                
            # Determine target CRS
            if target_crs is None:
                if left_df.crs is None:
                    if right_df.crs is None:
                        self.logger.error("Both dataframes have no CRS and no target CRS provided")
                        return None, None
                    target_crs = right_df.crs
                    self.logger.info(f"Using right dataframe CRS: {target_crs}")
                else:
                    target_crs = left_df.crs
                    self.logger.info(f"Using left dataframe CRS: {target_crs}")
            else:
                self.logger.info(f"Using provided target CRS: {target_crs}")
                
            # Convert left dataframe if needed
            if left_df.crs != target_crs:
                try:
                    self.logger.info(f"Converting left dataframe from {left_df.crs} to {target_crs}")
                    left_df = left_df.to_crs(target_crs)
                except Exception as e:
                    self.logger.error(f"Failed to convert left dataframe to {target_crs}: {e}")
                    return None, None
                    
            # Convert right dataframe if needed
            if right_df.crs != target_crs:
                try:
                    self.logger.info(f"Converting right dataframe from {right_df.crs} to {target_crs}")
                    right_df = right_df.to_crs(target_crs)
                except Exception as e:
                    self.logger.error(f"Failed to convert right dataframe to {target_crs}: {e}")
                    return None, None
                    
            return left_df, right_df
        except Exception as e:
            self.logger.error(f"Error in ensure_common_crs: {e}")
            return None, None

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

    def save_geojson(self, gdf, output_path):
        """
        Save geodataframe to GeoJSON file
        
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
            
            # Save data
            try:
                gdf.to_file(output_path, driver="GeoJSON")
                self.logger.info("Data successfully written to GeoJSON")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save to GeoJSON: {e}")
                return False
                    
        except Exception as e:
            self.logger.error(f"Error saving geodataframe: {e}")
            return False

    def ensure_output_directory(self, output_dir):
        """
        Ensure output directory exists, creating it if necessary
        
        Parameters:
        -----------
        output_dir : str
            Directory path to ensure exists
            
        Returns:
        --------
        bool
            True if directory exists or was created, False if creation failed
        """
        try:
            if not output_dir:
                self.logger.warning("Empty output directory provided")
                return False
                
            if not os.path.exists(output_dir):
                self.logger.info(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                
            return os.path.exists(output_dir)
        except Exception as e:
            self.logger.error(f"Failed to create output directory {output_dir}: {e}")
            return False