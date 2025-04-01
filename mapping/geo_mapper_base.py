import os
import geopandas as gpd
import pickle
import json
from utils.logger import get_logger
from shapely import Point

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
                try:
                    df = gpd.read_parquet(file_path)
                    # Check if this is a proper GeoParquet with geometry column
                    geom_col = self.detect_geometry_column(df)
                    if geom_col is None:
                        self.logger.warning(f"No geometry column found in parquet file, attempting to detect spatial data")
                        # Try to find columns that might represent coordinates
                        potential_x_cols = [col for col in df.columns if col.lower() in ['x', 'lon', 'longitude', 'easting']]
                        potential_y_cols = [col for col in df.columns if col.lower() in ['y', 'lat', 'latitude', 'northing']]
                        
                        if potential_x_cols and potential_y_cols:
                            x_col = potential_x_cols[0]
                            y_col = potential_y_cols[0]
                            self.logger.info(f"Converting columns {x_col} and {y_col} to geometry")
                            # Create geometry column from coordinates
                            df['geometry'] = [Point(x, y) for x, y in zip(df[x_col], df[y_col])]
                            df = gpd.GeoDataFrame(df, geometry='geometry')
                            if crs is not None:
                                df.set_crs(crs, inplace=True)
                        else:
                            self.logger.error(f"Could not identify geometry column or coordinate columns in {file_path}")
                except ImportError:
                    self.logger.error(f"Failed to read GeoParquet: pyarrow package may be missing")
                    return None
                except Exception as e:
                    self.logger.error(f"Failed to read GeoParquet: {e}")
                    return None
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
            
    def detect_geometry_column(self, df):
        """
        Detect the geometry column in a GeoDataFrame or DataFrame
        
        Parameters:
        -----------
        df : gpd.GeoDataFrame or pd.DataFrame
            DataFrame to examine
            
        Returns:
        --------
        str or None
            Name of the detected geometry column, or None if not found
        """
        try:
            # If it's already a proper GeoDataFrame with active geometry
            if hasattr(df, '_geometry_column_name') and df._geometry_column_name:
                return df._geometry_column_name
                
            # Check if there's a column named 'geometry' (the default)
            if 'geometry' in df.columns:
                return 'geometry'
                
            # Look for columns that might contain geometry objects
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check the first non-null value
                    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample and hasattr(sample, 'geom_type'):
                        return col
                        
            return None
        except Exception as e:
            self.logger.error(f"Error detecting geometry column: {e}")
            return None
            
    def save_outputs(self, merged_df, mapping_dict, mapping_df, output_prefix, output_dir):
        """Save mapping outputs to disk."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the mapping dataframe as parquet only
            mapping_parquet = os.path.join(output_dir, f"{output_prefix}_mapping.parquet")
            self.logger.info(f"Saving mapping dataframe to {mapping_parquet}")
            try:
                mapping_df.to_parquet(mapping_parquet, index=False)
                self.logger.info(f"Successfully saved mapping dataframe to Parquet")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save mapping Parquet: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving outputs: {e}")
            return False