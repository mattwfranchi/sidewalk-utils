import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import pickle
import fire
import json

# Import base class
from mapping.geo_mapper_base import GeoMapperBase

class PointMapper(GeoMapperBase):
    """Tool for mapping points between two geodataframes by spatial proximity."""
    
    def __init__(self):
        """Initialize the PointMapper with its own logger."""
        super().__init__(name=__name__)
    
    def spatial_join_nearest(self, left_df, right_df, left_id_col, right_id_col, max_distance=None):
        """
        Perform spatial join to find nearest point in right_df for each point in left_df
        
        Parameters:
        -----------
        left_df : gpd.GeoDataFrame
            First geodataframe (points to match)
        right_df : gpd.GeoDataFrame
            Second geodataframe (points to match to)
        left_id_col : str
            Column name for the ID in left_df
        right_id_col : str
            Column name for the ID in right_df
        max_distance : float, optional
            Maximum distance for a match, in CRS units
            
        Returns:
        --------
        tuple
            (merged_df, mapping_dict) or (None, None) if join fails
        """
        try:
            if left_df is None or right_df is None:
                self.logger.error("One or both input dataframes are None")
                return None, None
                
            if left_id_col not in left_df.columns:
                self.logger.error(f"Left ID column '{left_id_col}' not found in left dataframe")
                return None, None
                
            if right_id_col not in right_df.columns:
                self.logger.error(f"Right ID column '{right_id_col}' not found in right dataframe")
                return None, None
                
            # Validate geometries
            if not all(isinstance(geom, Point) for geom in left_df.geometry):
                self.logger.warning("Not all geometries in left_df are points, results may be unexpected")
                
            if not all(isinstance(geom, Point) for geom in right_df.geometry):
                self.logger.warning("Not all geometries in right_df are points, results may be unexpected")
                
            self.logger.info(f"Performing spatial join for {len(left_df)} left points and {len(right_df)} right points")
            
            # Create R-tree spatial index on right_df for efficient nearest-neighbor search
            self.logger.info("Building spatial index on right dataframe")
            right_sindex = right_df.sindex
            
            # Prepare result containers
            merged_data = []
            mapping_dict = {}
            
            # For each point in left_df, find nearest in right_df
            for idx, left_row in tqdm(left_df.iterrows(), total=len(left_df), desc="Matching points"):
                left_point = left_row.geometry
                left_id = left_row[left_id_col]
                
                # Query spatial index for candidate matches
                possible_matches_idx = list(right_sindex.nearest(left_point.bounds, 10))
                
                if not possible_matches_idx:
                    self.logger.debug(f"No candidate matches found for point {left_id}")
                    continue
                    
                possible_matches = right_df.iloc[possible_matches_idx]
                
                # Calculate distances and find the nearest
                distances = possible_matches.geometry.distance(left_point)
                nearest_idx = distances.idxmin()
                min_distance = distances.min()
                
                # Check max distance if specified
                if max_distance is not None and min_distance > max_distance:
                    self.logger.debug(f"Point {left_id} has no match within {max_distance} units")
                    continue
                    
                # Get the matched right row
                right_row = right_df.loc[nearest_idx]
                right_id = right_row[right_id_col]
                
                # Store the matching
                mapping_dict[left_id] = right_id
                
                # Create merged row
                merged_row = left_row.to_dict()
                merged_row.update({
                    f"right_{key}": value for key, value in right_row.to_dict().items()
                    if key != 'geometry'
                })
                merged_row['distance'] = min_distance
                merged_data.append(merged_row)
                
            # Create merged dataframe
            if not merged_data:
                self.logger.warning("No matches found")
                return None, None
                
            merged_df = gpd.GeoDataFrame(merged_data, crs=left_df.crs)
            
            match_percentage = (len(merged_df)/len(left_df))*100
            self.logger.info(f"Successfully matched {len(merged_df)} points ({match_percentage:.1f}%)")
            return merged_df, mapping_dict
        except Exception as e:
            self.logger.error(f"Error in spatial_join_nearest: {e}")
            return None, None

    def save_outputs(self, merged_df, mapping_dict, output_prefix, output_dir):
        """
        Save the merged dataframe and mapping dictionary to files
        
        Parameters:
        -----------
        merged_df : gpd.GeoDataFrame
            Merged geodataframe
        mapping_dict : dict
            Mapping dictionary of left_id -> right_id
        output_prefix : str
            Prefix for output filenames
        output_dir : str
            Directory to save outputs
            
        Returns:
        --------
        bool
            True if saving succeeded, False otherwise
        """
        try:
            if merged_df is None or mapping_dict is None:
                self.logger.error("Merged dataframe or mapping dictionary is None")
                return False
                
            # Ensure output directory exists using base class method
            if not self.ensure_output_directory(output_dir):
                return False
                
            # Save merged dataframe using base class method
            merged_file = os.path.join(output_dir, f"{output_prefix}_merged.geojson")
            self.logger.info(f"Saving merged dataframe to {merged_file}")
            if not self.save_geojson(merged_df, merged_file):
                return False
            
            # Save mapping dictionary
            mapping_file = os.path.join(output_dir, f"{output_prefix}_mapping.pickle")
            self.logger.info(f"Saving mapping dictionary to {mapping_file}")
            try:
                with open(mapping_file, 'wb') as f:
                    pickle.dump(mapping_dict, f)
            except Exception as e:
                self.logger.error(f"Failed to save mapping dictionary: {e}")
                return False
            
            # Also save a JSON version for better interoperability
            json_file = os.path.join(output_dir, f"{output_prefix}_mapping.json") 
            self.logger.info(f"Saving JSON mapping to {json_file}")
            try:
                with open(json_file, 'w') as f:
                    # Convert all keys to strings for JSON compatibility
                    json_dict = {str(k): str(v) for k, v in mapping_dict.items()}
                    json.dump(json_dict, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save JSON mapping: {e}")
                # Continue even if JSON save fails
                
            # Save as GeoParquet option
            try:
                parquet_file = os.path.join(output_dir, f"{output_prefix}_merged.parquet")
                self.logger.info(f"Also saving as GeoParquet to {parquet_file}")
                self.save_geoparquet(merged_df, parquet_file)
                # Continue even if GeoParquet save fails
            except Exception as e:
                self.logger.warning(f"Failed to save as GeoParquet (optional): {e}")
                
            self.logger.info(f"Successfully saved outputs with prefix {output_prefix}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving outputs: {e}")
            return False

    def map(self, left_file, right_file, 
            left_id_col="id", right_id_col="id", 
            left_crs=None, right_crs=None, target_crs=None, 
            max_distance=None, output_prefix="point_mapping", 
            output_dir="output/mappings"):
        """
        Create a point-to-point mapping between two geodataframes.
        
        Args:
            left_file: Path to the left geodataframe file (points to match)
            right_file: Path to the right geodataframe file (reference points)
            left_id_col: Column name for the ID in left_df (default: "id")
            right_id_col: Column name for the ID in right_df (default: "id")
            left_crs: CRS of the left geodataframe (optional)
            right_crs: CRS of the right geodataframe (optional)
            target_crs: Target CRS for the mapping, if None use left_crs (optional)
            max_distance: Maximum distance for a match, in target_crs units (optional)
            output_prefix: Prefix for output filenames (default: "point_mapping")
            output_dir: Directory to save outputs (default: "output/mappings")
            
        Returns:
            True if the process succeeded, False otherwise
        """
        try:
            self.logger.info("Starting point-to-point mapping process")
            
            # Load left dataframe using base class method
            left_df = self.read_geodataframe(left_file, left_crs, name="left dataset")
            if left_df is None:
                return False
                
            # Load right dataframe using base class method
            right_df = self.read_geodataframe(right_file, right_crs, name="right dataset")
            if right_df is None:
                return False
            
            # Detect geometry columns
            left_geom_col = self.detect_geometry_column(left_df)
            right_geom_col = self.detect_geometry_column(right_df)
            
            self.logger.info(f"Detected geometry column in left dataset: {left_geom_col}")
            self.logger.info(f"Detected geometry column in right dataset: {right_geom_col}")
            
            # Prompt user for confirmation
            confirmation = input(f"Proceed with geometry columns '{left_geom_col}' and '{right_geom_col}'? [Y/n]: ")
            if confirmation.lower() not in ['', 'y', 'yes']:
                self.logger.info("Operation canceled by user")
                return False
            
            # Set the active geometry columns if they're not the default 'geometry'
            if left_geom_col and left_geom_col != 'geometry':
                left_df = left_df.set_geometry(left_geom_col)
            
            if right_geom_col and right_geom_col != 'geometry':
                right_df = right_df.set_geometry(right_geom_col)
                
            # Ensure both dataframes have the same CRS using base class method
            left_df, right_df = self.ensure_common_crs(left_df, right_df, target_crs)
            if left_df is None or right_df is None:
                return False
                
            # Perform spatial join
            merged_df, mapping_dict = self.spatial_join_nearest(
                left_df, right_df, left_id_col, right_id_col, max_distance)
            if merged_df is None or mapping_dict is None:
                return False
                
            # Save outputs
            success = self.save_outputs(merged_df, mapping_dict, output_prefix, output_dir)
            
            if success:
                self.logger.info("Point-to-point mapping completed successfully")
            else:
                self.logger.error("Point-to-point mapping failed")
                
            return success
        except Exception as e:
            self.logger.error(f"Unhandled exception in map method: {e}")
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


if __name__ == "__main__":
    fire.Fire(PointMapper)

# Example usage from command line:
# python point2point.py map \
#   --left_file=data/points_to_match.geojson \
#   --right_file=data/reference_points.geojson \
#   --left_id_col=point_id \
#   --right_id_col=ref_id \
#   --target_crs="EPSG:4326" \
#   --max_distance=100 \
#   --output_prefix=point_matching_results \
#   --output_dir=output/mappings