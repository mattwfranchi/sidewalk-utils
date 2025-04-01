import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import fire

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
        """
        try:
            if left_df is None or right_df is None:
                self.logger.error("One or both input dataframes are None")
                return None, None
                
            # Make copies to avoid modifying originals
            left_df = left_df.copy()
            right_df = right_df.copy()
                
            # Create index-based IDs if columns don't exist
            if left_id_col not in left_df.columns:
                self.logger.warning(f"Left ID column '{left_id_col}' not found. Using index as ID.")
                left_df[left_id_col] = left_df.index.astype(str)  # Convert to string for safer joins
                    
            if right_id_col not in right_df.columns:
                self.logger.warning(f"Right ID column '{right_id_col}' not found. Using index as ID.")
                right_df[right_id_col] = right_df.index.astype(str)  # Convert to string for safer joins
            
            # Validate geometries
            if not all(isinstance(geom, Point) for geom in left_df.geometry):
                self.logger.warning("Not all geometries in left_df are points, results may be unexpected")
                    
            if not all(isinstance(geom, Point) for geom in right_df.geometry):
                self.logger.warning("Not all geometries in right_df are points, results may be unexpected")
            
            self.logger.info(f"Performing vectorized spatial join for {len(left_df)} left points and {len(right_df)} right points")
            
            # Create slim dataframes with ONLY the columns we absolutely need
            # CRITICAL: Include the ID columns explicitly
            left_slim = left_df[[left_id_col, 'geometry']].copy()
            right_slim = right_df[[right_id_col, 'geometry']].copy()
            
            # Perform the nearest join
            self.logger.info("Using sjoin_nearest for vectorized processing")
            joined = gpd.sjoin_nearest(
                left_slim, 
                right_slim,
                how="left", 
                max_distance=max_distance,
                distance_col="distance"
            )
            
            # Debug joined dataframe columns
            self.logger.debug(f"Columns in joined dataframe: {joined.columns.tolist()}")
            
            # Check for expected columns and handle missing ones
            right_id_joined = f"{right_id_col}_right"
            if right_id_joined not in joined.columns:
                self.logger.error(f"Expected right ID column '{right_id_joined}' not found in joined result")
                self.logger.error(f"Available columns: {joined.columns.tolist()}")
                # Try to find alternative column names that might contain right IDs
                potential_id_cols = [col for col in joined.columns if "right" in col.lower() or "index" in col.lower()]
                self.logger.info(f"Potential ID columns: {potential_id_cols}")
                if potential_id_cols:
                    right_id_joined = potential_id_cols[0]
                    self.logger.warning(f"Using '{right_id_joined}' as right ID column")
                else:
                    return None, None
            
            # Create the mapping dictionary with safer approach
            mapping_dict = {}
            for _, row in joined.iterrows():
                if left_id_col in row and right_id_joined in row and not pd.isna(row[right_id_joined]):
                    mapping_dict[row[left_id_col]] = row[right_id_joined]
            
            # Remove rows without a match
            joined = joined.dropna(subset=[right_id_joined])
            
            if len(joined) == 0:
                self.logger.warning("No matches found after spatial join")
                return None, None
            
            # Merge with the full left dataframe to get all attributes
            # Make sure to use the index for joining to preserve the relationship
            merged_df = left_df.merge(
                joined[["distance", right_id_joined]],
                left_index=True, 
                right_index=True
            )
            
            # Now merge with right dataframe to get all right attributes
            merged_df = merged_df.merge(
                right_df,
                left_on=right_id_joined,
                right_on=right_id_col,
                suffixes=('', '_right')
            )

            # Fix for multiple geometry columns and create a slim mapping dataframe
            self.logger.info("Creating mapping dataframe without geometry columns")

            # Create a slim version with just the mapping info
            mapping_df = pd.DataFrame({
                'left_id': merged_df[left_id_col],
                'right_id': merged_df[right_id_col],
                'distance': merged_df['distance']
            })

            match_percentage = (len(merged_df)/len(left_df))*100
            self.logger.info(f"Successfully matched {len(merged_df)} points ({match_percentage:.1f}%)")

            # Return both the full merged df for analysis and the slim mapping df for output
            return merged_df, mapping_dict, mapping_df
                
        except Exception as e:
            import traceback
            self.logger.error(f"Error in spatial join: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None

    def spatial_join_nearest_vectorized(self, left_df, right_df, left_id_col, right_id_col, max_distance=None):
        """Vectorized spatial join - much faster than point-by-point iteration"""
        try:
            self.logger.info("Performing vectorized spatial join")
            
            # Handle ID columns
            if left_id_col not in left_df.columns:
                self.logger.warning(f"Left ID column '{left_id_col}' not found. Using index as ID.")
                left_df[left_id_col] = left_df.index
                
            if right_id_col not in right_df.columns:
                self.logger.warning(f"Right ID column '{right_id_col}' not found. Using index as ID.")
                right_df[right_id_col] = right_df.index
            
            # Use GeoPandas' built-in spatial join with 'nearest' op
            self.logger.info("Using sjoin_nearest for vectorized processing")
            
            # Create a copy of the dataframes with only necessary columns to reduce memory usage
            left_slim = left_df[[left_id_col, 'geometry']].copy()
            right_slim = right_df[[right_id_col, 'geometry']].copy()
            
            # Perform the nearest join
            joined = gpd.sjoin_nearest(
                left_slim, 
                right_slim,
                how="left", 
                max_distance=max_distance,
                distance_col="distance"
            )
            
            # Create the mapping dictionary
            mapping_dict = dict(zip(joined[left_id_col], joined[f"{right_id_col}_right"]))
            
            # Remove rows without a match
            joined = joined.dropna(subset=[f"{right_id_col}_right"])
            
            # Merge with the full right dataframe to get all attributes
            merged_df = left_df.merge(
                joined[["distance", f"{right_id_col}_right"]], 
                left_index=True, 
                right_index=True
            )
            
            # Now merge with right dataframe to get all right attributes
            merged_df = merged_df.merge(
                right_df,
                left_on=f"{right_id_col}_right",
                right_on=right_id_col,
                suffixes=('', '_right')
            )
            
            match_percentage = (len(merged_df)/len(left_df))*100
            self.logger.info(f"Successfully matched {len(merged_df)} points ({match_percentage:.1f}%)")
            
            return merged_df, mapping_dict
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in vectorized spatial join: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None

    def map(self, left_file, right_file, 
            left_id_col="id", right_id_col="id", 
            target_crs=None, projected_crs=None,
            max_distance=None, output_prefix="point_mapping", 
            output_dir="output/mappings"):
        """
        Create a point-to-point mapping between two geodataframes.
        
        Args:
            left_file: Path to the left geodataframe file (points to match)
            right_file: Path to the right geodataframe file (reference points)
            left_id_col: Column name for the ID in left_df (default: "id")
            right_id_col: Column name for the ID in right_df (default: "id")
            target_crs: Target CRS for the mapping, if None use left_crs (optional)
            projected_crs: Projected CRS for accurate distance calculations (optional)
                If provided, data will be projected to this CRS before matching
                Example: "EPSG:26918" for UTM zone 18N
            max_distance: Maximum distance for a match, in target_crs units (optional)
                Note: If using projected_crs, this will be in meters
            output_prefix: Prefix for output filenames (default: "point_mapping")
            output_dir: Directory to save outputs (default: "output/mappings")
            
        Returns:
            True if the process succeeded, False otherwise
        """
        try:
            self.logger.info("Starting point-to-point mapping process")
            
            # Load left dataframe using base class method - CRS will be determined from the file
            left_df = self.read_geodataframe(left_file, name="left dataset")
            if left_df is None:
                return False
                
            # Load right dataframe using base class method - CRS will be determined from the file
            right_df = self.read_geodataframe(right_file, name="right dataset")
            if right_df is None:
                return False
            
            # Log the detected CRS
            self.logger.info(f"Left dataset CRS: {left_df.crs}")
            self.logger.info(f"Right dataset CRS: {right_df.crs}")
            
            # Detect geometry columns using base class method
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
                
            # Project to a projected CRS if specified
            if projected_crs:
                self.logger.info(f"Projecting data to {projected_crs} for accurate distance measurements")
                try:
                    left_df = left_df.to_crs(projected_crs)
                    right_df = right_df.to_crs(projected_crs)
                    self.logger.info(f"Successfully projected data. Distance units will be in the projected CRS units (usually meters).")
                    
                    # If max_distance was specified, inform the user about units
                    if max_distance is not None:
                        self.logger.info(f"Using max_distance={max_distance} in {projected_crs} units")
                except Exception as e:
                    self.logger.error(f"Error projecting data: {e}")
                    return False
            else:
                # Warn if using geographic CRS for distance calculations
                if left_df.crs.is_geographic:
                    self.logger.warning("Using a geographic CRS for distance calculations. "
                                        "Consider specifying a projected_crs for more accurate results.")
                    
                    if max_distance is not None:
                        self.logger.warning(f"Your max_distance={max_distance} is in degrees, which may not be what you expect.")
                        # Ask for confirmation
                        confirmation = input(f"Continue with max_distance={max_distance} in degrees? [y/N]: ")
                        if confirmation.lower() not in ['y', 'yes']:
                            self.logger.info("Operation canceled. Please specify a projected_crs.")
                            return False
                
            # Perform spatial join
            merged_df, mapping_dict, mapping_df = self.spatial_join_nearest(
                left_df, right_df, left_id_col, right_id_col, max_distance)
            if merged_df is None or mapping_dict is None:
                return False
                
            # Save outputs using base class method, passing the slim mapping dataframe
            success = self.save_outputs(merged_df, mapping_dict, mapping_df, output_prefix, output_dir)
            
            if success:
                self.logger.info("Point-to-point mapping completed successfully")
            else:
                self.logger.error("Point-to-point mapping failed")
                
            return success
        except Exception as e:
            self.logger.error(f"Unhandled exception in map method: {e}")
            return False


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