import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm
import fire

# Import base class
from mapping.geo_mapper_base import GeoMapperBase

class PointToPolygonMapper(GeoMapperBase):
    """Tool for mapping points to polygons between two geodataframes by spatial relationships.
    
    This mapper supports both Polygon and MultiPolygon geometries, as well as mixed geometry types
    in the polygon geodataframe. It efficiently maps each point to the containing polygon using
    spatial indexing, and handles edge cases like multiple containing polygons.
    """
    
    def __init__(self):
        """Initialize the PointToPolygonMapper with its own logger."""
        super().__init__(name=__name__)
    
    def spatial_join_contains(self, point_df, polygon_df, point_id_col, polygon_id_col):
        """
        Perform spatial join to find polygon in polygon_df that contains each point in point_df
        
        Parameters:
        -----------
        point_df : gpd.GeoDataFrame
            Point geodataframe (points to map)
        polygon_df : gpd.GeoDataFrame
            Polygon/MultiPolygon geodataframe (polygons that may contain points)
        point_id_col : str
            Column name for the ID in point_df
        polygon_id_col : str
            Column name for the ID in polygon_df
            
        Returns:
        --------
        tuple
            (merged_df, mapping_dict) or (None, None) if join fails
        """
        try:
            if point_df is None or polygon_df is None:
                self.logger.error("One or both input dataframes are None")
                return None, None
                
            if point_id_col not in point_df.columns:
                self.logger.error(f"Point ID column '{point_id_col}' not found in point dataframe")
                return None, None
                
            if polygon_id_col not in polygon_df.columns:
                self.logger.error(f"Polygon ID column '{polygon_id_col}' not found in polygon dataframe")
                return None, None
                
            # Validate geometries
            if not all(isinstance(geom, Point) for geom in point_df.geometry):
                self.logger.warning("Not all geometries in point_df are points, results may be unexpected")
                
            # Check for polygon types
            polygon_types = set(type(geom).__name__ for geom in polygon_df.geometry)
            self.logger.info(f"Polygon dataset contains geometry types: {polygon_types}")
            
            if not all(hasattr(geom, 'contains') for geom in polygon_df.geometry):
                self.logger.error("Some geometries in polygon_df don't have 'contains' method")
                return None, None
            
            # Count polygon types
            polygon_count = sum(1 for g in polygon_df.geometry if isinstance(g, Polygon))
            multipolygon_count = sum(1 for g in polygon_df.geometry if isinstance(g, MultiPolygon))
            other_count = len(polygon_df) - polygon_count - multipolygon_count
            
            self.logger.info(f"Polygon dataset contains: {polygon_count} Polygons, {multipolygon_count} MultiPolygons, {other_count} other geometries")
            
            self.logger.info(f"Performing spatial join for {len(point_df)} points and {len(polygon_df)} polygon features")
            
            # Create R-tree spatial index on polygon_df for efficient containment search
            self.logger.info("Building spatial index on polygon dataframe")
            polygon_sindex = polygon_df.sindex
            
            # Prepare result containers
            merged_data = []
            mapping_dict = {}
            
            # Statistics for logging
            total_points = len(point_df)
            matched_points = 0
            
            # For each point in point_df, find containing polygon in polygon_df
            for idx, point_row in tqdm(point_df.iterrows(), total=total_points, desc="Mapping points to polygons"):
                point_geom = point_row.geometry
                point_id = point_row[point_id_col]
                
                # Query spatial index for candidate polygons that might contain the point
                possible_matches_idx = list(polygon_sindex.query(point_geom.bounds))
                
                if not possible_matches_idx:
                    self.logger.debug(f"No candidate polygons found for point {point_id}")
                    continue
                    
                possible_matches = polygon_df.iloc[possible_matches_idx]
                
                # Find polygons that actually contain the point
                # Using a list comprehension for better performance with multipolygons
                containing_indices = [i for i, (_, poly_row) in enumerate(possible_matches.iterrows()) 
                                    if poly_row.geometry.contains(point_geom)]
                
                if not containing_indices:
                    self.logger.debug(f"Point {point_id} is not contained in any polygon")
                    continue
                
                # Extract containing polygons
                containing_polygons = possible_matches.iloc[containing_indices]
                
                # If multiple polygons contain the point, use the smallest one by area
                if len(containing_polygons) > 1:
                    containing_polygons['area'] = containing_polygons.geometry.area
                    containing_polygons = containing_polygons.sort_values('area')
                    self.logger.debug(f"Point {point_id} is contained in {len(containing_polygons)} polygons, using smallest")
                
                # Get the matched polygon row
                polygon_row = containing_polygons.iloc[0]
                polygon_id = polygon_row[polygon_id_col]
                
                # Store the matching
                mapping_dict[point_id] = polygon_id
                
                # Create merged row
                merged_row = point_row.to_dict()
                merged_row.update({
                    f"polygon_{key}": value for key, value in polygon_row.to_dict().items()
                    if key != 'geometry'
                })
                merged_data.append(merged_row)
                matched_points += 1
                
            # Create merged dataframe
            if not merged_data:
                self.logger.warning("No matches found")
                return None, None
                
            merged_df = gpd.GeoDataFrame(merged_data, crs=point_df.crs)
            
            # Log detailed match statistics
            match_percentage = (matched_points/total_points)*100
            self.logger.info(f"Successfully matched {matched_points} of {total_points} points ({match_percentage:.1f}%)")
            
            if matched_points < total_points:
                self.logger.warning(f"{total_points - matched_points} points ({100-match_percentage:.1f}%) could not be matched to any polygon")
                
            return merged_df, mapping_dict
        except Exception as e:
            self.logger.error(f"Error in spatial_join_contains: {e}")
            return None, None
            
    def map(self, point_file, polygon_file,
            point_id_col="id", polygon_id_col="id",
            point_crs=None, polygon_crs=None, target_crs=None,
            output_prefix="point_to_polygon_mapping",
            output_dir="output/mappings"):
        """
        Create a mapping from points to containing polygons.
        
        Args:
            point_file: Path to the point geodataframe file (points to map)
            polygon_file: Path to the polygon geodataframe file (containing polygons)
            point_id_col: Column name for the ID in point dataframe (default: "id")
            polygon_id_col: Column name for the ID in polygon dataframe (default: "id")
            point_crs: CRS of the point geodataframe (optional)
            polygon_crs: CRS of the polygon geodataframe (optional)
            target_crs: Target CRS for the mapping, if None use point_crs (optional)
            output_prefix: Prefix for output filenames (default: "point_to_polygon_mapping")
            output_dir: Directory to save outputs (default: "output/mappings")
            
        Returns:
            True if the process succeeded, False otherwise
        """
        try:
            self.logger.info("Starting point-to-polygon mapping process")
            
            # Load point dataframe using base class method
            point_df = self.read_geodataframe(point_file, point_crs, name="point dataset")
            if point_df is None:
                return False
                
            # Load polygon dataframe using base class method
            polygon_df = self.read_geodataframe(polygon_file, polygon_crs, name="polygon dataset")
            if polygon_df is None:
                return False
            
            # Detect geometry columns using base class method
            point_geom_col = self.detect_geometry_column(point_df)
            polygon_geom_col = self.detect_geometry_column(polygon_df)
            
            self.logger.info(f"Detected geometry column in point dataset: {point_geom_col}")
            self.logger.info(f"Detected geometry column in polygon dataset: {polygon_geom_col}")
            
            # Prompt user for confirmation
            confirmation = input(f"Proceed with geometry columns '{point_geom_col}' and '{polygon_geom_col}'? [Y/n]: ")
            if confirmation.lower() not in ['', 'y', 'yes']:
                self.logger.info("Operation canceled by user")
                return False
            
            # Set the active geometry columns if they're not the default 'geometry'
            if point_geom_col and point_geom_col != 'geometry':
                point_df = point_df.set_geometry(point_geom_col)
            
            if polygon_geom_col and polygon_geom_col != 'geometry':
                polygon_df = polygon_df.set_geometry(polygon_geom_col)
                
            # Ensure both dataframes have the same CRS using base class method
            point_df, polygon_df = self.ensure_common_crs(point_df, polygon_df, target_crs)
            if point_df is None or polygon_df is None:
                return False
                
            # Perform spatial join
            merged_df, mapping_dict = self.spatial_join_contains(
                point_df, polygon_df, point_id_col, polygon_id_col)
            if merged_df is None or mapping_dict is None:
                return False
                
            # Save outputs using base class method
            success = self.save_outputs(merged_df, mapping_dict, output_prefix, output_dir)
            
            if success:
                self.logger.info("Point-to-polygon mapping completed successfully")
            else:
                self.logger.error("Point-to-polygon mapping failed")
                
            return success
        except Exception as e:
            self.logger.error(f"Unhandled exception in map method: {e}")
            return False


if __name__ == "__main__":
    fire.Fire(PointToPolygonMapper)

# Example usage from command line:
# python point2polygon.py map \
#   --point_file=data/points.geojson \
#   --polygon_file=data/polygons.geojson \
#   --point_id_col=point_id \
#   --polygon_id_col=polygon_id \
#   --target_crs="EPSG:4326" \
#   --output_prefix=point_polygon_results \
#   --output_dir=output/mappings