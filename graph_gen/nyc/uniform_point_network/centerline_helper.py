import os
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString
from shapely.ops import linemerge, nearest_points
import geopandas as gpd
from geopandas import GeoDataFrame
from centerline.geometry import Centerline
import warnings

import sys 
sys.path.append('/share/ju/sidewalk_utils')

# Import constants and logger
from data.nyc.c import WGS, PROJ_FT
from utils.logger import get_logger

class CenterlineHelper:
    """Helper class for extracting centerlines from sidewalk polygons."""
    
    def __init__(self):
        """Initialize the CenterlineHelper."""
        self.logger = get_logger("CenterlineHelper")
        
    def get_centerline(self, row):
        """Generate centerline for a geometry with enhanced error handling for Qhull precision issues"""
        try: 
            if row['geometry'] is None:
                self.logger.warning(f"Empty geometry found at index {row.name}")
                return None
                
            if row['geometry'].is_empty:
                self.logger.warning(f"Empty geometry found at index {row.name}")
                return None
                
            if not row['geometry'].is_valid:
                self.logger.warning(f"Invalid geometry found at index {row.name}, attempting to fix")
                fixed_geom = row['geometry'].buffer(0)
                if not fixed_geom.is_valid:
                    self.logger.error(f"Unable to fix invalid geometry at index {row.name}")
                    return None
                geom = fixed_geom
            else:
                geom = row['geometry']
                
            # Use more conservative simplification to preserve sidewalk detail
            simplified_geom = geom.simplify(0.05, preserve_topology=True)
            
            try:
                # First attempt with simplified geometry
                return Centerline(simplified_geom)
            except Exception as e:
                if "qhull precision error" in str(e):
                    self.logger.warning(f"Qhull precision error at index {row.name}, trying buffer technique")
                    
                    # Adjust buffer values for better handling of narrow sections
                    # Use smaller values to minimize shape changes
                    buffered_geom = geom.buffer(0.3).buffer(-0.25)
                    
                    if buffered_geom.is_empty:
                        self.logger.warning(f"Buffer technique produced empty geometry at index {row.name}")
                        # Try an alternative buffer technique with smaller values
                        buffered_geom = geom.buffer(0.2).buffer(-0.15)
                        if buffered_geom.is_empty:
                            return None
                        
                    try:
                        return Centerline(buffered_geom)
                    except Exception as buffer_e:
                        self.logger.error(f"Failed to generate centerline after buffer at index {row.name}: {buffer_e}")
                        return None
                else:
                    self.logger.error(f"Failed to generate centerline at index {row.name}: {e}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to generate centerline at index {row.name}: {e}")
            return None

    def remove_short_lines(self, line):
        """Remove short dead-end lines from a MultiLineString"""
        try:
            if line is None:
                self.logger.warning("None value passed to remove_short_lines")
                return None
                
            if line.is_empty:
                self.logger.warning("Empty geometry passed to remove_short_lines")
                return line
                
            if line.geom_type == 'MultiLineString':
                passing_lines = []
                line = line.geoms
            
                for i, linestring in enumerate(line):
                    other_lines = MultiLineString([x for j, x in enumerate(line) if j != i])
                    
                    p0 = Point(linestring.coords[0])
                    p1 = Point(linestring.coords[-1])
                    
                    is_deadend = False
                    
                    if p0.disjoint(other_lines): is_deadend = True
                    if p1.disjoint(other_lines): is_deadend = True
                    
                    if not is_deadend or linestring.length > 5:                
                        passing_lines.append(linestring)
                    else:
                        self.logger.debug(f"Removing short line segment of length {linestring.length}")
                
                if not passing_lines:
                    self.logger.debug("All lines removed in remove_short_lines, returning original")
                    return MultiLineString(line)
                    
                return MultiLineString(passing_lines)
                    
            if line.geom_type == 'LineString':
                return line
                
            self.logger.warning(f"Unexpected geometry type in remove_short_lines: {line.geom_type}")
            return line
        except Exception as e:
            self.logger.error(f"Error in remove_short_lines: {e}")
            return line

    def convert_polygons_to_centerlines(self, input_path, output_path=None):
        """
        Convert sidewalk polygons to centerlines for network analysis.
        
        Parameters:
        -----------
        input_path : str
            Path to input sidewalk polygon GeoJSON
        output_path : str, optional
            Path to save output centerlines. If None, will be derived from input path.
            
        Returns:
        --------
        GeoDataFrame
            Centerlines in LINESTRING format with parent_id for tracking
        """
        try:
            self.logger.info(f"Loading sidewalk polygons from: {input_path}")
            
            # Load sidewalk data
            df = gpd.read_file(input_path)
            self.logger.info(f"Loaded {len(df)} sidewalk polygons")
            
            # Check geometry validity
            invalid_count = df[~df.geometry.is_valid].shape[0]
            if invalid_count > 0:
                self.logger.warning(f"Found {invalid_count} invalid geometries, attempting to fix")
                df.geometry = df.geometry.buffer(0)
                still_invalid = df[~df.geometry.is_valid].shape[0]
                if still_invalid > 0:
                    self.logger.warning(f"{still_invalid} geometries still invalid after fix attempt")
            
            # Convert to appropriate CRS for processing
            self.logger.info(f"Transforming coordinate reference system to EPSG:3627")
            try:
                df = df.to_crs('EPSG:3627')
            except Exception as e:
                self.logger.error(f"Failed to transform CRS: {e}")
                return None
            
            # Dissolve geometries to merge adjacent polygons
            self.logger.info("Dissolving geometries")
            try:
                unary_union = df.union_all()
                if unary_union.geom_type == 'Polygon':
                    geoms = [unary_union]
                else:  # MultiPolygon
                    geoms = list(unary_union.geoms)
                    
                self.logger.info(f"Dissolved into {len(geoms)} geometries")
                df_dissolved = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms))
            except Exception as e:
                self.logger.error(f"Failed to dissolve geometries: {e}")
                return None
            
            # Explode multi-part geometries
            self.logger.info("Exploding multi-part geometries")
            df_exploded = df_dissolved.explode()
            self.logger.info(f"Exploded to {len(df_exploded)} features")
            
            # Prepare geometries for centerline calculation
            self.logger.info("Preparing geometries for centerline calculation")
            try:
                # Calculate area and perimeter for filtering
                df_exploded['area'] = df_exploded.geometry.area
                df_exploded['perimeter'] = df_exploded.geometry.length
                
                # Quick check for potentially problematic geometries
                # A very low area-to-perimeter ratio often indicates narrow shapes
                narrowness = df_exploded['area'] / (df_exploded['perimeter'] * df_exploded['perimeter'])
                very_narrow_count = (narrowness < 0.01).sum()
                
                if very_narrow_count > 0:
                    self.logger.warning(f"Found {very_narrow_count} potentially narrow geometries that may cause Qhull errors")
                    
                # Drop any tiny geometries that will likely cause problems
                tiny_geometries = df_exploded[df_exploded['area'] < 1].shape[0]
                if tiny_geometries > 0:
                    self.logger.warning(f"Removing {tiny_geometries} tiny geometries with area < 1 sq meter")
                    df_exploded = df_exploded[df_exploded['area'] >= 1]
                    
            except Exception as e:
                self.logger.warning(f"Error during geometry preparation: {e}")

            # Calculate centerlines
            self.logger.info("Calculating centerlines (this may take a while)")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Use parallel processing for better performance
                df_exploded['centerlines'] = df_exploded.parallel_apply(
                    lambda row: self.get_centerline(row), axis=1)
            
            # Drop rows where centerline calculation failed
            centerline_failures = df_exploded['centerlines'].isna().sum()
            if centerline_failures > 0:
                self.logger.warning(f"Failed to generate centerlines for {centerline_failures} geometries")
                
            self.logger.info("Dropping geometries with failed centerline calculation")
            df_exploded = df_exploded.dropna(subset=['centerlines'])
            self.logger.info(f"{len(df_exploded)} features remaining after dropping failed centerlines")
            
            if df_exploded.empty:
                self.logger.error("No valid centerlines generated, aborting")
                return None
            
            # Extract centerline geometries
            self.logger.info("Extracting centerline geometries")
            try:
                # Extract centerline geometries
                df_exploded['cl_geom'] = df_exploded['centerlines'].apply(
                    lambda x: x.geometry.geoms if hasattr(x, 'geometry') and hasattr(x.geometry, 'geoms') else None)
                
                # Drop rows with None geometry
                null_geom_count = df_exploded['cl_geom'].isna().sum()
                if null_geom_count > 0:
                    self.logger.warning(f"Found {null_geom_count} rows with null geometries after extraction")
                    df_exploded = df_exploded.dropna(subset=['cl_geom'])
                
                # Merge linestrings
                self.logger.info("Merging linestrings")
                def safe_linemerge(geom):
                    try:
                        if geom is None:
                            return None
                        result = linemerge(geom)
                        return result
                    except Exception as e:
                        self.logger.warning(f"Error during linemerge: {e}")
                        return None
                
                df_exploded['cl_geom'] = df_exploded['cl_geom'].parallel_apply(safe_linemerge)
                
                # Drop rows with None geometry after linemerge
                null_geom_count = df_exploded['cl_geom'].isna().sum()
                if null_geom_count > 0:
                    self.logger.warning(f"Found {null_geom_count} rows with null geometries after linemerge")
                    df_exploded = df_exploded.dropna(subset=['cl_geom'])
                
                # Remove short line ends
                self.logger.info("Removing short line ends")
                df_exploded['cl_geom'] = df_exploded['cl_geom'].parallel_apply(
                    lambda row: self.remove_short_lines(row))
                
                # Check for empty geometries after processing
                def is_empty_geom(geom):
                    try:
                        if geom is None:
                            return True
                        return geom.is_empty
                    except Exception:
                        self.logger.warning(f"Error checking if geometry is empty")
                        return True
                
                empty_count = df_exploded['cl_geom'].apply(is_empty_geom).sum()
                if empty_count > 0:
                    self.logger.warning(f"{empty_count} empty geometries after short line removal")
                    df_exploded = df_exploded[~df_exploded['cl_geom'].apply(is_empty_geom)]
                
                self.logger.info(f"{len(df_exploded)} features remaining after processing centerlines")
                
                if df_exploded.empty:
                    self.logger.error("No valid geometries remaining after centerline processing")
                    return None

            except Exception as e:
                self.logger.error(f"Error during centerline geometry processing: {e}")
                return None
            
            # Simplify geometries
            self.logger.info("Simplifying geometries")
            df_exploded['cl_geom'] = df_exploded['cl_geom'].parallel_apply(
                lambda row: row.simplify(0.5, preserve_topology=True))
            
            # Convert to final format
            self.logger.info("Creating final centerline dataframe")
            
            # Create segments dataframe with parent_id for tracking
            data = {'geometry': [], 'parent_id': []}
            
            for i, row in df_exploded.iterrows():
                geom = row['cl_geom']
                
                if geom.geom_type == 'MultiLineString':
                    for j, linestring in enumerate(geom.geoms):
                        data['geometry'].append(linestring)
                        data['parent_id'].append(f"{i}_{j}")
                elif geom.geom_type == 'LineString':
                    data['geometry'].append(geom)
                    data['parent_id'].append(str(i))
                else:
                    self.logger.warning(f"Unexpected geometry type: {geom.geom_type}")
                    
            if not data['geometry']:
                self.logger.error("No valid centerlines generated, aborting")
                return None
                
            df_centerlines = pd.DataFrame(data)
            df_centerlines = GeoDataFrame(df_centerlines, crs='EPSG:3627', geometry='geometry')
            self.logger.info(f"Created dataframe with {len(df_centerlines)} centerlines")
            
            # Convert to NYC projected feet
            self.logger.info(f"Converting to {PROJ_FT}")
            try:
                df_centerlines = df_centerlines.to_crs(PROJ_FT)
            except Exception as e:
                self.logger.error(f"Error converting to feet: {e}")
                return None
            
            # Save output if path provided
            if output_path:
                self.logger.info(f"Saving centerlines to {output_path}")
                try:
                    df_centerlines.to_parquet(output_path)
                    self.logger.info(f"Successfully saved centerlines to {output_path}")
                except Exception as e:
                    self.logger.error(f"Error saving centerlines: {e}")
            
            self.logger.success(f"Successfully converted {len(df)} polygons to {len(df_centerlines)} centerlines")
            return df_centerlines
            
        except Exception as e:
            self.logger.error(f"Unhandled exception in convert_polygons_to_centerlines: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    helper = CenterlineHelper()
    
    # Convert sidewalk polygons to centerlines
    input_path = "/share/ju/sidewalk_utils/data/nyc/_raw/Sidewalk.geojson"
    output_path = "/share/ju/sidewalk_utils/data/nyc/geo/nyc_sidewalk_centerlines.parquet"
    
    centerlines = helper.convert_polygons_to_centerlines(input_path, output_path)
    
    if centerlines is not None:
        print(f"Successfully created {len(centerlines)} centerlines")
    else:
        print("Failed to create centerlines") 