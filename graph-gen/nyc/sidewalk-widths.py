import os
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString
from shapely.ops import linemerge, nearest_points
import geopandas as gpd
from geopandas import GeoDataFrame
from centerline.geometry import Centerline
from tqdm import tqdm 
from pandarallel import pandarallel
import warnings

# Import constants and logger
from utils.logger import get_logger
from data.nyc.c import WGS, PROJ_FT
from data.nyc.io import NYC_OPENDATA_SIDEWALKS, NYC_DATA_PROCESSING_OUTPUT_DIR

# Initialize logger
logger = get_logger(__name__)

# Initialize parallel processing
pandarallel.initialize(progress_bar=True, nb_workers=8)

def get_centerline(row):
    """Generate centerline for a geometry with enhanced error handling for Qhull precision issues"""
    try: 
        if row['geometry'] is None:
            logger.warning(f"Empty geometry found at index {row.name}")
            return None
            
        if row['geometry'].is_empty:
            logger.warning(f"Empty geometry found at index {row.name}")
            return None
            
        if not row['geometry'].is_valid:
            logger.warning(f"Invalid geometry found at index {row.name}, attempting to fix")
            fixed_geom = row['geometry'].buffer(0)
            if not fixed_geom.is_valid:
                logger.error(f"Unable to fix invalid geometry at index {row.name}")
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
                logger.warning(f"Qhull precision error at index {row.name}, trying buffer technique")
                
                # Adjust buffer values for better handling of narrow sections
                # Use smaller values to minimize shape changes
                buffered_geom = geom.buffer(0.3).buffer(-0.25)
                
                if buffered_geom.is_empty:
                    logger.warning(f"Buffer technique produced empty geometry at index {row.name}")
                    # Try an alternative buffer technique with smaller values
                    buffered_geom = geom.buffer(0.2).buffer(-0.15)
                    if buffered_geom.is_empty:
                        return None
                    
                try:
                    return Centerline(buffered_geom)
                except Exception as buffer_e:
                    logger.error(f"Failed to generate centerline after buffer at index {row.name}: {buffer_e}")
                    return None
            else:
                logger.error(f"Failed to generate centerline at index {row.name}: {e}")
                return None
                
    except Exception as e:
        logger.error(f"Failed to generate centerline at index {row.name}: {e}")
        return None

def remove_short_lines(line):
    """Remove short dead-end lines from a MultiLineString"""
    try:
        if line is None:
            logger.warning("None value passed to remove_short_lines")
            return None
            
        if line.is_empty:
            logger.warning("Empty geometry passed to remove_short_lines")
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
                    logger.debug(f"Removing short line segment of length {linestring.length}")
            
            if not passing_lines:
                logger.warning("All lines removed in remove_short_lines, returning original")
                return MultiLineString(line)
                
            return MultiLineString(passing_lines)
                
        if line.geom_type == 'LineString':
            return line
            
        logger.warning(f"Unexpected geometry type in remove_short_lines: {line.geom_type}")
        return line
    except Exception as e:
        logger.error(f"Error in remove_short_lines: {e}")
        return line

def linestring_to_segments(linestring):
    """Convert LineString to list of segment LineStrings"""
    try:
        if linestring is None or linestring.is_empty:
            return []
            
        if len(linestring.coords) < 2:
            logger.warning(f"LineString with insufficient coordinates: {len(linestring.coords)}")
            return []
            
        return [LineString([linestring.coords[i], linestring.coords[i+1]]) for i in range(len(linestring.coords) - 1)]
    except Exception as e:
        logger.error(f"Error in linestring_to_segments: {e}")
        return []

def get_segments(line):
    """Extract all segments from a LineString or MultiLineString"""
    try:
        if line is None:
            logger.warning("None value passed to get_segments")
            return []
            
        line_segments = []

        if line.geom_type == 'MultiLineString':
            for linestring in line.geoms:
                segments = linestring_to_segments(linestring)
                line_segments.extend(segments)
        elif line.geom_type == 'LineString':
            line_segments.extend(linestring_to_segments(line))
        else:
            logger.warning(f"Unexpected geometry type in get_segments: {line.geom_type}")
            
        if not line_segments:
            logger.warning("No segments generated")
            
        return line_segments
    except Exception as e:
        logger.error(f"Error in get_segments: {e}")
        return []

def interpolate_by_distance(linestring):
    """Interpolate points along a linestring at regular intervals"""
    try:
        if linestring is None or linestring.is_empty:
            return []
            
        distance = 1
        all_points = []
        
        if linestring.length == 0:
            logger.warning("Zero-length linestring encountered in interpolation")
            return []
            
        count = round(linestring.length / distance) + 1
        
        if count == 1:
            all_points.append(linestring.interpolate(linestring.length / 2))
        else:
            for i in range(count):
                all_points.append(linestring.interpolate(distance * i))
        
        return all_points
    except Exception as e:
        logger.error(f"Error in interpolate_by_distance: {e}")
        return []

def interpolate(line):
    """Interpolate points along a LineString or MultiLineString"""
    try:
        if line is None:
            logger.warning("None value passed to interpolate")
            return MultiPoint([])
            
        if line.is_empty:
            logger.warning("Empty geometry passed to interpolate")
            return MultiPoint([])
            
        if line.geom_type == 'MultiLineString':
            all_points = []
            for linestring in line.geoms:
                all_points.extend(interpolate_by_distance(linestring))
            
            if not all_points:
                logger.warning("No points generated during MultiLineString interpolation")
                
            return MultiPoint(all_points)
                
        if line.geom_type == 'LineString':
            points = interpolate_by_distance(line)
            
            if not points:
                logger.warning("No points generated during LineString interpolation")
                
            return MultiPoint(points)
            
        logger.warning(f"Unexpected geometry type in interpolate: {line.geom_type}")
        return MultiPoint([])
    except Exception as e:
        logger.error(f"Error in interpolate: {e}")
        return MultiPoint([])
    
def polygon_to_multilinestring(polygon):
    """Convert polygon to MultiLineString of boundaries"""
    try:
        if polygon is None:
            logger.warning("None value passed to polygon_to_multilinestring")
            return None
            
        if polygon.is_empty:
            logger.warning("Empty geometry passed to polygon_to_multilinestring")
            return MultiLineString([])
            
        exterior = polygon.exterior
        if exterior is None:
            logger.warning("Polygon with no exterior found")
            return MultiLineString([])
            
        return MultiLineString([exterior] + [line for line in polygon.interiors])
    except Exception as e:
        logger.error(f"Error in polygon_to_multilinestring: {e}")
        return MultiLineString([])
    
def get_avg_distances(row):
    """Calculate average distances from centerline to sidewalk edges"""
    try:
        avg_distances = []
        
        if row.geometry is None or row.geometry.is_empty:
            logger.warning(f"Empty geometry at index {row.name}")
            return []
            
        if not hasattr(row, 'segments') or not row.segments:
            logger.warning(f"No segments found at index {row.name}")
            return []
            
        sidewalk_lines = polygon_to_multilinestring(row.geometry)
        
        if sidewalk_lines is None or sidewalk_lines.is_empty:
            logger.warning(f"Failed to convert polygon to multilinestring at index {row.name}")
            return []
        
        for segment in row.segments:
            points = interpolate(segment)
            
            if points is None or points.is_empty or len(points.geoms) == 0:
                logger.warning(f"No interpolation points generated for segment at index {row.name}")
                avg_distances.append(0)
                continue
                
            distances = []
            valid_point_count = 0
            
            for point in points.geoms:
                try:
                    p1, p2 = nearest_points(sidewalk_lines, point)
                    distance = p1.distance(p2)
                    # Filter out unreasonable distances (e.g., more than 50m - likely errors)
                    if distance <= 50:
                        distances.append(distance)
                        valid_point_count += 1
                except Exception as e:
                    logger.warning(f"Failed to calculate distance: {e}")
                    
            if distances and valid_point_count > 0:
                avg_distances.append(sum(distances) / valid_point_count)
            else:
                logger.warning(f"No valid distances calculated for segment at index {row.name}")
                avg_distances.append(0)
                
        return avg_distances
    except Exception as e:
        logger.error(f"Error in get_avg_distances at index {row.name}: {e}")
        return []

def process_sidewalks(input_path, output_path):
    """Process sidewalk data to extract widths"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Check if input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return False
            
        logger.info(f"Loading sidewalk data from {input_path}")
        
        # Try to load the file with error handling
        try:
            df = gpd.read_file(input_path)
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            return False
            
        if df.empty:
            logger.error(f"Input file contains no data")
            return False
        
        logger.info(f"Loaded {len(df)} features")
        
        # Check geometry validity
        invalid_count = df[~df.geometry.is_valid].shape[0]
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid geometries, attempting to fix")
            df.geometry = df.geometry.buffer(0)
            still_invalid = df[~df.geometry.is_valid].shape[0]
            if still_invalid > 0:
                logger.warning(f"{still_invalid} geometries still invalid after fix attempt")
        
        # Convert to appropriate CRS for processing
        logger.info(f"Transforming coordinate reference system to EPSG:3627")
        try:
            df = df.to_crs('EPSG:3627')
        except Exception as e:
            logger.error(f"Failed to transform CRS: {e}")
            return False
        
        # Dissolve geometries
        logger.info("Dissolving geometries")
        try:
            unary_union = df.union_all()
            if unary_union.geom_type == 'Polygon':
                geoms = [unary_union]
            else:  # MultiPolygon
                geoms = list(unary_union.geoms)
                
            logger.info(f"Dissolved into {len(geoms)} geometries")
            df_dissolved = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms))
        except Exception as e:
            logger.error(f"Failed to dissolve geometries: {e}")
            return False
        
        # Explode multi-part geometries
        logger.info("Exploding multi-part geometries")
        df_exploded = df_dissolved.explode()
        logger.info(f"Exploded to {len(df_exploded)} features")
        
        # Prepare geometries for centerline calculation
        logger.info("Preparing geometries for centerline calculation")
        try:
            # Identify very narrow geometries that might cause Qhull errors
            # Calculate width-to-length ratio as a proxy for "narrowness"
            df_exploded['area'] = df_exploded.geometry.area
            df_exploded['perimeter'] = df_exploded.geometry.length
            
            # Quick check for potentially problematic geometries
            # A very low area-to-perimeter ratio often indicates narrow shapes
            narrowness = df_exploded['area'] / (df_exploded['perimeter'] * df_exploded['perimeter'])
            very_narrow_count = (narrowness < 0.01).sum()
            
            if very_narrow_count > 0:
                logger.warning(f"Found {very_narrow_count} potentially narrow geometries that may cause Qhull errors")
                
            # Drop any tiny geometries that will likely cause problems
            tiny_geometries = df_exploded[df_exploded['area'] < 1].shape[0]
            if tiny_geometries > 0:
                logger.warning(f"Removing {tiny_geometries} tiny geometries with area < 1 sq meter")
                df_exploded = df_exploded[df_exploded['area'] >= 1]
                
        except Exception as e:
            logger.warning(f"Error during geometry preparation: {e}")

        # Calculate centerlines
        logger.info("Calculating centerlines (this may take a while)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use a higher memory batch size to help with large geometries
            pandarallel.initialize(progress_bar=True, nb_workers=8, use_memory_fs=True)
            df_exploded['centerlines'] = df_exploded.parallel_apply(lambda row: get_centerline(row), axis=1)
        
        # Drop rows where centerline calculation failed
        centerline_failures = df_exploded['centerlines'].isna().sum()
        if centerline_failures > 0:
            logger.warning(f"Failed to generate centerlines for {centerline_failures} geometries")
            
        logger.info("Dropping geometries with failed centerline calculation")
        df_exploded = df_exploded.dropna(subset=['centerlines'])
        logger.info(f"{len(df_exploded)} features remaining after dropping failed centerlines")
        
        if df_exploded.empty:
            logger.error("No valid centerlines generated, aborting")
            return False
        
        # Extract centerline geometries
        logger.info("Extracting centerline geometries")
        try:
            # Check if all centerlines have the expected structure
            for idx, centerline in df_exploded['centerlines'].items():
                if not hasattr(centerline, 'geometry'):
                    logger.warning(f"Centerline at index {idx} has no geometry attribute")
            
            # Extract centerline geometries with validation
            df_exploded['cl_geom'] = df_exploded['centerlines'].apply(
                lambda x: x.geometry.geoms if hasattr(x, 'geometry') and hasattr(x.geometry, 'geoms') else None)
            
            # Drop rows with None geometry
            null_geom_count = df_exploded['cl_geom'].isna().sum()
            if null_geom_count > 0:
                logger.warning(f"Found {null_geom_count} rows with null geometries after extraction")
                df_exploded = df_exploded.dropna(subset=['cl_geom'])
            
            # Merge linestrings with validation
            logger.info("Merging linestrings")
            def safe_linemerge(geom):
                try:
                    if geom is None:
                        return None
                    result = linemerge(geom)
                    return result
                except Exception as e:
                    logger.warning(f"Error during linemerge: {e}")
                    return None
            
            df_exploded['cl_geom'] = df_exploded['cl_geom'].parallel_apply(safe_linemerge)
            
            # Drop rows with None geometry after linemerge
            null_geom_count = df_exploded['cl_geom'].isna().sum()
            if null_geom_count > 0:
                logger.warning(f"Found {null_geom_count} rows with null geometries after linemerge")
                df_exploded = df_exploded.dropna(subset=['cl_geom'])
            
            # Remove short line ends with validation
            logger.info("Removing short line ends")
            df_exploded['cl_geom'] = df_exploded['cl_geom'].parallel_apply(remove_short_lines)
            
            # Check for empty geometries after processing
            # Use a safer approach to check for empty geometries
            def is_empty_geom(geom):
                try:
                    if geom is None:
                        return True
                    return geom.is_empty
                except Exception:
                    logger.warning(f"Error checking if geometry is empty")
                    return True
            
            empty_count = df_exploded['cl_geom'].apply(is_empty_geom).sum()
            if empty_count > 0:
                logger.warning(f"{empty_count} empty geometries after short line removal")
                df_exploded = df_exploded[~df_exploded['cl_geom'].apply(is_empty_geom)]
            
            logger.info(f"{len(df_exploded)} features remaining after processing centerlines")
            
            if df_exploded.empty:
                logger.error("No valid geometries remaining after centerline processing")
                return False

        except Exception as e:
            logger.error(f"Error during centerline geometry processing: {e}")
            return False
        
        # Simplify geometries with more conservative value
        logger.info("Simplifying geometries")
        df_exploded['cl_geom'] = df_exploded['cl_geom'].parallel_apply(
            lambda row: row.simplify(0.5, preserve_topology=True))
        
        # Get segments
        logger.info("Creating segments")
        df_exploded['segments'] = df_exploded['cl_geom'].parallel_apply(get_segments)
        
        # Check if segments were created successfully
        segment_counts = df_exploded['segments'].apply(len)
        logger.info(f"Created {segment_counts.sum()} segments in total")
        zero_segments = (segment_counts == 0).sum()
        if zero_segments > 0:
            logger.warning(f"{zero_segments} features have zero segments")
            df_exploded = df_exploded[segment_counts > 0]
        
        if df_exploded.empty:
            logger.error("No valid segments generated, aborting")
            return False
            
        df_exploded['centerlines'] = df_exploded['cl_geom']
        
        # Calculate average distances (sidewalk widths)
        logger.info("Calculating sidewalk widths")
        df_exploded['avg_distances'] = df_exploded.parallel_apply(lambda row: get_avg_distances(row), axis=1)
        
        # Check for empty avg_distances
        empty_distances = df_exploded['avg_distances'].apply(lambda x: len(x) == 0).sum()
        if empty_distances > 0:
            logger.warning(f"{empty_distances} features have no distance calculations")
        
        # Create segments dataframe
        logger.info("Creating final segment dataframe")
        data = {'geometry': [], 'width': []}
        
        try:
            for i, row in df_exploded.iterrows():
                segments = row.segments
                distances = row.avg_distances
                
                if len(segments) != len(distances):
                    logger.warning(f"Mismatch in segments ({len(segments)}) and distances ({len(distances)}) for row {i}")
                    continue
                    
                for j, segment in enumerate(segments):
                    if j < len(distances):
                        data['geometry'].append(segment)
                        data['width'].append(distances[j] * 2)
                    else:
                        logger.warning(f"Missing distance for segment {j} in row {i}")
        except Exception as e:
            logger.error(f"Error creating segment dataframe: {e}")
            return False
                
        if not data['geometry']:
            logger.error("No valid segments with widths generated, aborting")
            return False
            
        df_segments = pd.DataFrame(data)
        df_segments = GeoDataFrame(df_segments, crs='EPSG:3627', geometry='geometry')
        logger.info(f"Created dataframe with {len(df_segments)} segments")
        
        # Check for zero or negative widths
        zero_widths = (df_segments['width'] <= 0).sum()
        if zero_widths > 0:
            logger.warning(f"{zero_widths} segments have zero or negative width")
            df_segments = df_segments[df_segments['width'] > 0]
        
        # Convert to NYC projected feet and convert width to feet
        logger.info(f"Converting to {PROJ_FT} and calculating width in feet")
        try:
            df_segments = df_segments.to_crs(PROJ_FT)
            df_segments['width'] = df_segments['width'] * 3.28084  # Convert meters to feet
            
            # Identify unreasonable widths
            very_narrow = (df_segments['width'] < 1).sum()
            very_wide = (df_segments['width'] > 30).sum()
            if very_narrow > 0:
                logger.warning(f"{very_narrow} segments have widths less than 1 foot")
            if very_wide > 0:
                logger.warning(f"{very_wide} segments have widths greater than 30 feet")
        except Exception as e:
            logger.error(f"Error converting to feet: {e}")
            return False
        
        # Output statistics
        width_stats = df_segments['width'].describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        logger.info(f"Sidewalk width statistics (feet):\n{width_stats}")
        
        # Save to file
        logger.info(f"Saving results to {output_path}")
        try:
            df_segments.to_file(output_path, driver='GeoJSON')
        except Exception as e:
            logger.error(f"Failed to save output file: {e}")
            return False
            
        logger.success(f"Successfully generated sidewalk widths and saved to {output_path}")
        logger.info(f"Processed {len(df)} input features into {len(df_segments)} sidewalk segments")
        return True
    except Exception as e:
        logger.error(f"Unhandled exception in process_sidewalks: {e}")
        return False

if __name__ == "__main__":
    # Use constants from io module
    input_file = NYC_OPENDATA_SIDEWALKS
    output_dir = NYC_DATA_PROCESSING_OUTPUT_DIR
    output_file = os.path.join(output_dir, "sidewalkwidths_nyc.geojson")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting sidewalk width calculation process")
    success = process_sidewalks(input_file, output_file)
    
    if success:
        logger.success("Sidewalk width calculation completed successfully")
    else:
        logger.error("Sidewalk width calculation failed")