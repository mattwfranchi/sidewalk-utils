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
import fire

# Import constants and logger
from data.nyc.c import WGS, PROJ_FT
from data.nyc.io import NYC_OPENDATA_SIDEWALKS, NYC_DATA_PROCESSING_OUTPUT_DIR
from geo_processor_base import GeoDataProcessor

# Initialize parallel processing
pandarallel.initialize(progress_bar=True, nb_workers=8)

class SidewalkWidthCalculator(GeoDataProcessor):
    """Tool for calculating sidewalk widths using centerline methods."""
    
    def __init__(self):
        """Initialize the SidewalkWidthCalculator."""
        super().__init__(name=__name__)
        
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

    def linestring_to_segments(self, linestring):
        """Convert LineString to list of segment LineStrings"""
        try:
            if linestring is None or linestring.is_empty:
                return []
                
            if len(linestring.coords) < 2:
                self.logger.warning(f"LineString with insufficient coordinates: {len(linestring.coords)}")
                return []
                
            return [LineString([linestring.coords[i], linestring.coords[i+1]]) for i in range(len(linestring.coords) - 1)]
        except Exception as e:
            self.logger.error(f"Error in linestring_to_segments: {e}")
            return []

    def get_segments(self, line):
        """Extract all segments from a LineString or MultiLineString"""
        try:
            if line is None:
                self.logger.warning("None value passed to get_segments")
                return []
                
            line_segments = []

            if line.geom_type == 'MultiLineString':
                for linestring in line.geoms:
                    segments = self.linestring_to_segments(linestring)
                    line_segments.extend(segments)
            elif line.geom_type == 'LineString':
                line_segments.extend(self.linestring_to_segments(line))
            else:
                self.logger.warning(f"Unexpected geometry type in get_segments: {line.geom_type}")
                
            if not line_segments:
                self.logger.warning("No segments generated")
                
            return line_segments
        except Exception as e:
            self.logger.error(f"Error in get_segments: {e}")
            return []

    def interpolate_by_distance(self, linestring):
        """Interpolate points along a linestring at regular intervals"""
        try:
            if linestring is None or linestring.is_empty:
                return []
                
            distance = 1
            all_points = []
            
            if linestring.length == 0:
                self.logger.warning("Zero-length linestring encountered in interpolation")
                return []
                
            count = round(linestring.length / distance) + 1
            
            if count == 1:
                all_points.append(linestring.interpolate(linestring.length / 2))
            else:
                for i in range(count):
                    all_points.append(linestring.interpolate(distance * i))
            
            return all_points
        except Exception as e:
            self.logger.error(f"Error in interpolate_by_distance: {e}")
            return []

    def interpolate(self, line):
        """Interpolate points along a LineString or MultiLineString"""
        try:
            if line is None:
                self.logger.warning("None value passed to interpolate")
                return MultiPoint([])
                
            if line.is_empty:
                self.logger.warning("Empty geometry passed to interpolate")
                return MultiPoint([])
                
            if line.geom_type == 'MultiLineString':
                all_points = []
                for linestring in line.geoms:
                    all_points.extend(self.interpolate_by_distance(linestring))
                
                if not all_points:
                    self.logger.warning("No points generated during MultiLineString interpolation")
                    
                return MultiPoint(all_points)
                    
            if line.geom_type == 'LineString':
                points = self.interpolate_by_distance(line)
                
                if not points:
                    self.logger.warning("No points generated during LineString interpolation")
                    
                return MultiPoint(points)
                
            self.logger.warning(f"Unexpected geometry type in interpolate: {line.geom_type}")
            return MultiPoint([])
        except Exception as e:
            self.logger.error(f"Error in interpolate: {e}")
            return MultiPoint([])
        
    def polygon_to_multilinestring(self, polygon):
        """Convert polygon to MultiLineString of boundaries"""
        try:
            if polygon is None:
                self.logger.warning("None value passed to polygon_to_multilinestring")
                return None
                
            if polygon.is_empty:
                self.logger.warning("Empty geometry passed to polygon_to_multilinestring")
                return MultiLineString([])
                
            exterior = polygon.exterior
            if exterior is None:
                self.logger.warning("Polygon with no exterior found")
                return MultiLineString([])
                
            return MultiLineString([exterior] + [line for line in polygon.interiors])
        except Exception as e:
            self.logger.error(f"Error in polygon_to_multilinestring: {e}")
            return MultiLineString([])
        
    def get_avg_distances(self, row):
        """Calculate average distances from centerline to sidewalk edges"""
        try:
            avg_distances = []
            
            if row.geometry is None or row.geometry.is_empty:
                self.logger.warning(f"Empty geometry at index {row.name}")
                return []
                
            if not hasattr(row, 'segments') or not row.segments:
                self.logger.warning(f"No segments found at index {row.name}")
                return []
                
            sidewalk_lines = self.polygon_to_multilinestring(row.geometry)
            
            if sidewalk_lines is None or sidewalk_lines.is_empty:
                self.logger.warning(f"Failed to convert polygon to multilinestring at index {row.name}")
                return []
            
            for segment in row.segments:
                points = self.interpolate(segment)
                
                if points is None or points.is_empty or len(points.geoms) == 0:
                    self.logger.warning(f"No interpolation points generated for segment at index {row.name}")
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
                        self.logger.warning(f"Failed to calculate distance: {e}")
                        
                if distances and valid_point_count > 0:
                    avg_distances.append(sum(distances) / valid_point_count)
                else:
                    self.logger.warning(f"No valid distances calculated for segment at index {row.name}")
                    avg_distances.append(0)
                    
            return avg_distances
        except Exception as e:
            self.logger.error(f"Error in get_avg_distances at index {row.name}: {e}")
            return []

    def clamp_sidewalk_widths(self, df_segments, lower_percentile=0.1, upper_percentile=99.9):
        """
        Clamp sidewalk widths to remove extreme outliers
        
        Parameters:
        -----------
        df_segments : GeoDataFrame
            Segments dataframe with 'width' column
        lower_percentile : float
            Lower percentile bound (0-100)
        upper_percentile : float
            Upper percentile bound (0-100)
            
        Returns:
        --------
        GeoDataFrame
            Dataframe with clamped width values
        """
        try:
            if df_segments is None or df_segments.empty:
                self.logger.error("Cannot clamp widths in empty dataframe")
                return df_segments
                
            if 'width' not in df_segments.columns:
                self.logger.error("Width column not found in segments dataframe")
                return df_segments
                
            self.logger.info("Clamping extreme width values to remove outliers")
            
            # Calculate original width distribution statistics for comparison
            # Convert percentiles from 0-100 scale to 0-1 scale for pandas
            original_stats = df_segments['width'].describe([0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999])
            self.logger.info(f"Original width distribution (feet):")
            self.logger.info(f"  - Min: {original_stats['min']:.2f}")
            self.logger.info(f"  - 0.1%: {original_stats['0.1%']:.2f}")  # Use appropriate keys based on describe percentiles
            self.logger.info(f"  - 1%: {original_stats['1%']:.2f}")
            self.logger.info(f"  - 5%: {original_stats['5%']:.2f}")
            self.logger.info(f"  - Median: {original_stats['50%']:.2f}")
            self.logger.info(f"  - 95%: {original_stats['95%']:.2f}")
            self.logger.info(f"  - 99%: {original_stats['99%']:.2f}")
            self.logger.info(f"  - 99.9%: {original_stats['99.9%']:.2f}")
            self.logger.info(f"  - Max: {original_stats['max']:.2f}")
            
            # Use the base class method to perform the clamping
            df_segments = self.clamp_column_values(
                df_segments, 'width', 
                lower_percentile=lower_percentile, 
                upper_percentile=upper_percentile,
                inplace=True,
                track_changes=True
            )
            
            # Report sidewalk widths histogram after clamping
            try:
                hist_values, hist_bins = np.histogram(df_segments['width'], bins=10)
                hist_str = "Width histogram after clamping (feet):\n"
                for i in range(len(hist_values)):
                    bin_start = hist_bins[i]
                    bin_end = hist_bins[i+1]
                    count = hist_values[i]
                    percentage = (count / len(df_segments)) * 100
                    hist_str += f"  - {bin_start:.1f} to {bin_end:.1f}: {count} segments ({percentage:.1f}%)\n"
                self.logger.info(hist_str)
            except Exception as e:
                self.logger.debug(f"Could not generate histogram: {e}")
            
            return df_segments
            
        except Exception as e:
            self.logger.error(f"Error in clamp_sidewalk_widths: {e}")
            return df_segments

    def process(self, input_path=None, output_path=None, clamp_widths=True, 
                lower_percentile=0.5, upper_percentile=99.5):
        """
        Process sidewalk data to extract widths
        
        Args:
            input_path: Path to input sidewalk data (default: from NYC_OPENDATA_SIDEWALKS)
            output_path: Path to save output data (default: derived from input path)
            clamp_widths: Whether to clamp width values to remove outliers (default: True)
            lower_percentile: Lower percentile bound for clamping (default: 0.5)
            upper_percentile: Upper percentile bound for clamping (default: 99.5)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use default paths if not provided
            if input_path is None:
                input_path = NYC_OPENDATA_SIDEWALKS
                
            if output_path is None:
                output_dir = NYC_DATA_PROCESSING_OUTPUT_DIR
                output_path = os.path.join(output_dir, "sidewalkwidths_nyc.parquet")
                
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load sidewalk data using the base class method
            df = self.read_geodataframe(input_path, name="sidewalk data")
            if df is None:
                return False
            
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
                return False
            
            # Dissolve geometries
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
                return False
            
            # Explode multi-part geometries
            self.logger.info("Exploding multi-part geometries")
            df_exploded = df_dissolved.explode()
            self.logger.info(f"Exploded to {len(df_exploded)} features")
            
            # Prepare geometries for centerline calculation
            self.logger.info("Preparing geometries for centerline calculation")
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
                # Use a higher memory batch size to help with large geometries
                pandarallel.initialize(progress_bar=True, nb_workers=8, use_memory_fs=True)
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
                return False
            
            # Extract centerline geometries
            self.logger.info("Extracting centerline geometries")
            try:
                # Check if all centerlines have the expected structure
                for idx, centerline in df_exploded['centerlines'].items():
                    if not hasattr(centerline, 'geometry'):
                        self.logger.warning(f"Centerline at index {idx} has no geometry attribute")
                
                # Extract centerline geometries with validation
                df_exploded['cl_geom'] = df_exploded['centerlines'].apply(
                    lambda x: x.geometry.geoms if hasattr(x, 'geometry') and hasattr(x.geometry, 'geoms') else None)
                
                # Drop rows with None geometry
                null_geom_count = df_exploded['cl_geom'].isna().sum()
                if null_geom_count > 0:
                    self.logger.warning(f"Found {null_geom_count} rows with null geometries after extraction")
                    df_exploded = df_exploded.dropna(subset=['cl_geom'])
                
                # Merge linestrings with validation
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
                
                # Remove short line ends with validation
                self.logger.info("Removing short line ends")
                df_exploded['cl_geom'] = df_exploded['cl_geom'].parallel_apply(
                    lambda row: self.remove_short_lines(row))
                
                # Check for empty geometries after processing
                # Use a safer approach to check for empty geometries
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
                    return False

            except Exception as e:
                self.logger.error(f"Error during centerline geometry processing: {e}")
                return False
            
            # Simplify geometries with more conservative value
            self.logger.info("Simplifying geometries")
            df_exploded['cl_geom'] = df_exploded['cl_geom'].parallel_apply(
                lambda row: row.simplify(0.5, preserve_topology=True))
            
            # Get segments
            self.logger.info("Creating segments")
            df_exploded['segments'] = df_exploded['cl_geom'].parallel_apply(
                lambda row: self.get_segments(row))
            
            # Check if segments were created successfully
            segment_counts = df_exploded['segments'].apply(len)
            self.logger.info(f"Created {segment_counts.sum()} segments in total")
            zero_segments = (segment_counts == 0).sum()
            if zero_segments > 0:
                self.logger.warning(f"{zero_segments} features have zero segments")
                df_exploded = df_exploded[segment_counts > 0]
            
            if df_exploded.empty:
                self.logger.error("No valid segments generated, aborting")
                return False
                
            df_exploded['centerlines'] = df_exploded['cl_geom']
            
            # Calculate average distances (sidewalk widths)
            self.logger.info("Calculating sidewalk widths")
            df_exploded['avg_distances'] = df_exploded.parallel_apply(
                lambda row: self.get_avg_distances(row), axis=1)
            
            # Check for empty avg_distances
            empty_distances = df_exploded['avg_distances'].apply(lambda x: len(x) == 0).sum()
            if empty_distances > 0:
                self.logger.warning(f"{empty_distances} features have no distance calculations")
            
            # Create segments dataframe
            self.logger.info("Creating final segment dataframe")
            data = {'geometry': [], 'width': []}
            
            try:
                for i, row in df_exploded.iterrows():
                    segments = row.segments
                    distances = row.avg_distances
                    
                    if len(segments) != len(distances):
                        self.logger.warning(f"Mismatch in segments ({len(segments)}) and distances ({len(distances)}) for row {i}")
                        continue
                        
                    for j, segment in enumerate(segments):
                        if j < len(distances):
                            data['geometry'].append(segment)
                            data['width'].append(distances[j] * 2)
                        else:
                            self.logger.warning(f"Missing distance for segment {j} in row {i}")
            except Exception as e:
                self.logger.error(f"Error creating segment dataframe: {e}")
                return False
                    
            if not data['geometry']:
                self.logger.error("No valid segments with widths generated, aborting")
                return False
                
            df_segments = pd.DataFrame(data)
            df_segments = GeoDataFrame(df_segments, crs='EPSG:3627', geometry='geometry')
            self.logger.info(f"Created dataframe with {len(df_segments)} segments")
            
            # Check for zero or negative widths
            zero_widths = (df_segments['width'] <= 0).sum()
            if zero_widths > 0:
                self.logger.warning(f"{zero_widths} segments have zero or negative width")
                df_segments = df_segments[df_segments['width'] > 0]
            
            # Convert to NYC projected feet and convert width to feet
            self.logger.info(f"Converting to {PROJ_FT} and calculating width in feet")
            try:
                df_segments = df_segments.to_crs(PROJ_FT)
                df_segments['width'] = df_segments['width'] * 3.28084  # Convert meters to feet
                
                # Identify unreasonable widths
                very_narrow = (df_segments['width'] < 1).sum()
                very_wide = (df_segments['width'] > 30).sum()
                if very_narrow > 0:
                    self.logger.warning(f"{very_narrow} segments have widths less than 1 foot")
                if very_wide > 0:
                    self.logger.warning(f"{very_wide} segments have widths greater than 30 feet")
            except Exception as e:
                self.logger.error(f"Error converting to feet: {e}")
                return False
            
            # Output statistics before clamping
            width_stats = df_segments['width'].describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            self.logger.info(f"Sidewalk width statistics before clamping (feet):\n{width_stats}")
            
            # Apply width clamping if enabled
            if clamp_widths:
                df_segments = self.clamp_sidewalk_widths(
                    df_segments, 
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile
                )
                
                # If clamping failed, log a warning but continue
                if df_segments is None or df_segments.empty:
                    self.logger.error("Width clamping failed, aborting")
                    return False
            else:
                self.logger.info("Width clamping is disabled")
            
            # Save output using the base class method for GeoParquet
            self.logger.info(f"Saving results to {output_path}")
            save_success = self.save_geoparquet(df_segments, output_path)
            if not save_success:
                return False
                
            self.logger.success(f"Successfully generated sidewalk widths and saved to {output_path}")
            self.logger.info(f"Processed {len(df)} input features into {len(df_segments)} sidewalk segments")
            return True
        except Exception as e:
            self.logger.error(f"Unhandled exception in process method: {e}")
            return False

if __name__ == "__main__":
    # Use the Fire CLI library to expose the SidewalkWidthCalculator class
    fire.Fire(SidewalkWidthCalculator)

# Example command line usage:
# python sidewalk-widths.py process \
#   --input_path="../data/sidewalks.geojson" \
#   --output_path="../data/sidewalkwidths_nyc.parquet"