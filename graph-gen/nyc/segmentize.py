import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import warnings
import fire
import numpy as np

# Import constants and logger
from data.nyc.c import PROJ_FT
from data.nyc.io import NYC_DATA_PROCESSING_OUTPUT_DIR
from geo_processor_base import GeoDataProcessor

class SidewalkSegmentizer(GeoDataProcessor):
    """Tool for segmentizing sidewalk geometries into points for analysis."""

    def __init__(self):
        """Initialize the SidewalkSegmentizer with its own logger."""
        super().__init__(name=__name__)

    def clip_to_neighborhood(self, data_gdf, nta_gdf, neighborhood_name):
        """
        Clip geodataframe to a specific neighborhoodgi
        
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
                available_neighborhoods = nta_gdf.NTAName.unique().tolist()
                self.logger.info(f"Available neighborhoods: {available_neighborhoods}")
                return None
                
            # Log neighborhood information
            nta_area = target_nta.geometry.area.iloc[0]
            nta_bounds = target_nta.total_bounds
            self.logger.info(f"Neighborhood area: {nta_area:.2f} square units")
            self.logger.info(f"Neighborhood bounds [minx, miny, maxx, maxy]: {nta_bounds}")
            
            # Calculate initial data stats for comparison
            initial_count = len(data_gdf)
            initial_length = data_gdf.geometry.length.sum() if hasattr(data_gdf.geometry.iloc[0], 'length') else None
            
            # Perform spatial join
            try:
                self.logger.info(f"Performing spatial join operation")
                clipped_data = gpd.sjoin(data_gdf, target_nta, predicate='within')
                
                # Calculate stats after clipping for comparison
                final_count = len(clipped_data)
                feature_reduction = ((initial_count - final_count) / initial_count) * 100
                self.logger.info(f"Clipped data contains {final_count} features "
                              f"({feature_reduction:.1f}% reduction from original)")
                
                if initial_length is not None:
                    final_length = clipped_data.geometry.length.sum()
                    length_reduction = ((initial_length - final_length) / initial_length) * 100
                    self.logger.info(f"Total length reduced from {initial_length:.2f} to {final_length:.2f} units "
                                  f"({length_reduction:.1f}% reduction)")
                
                # Check if clipping produced an empty result
                if clipped_data.empty:
                    self.logger.warning("Clipping resulted in empty dataset. Check if the data actually "
                                     f"intersects with the {neighborhood_name} neighborhood.")
                
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
            
            # Gather stats before simplification
            initial_vertex_count = sum(len(g.coords) if hasattr(g, 'coords') 
                                   else sum(len(geom.coords) for geom in g.geoms) 
                                   for g in gdf.geometry)
            initial_lengths = gdf.geometry.length
            
            # Create a copy to avoid modifying the original
            simplified = gdf.copy()
            
            # Check geometry validity
            invalid_count = simplified[~simplified.geometry.is_valid].shape[0]
            if invalid_count > 0:
                self.logger.warning(f"Found {invalid_count} invalid geometries ({invalid_count/len(gdf)*100:.1f}%), attempting to fix")
                simplified.geometry = simplified.geometry.buffer(0)
                
            # Apply simplification
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                simplified.geometry = simplified.geometry.simplify(tolerance)
                
            # Gather stats after simplification
            final_vertex_count = sum(len(g.coords) if hasattr(g, 'coords') 
                                 else sum(len(geom.coords) for geom in g.geoms) 
                                 for g in simplified.geometry)
            vertex_reduction = ((initial_vertex_count - final_vertex_count) / initial_vertex_count) * 100
            self.logger.info(f"Vertices reduced from {initial_vertex_count} to {final_vertex_count} "
                          f"({vertex_reduction:.1f}% reduction)")
            
            # Check how much the geometry lengths changed
            final_lengths = simplified.geometry.length
            length_change = ((final_lengths - initial_lengths) / initial_lengths * 100).mean()
            self.logger.info(f"Average length change after simplification: {length_change:.2f}%")
            
            # Validate results
            error_count = simplified[~simplified.geometry.is_valid].shape[0]
            if error_count > 0:
                self.logger.warning(f"{error_count} geometries are invalid after simplification "
                                 f"({error_count/len(simplified)*100:.1f}%)")
                
            empty_count = simplified[simplified.geometry.is_empty].shape[0]
            if empty_count > 0:
                self.logger.warning(f"{empty_count} geometries are empty after simplification "
                                 f"({empty_count/len(simplified)*100:.1f}%)")
                simplified = simplified[~simplified.geometry.is_empty]
                
            self.logger.info(f"Simplification complete, {len(simplified)} valid features remain")
            return simplified
            
        except Exception as e:
            self.logger.error(f"Error in simplify_geometries: {e}")
            return None

    def segmentize_and_extract_points(self, gdf, distance=50):
        """
        Segmentize geometries and extract unique points
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input geodataframe
        distance : float
            Distance between points in segmentation
            
        Returns:
        --------
        GeoDataFrame
            Geodataframe with point geometries
        """
        try:
            if gdf is None or gdf.empty:
                self.logger.error("Cannot segmentize empty geodataframe")
                return None
                
            self.logger.info(f"Segmentizing geometries at {distance} foot intervals")
            
            # Log input geometry statistics
            total_length = gdf.geometry.length.sum()
            avg_length = gdf.geometry.length.mean()
            expected_points = int(total_length / distance)
            self.logger.info(f"Total length to segmentize: {total_length:.2f} units")
            self.logger.info(f"Average feature length: {avg_length:.2f} units")
            self.logger.info(f"Expected point count (estimate): ~{expected_points} points")
            
            try:
                # Segmentize and extract points
                segmentized = gdf.segmentize(distance).extract_unique_points()
                
                # Explode multipoint geometries into individual points
                self.logger.info("Exploding multipoint geometries")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    segmentized = segmentized.explode(index_parts=True)
                    
                actual_points = len(segmentized)
                point_ratio = actual_points / expected_points
                self.logger.info(f"Generated {actual_points} points ({point_ratio:.2f}x the estimate)")
                
                # Log spatial distribution stats
                if len(segmentized) > 0:
                    bounds = segmentized.total_bounds
                    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                    point_density = actual_points / area if area > 0 else 0
                    self.logger.info(f"Point density: {point_density:.6f} points per square unit")
                    
                    # Calculate nearest-neighbor distances
                    try:
                        sample_size = min(1000, len(segmentized))
                        if sample_size > 10:  # Only calculate for reasonably sized samples
                            sample = segmentized.sample(sample_size) if sample_size < len(segmentized) else segmentized
                            distances = []
                            for idx, point in sample.iterrows():
                                if len(sample) > 1:  # Need at least 2 points
                                    others = sample[sample.index != idx]
                                    min_dist = others.distance(point.geometry).min()
                                    distances.append(min_dist)
                            
                            if distances:
                                avg_nn_dist = sum(distances) / len(distances)
                                self.logger.info(f"Average nearest neighbor distance (sample): {avg_nn_dist:.2f} units")
                                self.logger.info(f"Min/Max nearest neighbor distance: {min(distances):.2f}/{max(distances):.2f} units")
                    except Exception as e:
                        self.logger.warning(f"Could not calculate point distribution stats: {e}")
                
                if segmentized.empty:
                    self.logger.error("No points were generated during segmentization")
                    return None
                    
                return segmentized
                
            except Exception as e:
                self.logger.error(f"Error during segmentization: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in segmentize_and_extract_points: {e}")
            return None

    def prepare_segmentized_dataframe(self, segmentized, source_gdf):
        """
        Prepare the final segmentized dataframe with attributes
        
        Parameters:
        -----------
        segmentized : GeoDataFrame
            Segmentized point geometries
        source_gdf : GeoDataFrame
            Original geodataframe with attributes
            
        Returns:
        --------
        GeoDataFrame
            Final geodataframe with points and attributes
        """
        try:
            if segmentized is None or segmentized.empty:
                self.logger.error("Segmentized data is empty")
                return None
                
            if source_gdf is None or source_gdf.empty:
                self.logger.error("Source data is empty")
                return None
                
            self.logger.info("Preparing final segmentized dataframe")
            self.logger.info(f"Source data contains {len(source_gdf)} features with {len(source_gdf.columns)} attributes")
            
            # Reset index to get level_0 and level_1 columns - exactly as in the notebook
            self.logger.info("Resetting index to get level_0 and level_1 columns")
            segmentized_df = gpd.GeoDataFrame(segmentized).reset_index()
            
            # Log columns for debugging
            self.logger.info(f"Columns after reset_index: {segmentized_df.columns.tolist()}")
            
            # Following the exact notebook pattern
            self.logger.info("Merging with original dataframe and handling geometry column")
            
            # 1. Merge using level_0 with the original dataframe, as in notebook
            result = segmentized_df.merge(
                source_gdf, 
                left_on='level_0',
                right_index=True
            )
            
            # 2. Drop level_0, level_1, and the original geometry column from source_gdf
            result = result.drop(columns=['level_0', 'level_1', 'geometry'])
            
            # 3. Set the first column (point geometry) as 'geometry'
            result['geometry'] = result.iloc[:, 0]
            
            # 4. Drop the first column (now redundant)
            result = result.drop(columns=[result.columns[0]])
            
            # 5. Create a new GeoDataFrame with explicit geometry column
            result = gpd.GeoDataFrame(result, geometry='geometry', crs=segmentized.crs)
            
            # Log results
            self.logger.info(f"Final dataframe has {len(result)} points with {len(result.columns)-1} attributes")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in prepare_segmentized_dataframe: {e}")
            self.logger.error(f"Error details: {str(e)}")
            return None

    def process(self, input_path, output_path=None, nta_path=None, neighborhood=None, segmentation_distance=50):
        """
        Process sidewalk data to segmentize into points
        
        Args:
            input_path: Path to input sidewalk data
            output_path: Path to save output data (default: derived from input path)
            nta_path: Path to neighborhood data (optional)
            neighborhood: Name of neighborhood to focus on (optional)
            segmentation_distance: Distance between points in segmentation, feet (default: 50)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Log processing parameters
            self.logger.info(f"Processing parameters:")
            self.logger.info(f"  - Input path: {input_path}")
            self.logger.info(f"  - Neighborhood filter: {'Yes, ' + neighborhood if neighborhood else 'No'}")
            self.logger.info(f"  - Segmentation distance: {segmentation_distance} feet")
            
            # Set default output path if not provided
            if output_path is None:
                input_dir = os.path.dirname(input_path)
                input_basename = os.path.basename(input_path).split('.')[0]
                output_path = os.path.join(input_dir, f"{input_basename}_segmentized.parquet")
                self.logger.info(f"Using default output path: {output_path}")
            else:
                self.logger.info(f"Output path: {output_path}")
            
            self.logger.info("Starting sidewalk segmentization process")
            start_time = pd.Timestamp.now()
            
            # Load sidewalk data
            use_sidewalk_widths = "sidewalkwidths" in input_path.lower()
            self.logger.info(f"Data source identified as: {'sidewalk widths' if use_sidewalk_widths else 'standard sidewalks'}")
            sidewalks = self.read_geodataframe(
                input_path, 
                crs=PROJ_FT if use_sidewalk_widths else None, 
                name="sidewalk data"
            )
            
            if sidewalks is None:
                return False
            
            # Load neighborhood data and crop if requested
            if nta_path and neighborhood:
                # Load neighborhood boundaries
                nta_gdf = self.read_geodataframe(nta_path, crs=PROJ_FT, name="neighborhood data")
                
                if nta_gdf is None:
                    self.logger.warning("Could not load neighborhood data, proceeding with full dataset")
                else:
                    # Crop to neighborhood of interest
                    cropped_sidewalks = self.clip_to_neighborhood(sidewalks, nta_gdf, neighborhood)
                    
                    if cropped_sidewalks is None:
                        self.logger.warning("Could not crop to neighborhood, proceeding with full dataset")
                    else:
                        self.logger.info(f"Working with {len(cropped_sidewalks)} features in {neighborhood}")
                        sidewalks = cropped_sidewalks
            
            # Simplify geometries if not using sidewalk widths (which are already simplified)
            if not use_sidewalk_widths:
                self.logger.info("Simplifying geometries")
                sidewalks = self.simplify_geometries(sidewalks, tolerance=10)
                if sidewalks is None:
                    return False
            
            # Segmentize and extract points
            segmentized = self.segmentize_and_extract_points(sidewalks, distance=segmentation_distance)
            if segmentized is None:
                return False
                
            # Prepare final dataframe with attributes
            result = self.prepare_segmentized_dataframe(segmentized, sidewalks)
            if result is None:
                return False
            
            # Ensure result is in the correct CRS
            self.logger.info(f"Ensuring output is in {PROJ_FT}")
            prev_crs = result.crs
            result = self.ensure_crs(result, PROJ_FT)
            if result is None:
                return False
                
            if prev_crs != PROJ_FT:
                self.logger.info(f"CRS transformed from {prev_crs} to {PROJ_FT}")
                bounds = result.total_bounds
                self.logger.info(f"Output bounds [minx, miny, maxx, maxy]: {bounds}")
            
            # Save output
            self.logger.info(f"Saving {len(result)} segmentized points to {output_path}")
            success = self.save_geoparquet(result, output_path)
            if not success:
                return False
            
            # Final stats and summary
            end_time = pd.Timestamp.now()
            elapsed_time = (end_time - start_time).total_seconds()
            points_per_second = len(result) / elapsed_time if elapsed_time > 0 else 0
            
            self.logger.info(f"Processing statistics:")
            self.logger.info(f"  - Input features: {len(sidewalks)}")
            self.logger.info(f"  - Output points: {len(result)}")
            self.logger.info(f"  - Points-to-feature ratio: {len(result)/len(sidewalks):.1f}")
            self.logger.info(f"  - Processing time: {elapsed_time:.1f} seconds")
            self.logger.info(f"  - Processing speed: {points_per_second:.1f} points/second")
            
            self.logger.success("Sidewalk segmentization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Unhandled exception in process method: {e}")
            return False


if __name__ == "__main__":
    # Use the Fire CLI library to expose the SidewalkSegmentizer class
    fire.Fire(SidewalkSegmentizer)

# Example command line usage:
# python segmentize.py process \
#   --input_path="../data/sidewalkwidths_nyc.geojson" \
#   --output_path="../data/segmentized_nyc_sidewalks.parquet" \
#   --nta_path="../data/nynta2020_24b/nynta2020.shp" \
#   --neighborhood="Greenpoint" \
#   --segmentation_distance=50