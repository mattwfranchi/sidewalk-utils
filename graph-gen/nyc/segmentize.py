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

from functools import partial 
from shapely.strtree import STRtree
import multiprocessing

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
                            for idx, point in sample.items():
                                if len(sample) > 1:  # Need at least 2 points
                                    others = sample[sample.index != idx]
                                    # FIX: Use point directly, not point.geometry
                                    min_dist = others.distance(point).min()
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

    def prepare_segmentized_dataframe(self, segmentized_points, source_gdf):
        """
        Prepare the final segmentized dataframe with attributes - FIXED version
        
        Parameters:
        -----------
        segmentized_points : GeoDataFrame
            Segmentized point geometries
        source_gdf : GeoDataFrame
            Original geodataframe with attributes
            
        Returns:
        --------
        GeoDataFrame
            Final geodataframe with points and attributes
        """
        try:
            if segmentized_points is None or segmentized_points.empty:
                self.logger.error("Segmentized data is empty")
                return None
                
            if source_gdf is None or source_gdf.empty:
                self.logger.error("Source data is empty")
                return None
                
            self.logger.info("Preparing final segmentized dataframe")
            self.logger.info(f"Source data contains {len(source_gdf)} features with {len(source_gdf.columns)} attributes")
            
            # Reset index to get level_0 and level_1 columns
            self.logger.info("Resetting index to get level_0 and level_1 columns")
            segmentized_df = segmentized_points.reset_index()
            
            # Log columns for debugging
            self.logger.info(f"Columns after reset_index: {segmentized_df.columns.tolist()}")
            
            # In your case, the geometry column is named 0 (integer)
            self.logger.info("Explicitly handling column named 0 as geometry column")
            
            # Save the geometries from column 0
            if 0 in segmentized_df.columns:
                geometries = segmentized_df[0].copy()
                self.logger.info(f"Saved {len(geometries)} geometries from column 0")
            else:
                self.logger.error("Column 0 not found in segmentized dataframe")
                return None
            
            # Following the exact notebook pattern but with explicit handling of column 0
            self.logger.info("Merging with original dataframe and handling geometry column")
            
            # 1. Merge using level_0 with the original dataframe
            try:
                result = segmentized_df.merge(
                    source_gdf, 
                    left_on='level_0',
                    right_index=True
                )
                self.logger.info(f"Merged dataframe has {len(result)} rows and {len(result.columns)} columns")
            except Exception as e:
                self.logger.error(f"Merge failed: {e}")
                return None
            
            # 2. Drop level_0, level_1, and the original geometry column from source_gdf
            drop_cols = ['level_0', 'level_1']
            if 'geometry' in result.columns:
                drop_cols.append('geometry')
            result = result.drop(columns=drop_cols, errors='ignore')
            
            # 3. Drop column 0 (we've saved the geometries)
            if 0 in result.columns:
                result = result.drop(columns=[0])
            
            # 4. Add the saved geometries as 'geometry' column
            result['geometry'] = geometries
            
            # 5. Create a new GeoDataFrame with explicit geometry column
            result = gpd.GeoDataFrame(result, geometry='geometry', crs=segmentized_points.crs)
            
            # Log results
            self.logger.info(f"Final dataframe has {len(result)} points with {len(result.columns)-1} attributes")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in prepare_segmentized_dataframe: {e}")
            self.logger.error(f"Error details: {str(e)}")
            return None
    def compute_adjacency(self, gdf, tolerance=0.1):
        """
        Compute adjacency relationships between sidewalk geometries using vectorized operations
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input geodataframe with sidewalk geometries
        tolerance : float
            Distance tolerance for considering geometries adjacent (in feet)
            
        Returns:
        --------
        GeoDataFrame
            Input geodataframe with additional adjacency columns
        """
        try:
            if gdf is None or gdf.empty:
                self.logger.error("Cannot compute adjacency for empty geodataframe")
                return None
                
            self.logger.info(f"Computing adjacency relationships with tolerance {tolerance}")
            
            # Create a spatial index
            self.logger.info("Building spatial index")
            sindex = gdf.sindex
            
            # Use the nearest method with max_distance to find adjacent geometries
            self.logger.info(f"Finding adjacent features with vectorized nearest operation (max_distance={tolerance})")
            nearest_indices = sindex.nearest(
                gdf.geometry, 
                return_all=True,  # Get all equidistant neighbors
                max_distance=tolerance  # Only find neighbors within our tolerance
            )
            
            # Convert results to adjacency dictionary
            self.logger.info("Processing adjacency results")
            adjacency_dict = {idx: [] for idx in gdf.index}
            
            # Process the nearest indices
            input_indices, tree_indices = nearest_indices
            
            # Use numpy operations for speed
            input_idx_values = gdf.index.values[input_indices]
            tree_idx_values = gdf.index.values[tree_indices]
            
            # Track progress
            total_pairs = len(input_indices)
            self.logger.info(f"Processing {total_pairs} potential adjacency pairs")
            
            # Process all pairs at once
            for i in range(len(input_indices)):
                source_idx = input_idx_values[i]
                target_idx = tree_idx_values[i]
                
                # Don't add self-adjacency
                if source_idx != target_idx:
                    adjacency_dict[source_idx].append(target_idx)
            
            # Add adjacency information to the dataframe
            self.logger.info("Adding adjacency information to dataframe")
            gdf['adjacent_ids'] = gdf.index.map(adjacency_dict)
            gdf['adjacency_count'] = gdf['adjacent_ids'].apply(len)
            
            # Log adjacency statistics
            feature_count = len(gdf)
            avg_adjacency = gdf['adjacency_count'].mean()
            max_adjacency = gdf['adjacency_count'].max()
            isolated_count = (gdf['adjacency_count'] == 0).sum()
            
            self.logger.info(f"Adjacency statistics:")
            self.logger.info(f"  - Average adjacent features: {avg_adjacency:.2f}")
            self.logger.info(f"  - Maximum adjacent features: {max_adjacency}")
            self.logger.info(f"  - Isolated features: {isolated_count} ({isolated_count/feature_count*100:.1f}%)")
            
            return gdf
            
        except Exception as e:
            self.logger.error(f"Error in compute_adjacency: {e}")
            return None

    def cleanup(self, segmentized_points, og_sidewalk_file_path: str):
        try:
            # Load RAPIDS libraries
            import cudf
            import cuspatial
            from shapely.geometry import Point

            self.logger.info("Using RAPIDS/cuSpatial for cleanup")

            # Load original sidewalks and convert to RAPIDS GeoSeries
            og_sidewalks = gpd.read_file(og_sidewalk_file_path).to_crs(PROJ_FT)['geometry']
            og_sidewalks_cuspatial = cuspatial.from_geopandas(og_sidewalks)

            # Convert segmentized points to RAPIDS GeoSeries
            segmentized_points_cuspatial = cuspatial.from_geopandas(segmentized_points)

            # Perform intersection check using RAPIDS
            self.logger.info("Performing intersection check on GPU")
            intersects_mask = cuspatial.point_in_polygon(
                segmentized_points_cuspatial.points.xy,
                og_sidewalks_cuspatial.polygons.xy
            )

            # Filter points based on intersection results
            len_before = len(segmentized_points)
            segmentized_points = segmentized_points[intersects_mask.to_pandas()]
            len_after = len(segmentized_points)

            self.logger.info(f"Segmentized points cleaned up, {len_before} -> {len_after}")
            return segmentized_points

        except ImportError as e:
            self.logger.warning(f"RAPIDS/cuSpatial not available, falling back to CPU implementation: {e}")
            return self.cleanup_cpu(segmentized_points, og_sidewalk_file_path)

        except Exception as e:
            self.logger.error(f"Error in cleanup with RAPIDS/cuSpatial: {e}")
            return None

    def compute_point_adjacency(self, segmentized_points, source_gdf, point_distance_threshold=10, batch_size=1000):
        """
        Compute adjacency using GPU acceleration via RAPIDS libraries with optimized 
        implementation for the standard cuSpatial API (no special cross-distance function).
        
        Parameters:
        -----------
        segmentized_points : GeoDataFrame
            The segmentized points with level_0 indicating the parent segment
        source_gdf : GeoDataFrame
            The original segments with adjacent_ids column from previous computation
        point_distance_threshold : float
            Maximum distance between points to be considered adjacent (in feet)
        batch_size : int
            Size of segment pair batches to process (adjust based on GPU memory)
            
        Returns:
        --------
        GeoDataFrame
            Points with point-level adjacency information
        """
        try:
            # Try to import RAPIDS libraries
            try:
                import cudf
                import cuspatial
                from shapely.geometry import Point
                has_rapids = True
                self.logger.info("RAPIDS libraries found, using GPU acceleration")
            except ImportError:
                has_rapids = False
                self.logger.warning("RAPIDS libraries not found, falling back to CPU implementation")
                return self.compute_point_adjacency_parallel(segmentized_points, source_gdf, point_distance_threshold)
                
            if 'adjacent_ids' not in source_gdf.columns:
                self.logger.warning("No adjacency information found in source data, skipping point adjacency")
                return segmentized_points
                
            self.logger.info("Computing point-level adjacency relationships with GPU acceleration")
            self.logger.info(f"Using point distance threshold of {point_distance_threshold} feet")
            
            # Reset index to get segment ID as level_0 column
            points_df = segmentized_points.reset_index()
            
            # Set the active geometry column explicitly
            if 0 in points_df.columns:
                points_df = points_df.set_geometry(0)
                geometry_column = 0  
            else:
                self.logger.error("Could not find geometry column in segmentized points")
                return segmentized_points
                
            # Pre-process data for faster lookup
            self.logger.info("Building segment-to-points mapping for faster lookup")
            segment_to_points = {}
            for idx, row in points_df.iterrows():
                segment_id = row['level_0']
                if segment_id not in segment_to_points:
                    segment_to_points[segment_id] = []
                segment_to_points[segment_id].append((idx, row[geometry_column]))
            
            # Get all segment pairs that need processing
            segment_pairs = []
            for segment_id in source_gdf.index:
                if segment_id not in segment_to_points:
                    continue
                    
                adjacent_segment_ids = source_gdf.loc[segment_id, 'adjacent_ids']
                if not adjacent_segment_ids:
                    continue
                    
                for adj_segment_id in adjacent_segment_ids:
                    if adj_segment_id in segment_to_points:
                        segment_pairs.append((segment_id, adj_segment_id))
            
            self.logger.info(f"Found {len(segment_pairs)} segment pairs to process")
            
            # Create a set to collect adjacency pairs 
            # Using a set for faster duplicate checking and to ensure unique pairs
            all_adjacency_pairs = set()
            
            # Process segment pairs in batches to avoid GPU memory issues
            total_pairs = len(segment_pairs)
            
            # For large collections, use vectorized operation if possible
            try:
                # Try to find an optimal vectorized way to compute distances
                import inspect
                if hasattr(cuspatial, "pairwise_point_distance"):
                    func_sig = inspect.signature(cuspatial.pairwise_point_distance)
                    self.logger.info(f"Found pairwise_point_distance with signature: {func_sig}")
                    has_vectorized = True
                else:
                    has_vectorized = False
                    self.logger.info("No vectorized point distance function found")
            except Exception:
                has_vectorized = False
            
            # Process in batches
            for batch_start in range(0, total_pairs, batch_size):
                batch_end = min(batch_start + batch_size, total_pairs)
                batch = segment_pairs[batch_start:batch_end]
                
                self.logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_pairs + batch_size - 1)//batch_size}")
                
                # Process each segment pair in the batch
                for i in range(len(batch)):
                    segment_id, adj_segment_id = batch[i]
                    
                    # Get points from each segment
                    segment_points = segment_to_points.get(segment_id, [])
                    adj_segment_points = segment_to_points.get(adj_segment_id, [])
                    
                    if not segment_points or not adj_segment_points:
                        continue
                    
                    # Extract point coordinates and indices
                    seg1_indices = [idx for idx, _ in segment_points]
                    seg1_points = [geom for _, geom in segment_points]
                    
                    seg2_indices = [idx for idx, _ in adj_segment_points]
                    seg2_points = [geom for _, geom in adj_segment_points]
                    
                    # CPU-based distance calculation is faster for small point sets
                    # GPU acceleration is better for larger sets
                    if len(seg1_points) * len(seg2_points) < 100:
                        # For small sets, just use direct calculation
                        for i, (idx1, geom1) in enumerate(segment_points):
                            for j, (idx2, geom2) in enumerate(adj_segment_points):
                                if geom1.distance(geom2) <= point_distance_threshold:
                                    all_adjacency_pairs.add((idx1, idx2))
                                    all_adjacency_pairs.add((idx2, idx1))  # Add bidirectional
                    else:
                        # For larger sets, try using GPU acceleration
                        try:
                            # Create GPU-accelerated GeoSeries
                            gpu_points1 = cuspatial.GeoSeries(seg1_points)
                            gpu_points2 = cuspatial.GeoSeries(seg2_points)
                            
                            # Vectorized approach for large point collections
                            if has_vectorized and len(seg1_points) == len(seg2_points):
                                try:
                                    # Try vectorized distance if points have 1:1 correspondence
                                    self.logger.info("Attempting vectorized distance calculation")
                                    distances = cuspatial.pairwise_point_distance(gpu_points1, gpu_points2)
                                    
                                    # Find points within threshold
                                    mask = distances.to_pandas() <= point_distance_threshold
                                    for i in range(len(mask)):
                                        if mask.iloc[i]:
                                            all_adjacency_pairs.add((seg1_indices[i], seg2_indices[i]))
                                            all_adjacency_pairs.add((seg2_indices[i], seg1_indices[i]))
                                    continue
                                except Exception as e:
                                    self.logger.warning(f"Vectorized approach failed: {e}")
                                    # Fall through to one-by-one approach
                            
                            # Convert to pandas for faster processing
                            gpu_points1_pd = gpu_points1.to_pandas()
                            gpu_points2_pd = gpu_points2.to_pandas()
                            
                            # Process in sub-batches for better GPU utilization
                            sub_batch_size = 100
                            for i_start in range(0, len(seg1_points), sub_batch_size):
                                i_end = min(i_start + sub_batch_size, len(seg1_points))
                                
                                for j_start in range(0, len(seg2_points), sub_batch_size):
                                    j_end = min(j_start + sub_batch_size, len(seg2_points))
                                    
                                    # Process this sub-batch
                                    for i in range(i_start, i_end):
                                        point1 = gpu_points1_pd.iloc[i]
                                        idx1 = seg1_indices[i]
                                        
                                        for j in range(j_start, j_end):
                                            point2 = gpu_points2_pd.iloc[j]
                                            idx2 = seg2_indices[j]
                                            
                                            # Use Shapely's distance
                                            if point1.distance(point2) <= point_distance_threshold:
                                                all_adjacency_pairs.add((idx1, idx2))
                                                all_adjacency_pairs.add((idx2, idx1))
                        
                        except Exception as e:
                            self.logger.warning(f"GPU distance calculation failed: {e}")
                            self.logger.warning("Falling back to CPU calculation for this segment pair")
                            
                            # CPU fallback for this segment pair
                            for i, (idx1, geom1) in enumerate(segment_points):
                                for j, (idx2, geom2) in enumerate(adj_segment_points):
                                    if geom1.distance(geom2) <= point_distance_threshold:
                                        all_adjacency_pairs.add((idx1, idx2))
                                        all_adjacency_pairs.add((idx2, idx1))
                
                # Log progress
                progress = min(100, int((batch_end / total_pairs) * 100))
                self.logger.info(f"Processing: {progress}% complete ({batch_end}/{total_pairs} pairs)")
                        
            # Convert set to list and build adjacency dictionary
            self.logger.info(f"Building final adjacency structure from {len(all_adjacency_pairs)//2} unique point pairs")
            point_adjacency = {idx: [] for idx in points_df.index}
            
            for point1, point2 in all_adjacency_pairs:
                if point2 not in point_adjacency[point1]:
                    point_adjacency[point1].append(point2)
            
            # Add adjacency information to the points dataframe
            self.logger.info("Adding adjacency information to points dataframe")
            points_df['point_adjacent_ids'] = points_df.index.map(point_adjacency)
            points_df['point_adjacency_count'] = points_df['point_adjacent_ids'].apply(len)
            
            # Log adjacency statistics
            avg_adjacency = points_df['point_adjacency_count'].mean()
            max_adjacency = points_df['point_adjacency_count'].max()
            isolated_count = (points_df['point_adjacency_count'] == 0).sum()
            
            self.logger.info(f"Point adjacency statistics:")
            self.logger.info(f"  - Average adjacent points per point: {avg_adjacency:.2f}")
            self.logger.info(f"  - Maximum adjacent points: {max_adjacency}")
            self.logger.info(f"  - Isolated points: {isolated_count} ({isolated_count/len(points_df)*100:.1f}%)")
            
            return points_df
            
        except Exception as e:
            self.logger.error(f"Error in compute_point_adjacency_gpu: {e}")
            self.logger.error(f"Error details: {str(e)}")
            # Fall back to parallel CPU implementation
            self.logger.info("Falling back to CPU implementation")
            return self.compute_point_adjacency_parallel(segmentized_points, source_gdf, point_distance_threshold)

    def compute_point_adjacency_parallel(self, segmentized_points, source_gdf, point_distance_threshold=10, n_jobs=None):
        """
        Compute adjacency between segmentized points based on original segment adjacency
        with parallel processing for maximum performance
        
        Parameters:
        -----------
        segmentized_points : GeoDataFrame
            The segmentized points with level_0 indicating the parent segment
        source_gdf : GeoDataFrame
            The original segments with adjacent_ids column from previous computation
        point_distance_threshold : float
            Maximum distance between points to be considered adjacent (in feet)
        n_jobs : int, optional
            Number of parallel jobs to run. If None, uses CPU count - 1
            
        Returns:
        --------
        GeoDataFrame
            Points with point-level adjacency information
        """
        try:
            if 'adjacent_ids' not in source_gdf.columns:
                self.logger.warning("No adjacency information found in source data, skipping point adjacency")
                return segmentized_points
                
            self.logger.info("Computing point-level adjacency relationships with parallel processing")
            self.logger.info(f"Using point distance threshold of {point_distance_threshold} feet")
            
            # Reset index to get segment ID as level_0 column
            points_df = gpd.GeoDataFrame(segmentized_points).reset_index()
            
            # Set the active geometry column explicitly
            if 0 in points_df.columns:
                points_df = points_df.set_geometry(0)
                geometry_column = 0  
            else:
                self.logger.error("Could not find geometry column in segmentized points")
                return segmentized_points
            
            # Create a numpy array of geometries for faster access
            self.logger.info("Creating optimized data structures")
            point_geometries = np.array(points_df[geometry_column].tolist())
            point_segment_ids = np.array(points_df['level_0'].tolist())
            
            # Group points by segment for faster access
            segment_to_point_indices = {}
            for i, segment_id in enumerate(point_segment_ids):
                if segment_id not in segment_to_point_indices:
                    segment_to_point_indices[segment_id] = []
                segment_to_point_indices[segment_id].append(i)
            
            # Create STRtree for even faster spatial queries (better than geopandas sindex)
            self.logger.info("Building STRtree for optimal spatial indexing")
            point_index = STRtree(point_geometries)
            
            # Get all segment pairs that need processing
            segment_pairs = []
            for segment_id in source_gdf.index:
                if segment_id not in segment_to_point_indices:
                    continue
                    
                adjacent_segment_ids = source_gdf.loc[segment_id, 'adjacent_ids']
                if not adjacent_segment_ids:
                    continue
                    
                for adj_segment_id in adjacent_segment_ids:
                    if adj_segment_id in segment_to_point_indices:
                        # Only process each pair once (i,j) where i < j to avoid duplicates
                        if segment_id < adj_segment_id:
                            segment_pairs.append((segment_id, adj_segment_id))
            
            self.logger.info(f"Found {len(segment_pairs)} segment pairs to process")
            
            # Define worker function for parallel processing
            def process_segment_pair(pair_data):
                segment_id, adj_segment_id = pair_data
                adjacency_results = []
                
                # Get indices of points in both segments
                segment_point_indices = segment_to_point_indices[segment_id]
                adj_segment_point_indices = segment_to_point_indices[adj_segment_id]
                
                # For each point in first segment
                for i in segment_point_indices:
                    point_geom = point_geometries[i]
                    point_idx = points_df.index[i]
                    
                    # Use spatial index to find nearby points from adjacent segment
                    # This is much faster than checking all point pairs
                    buffer_bounds = point_geom.buffer(point_distance_threshold).bounds
                    candidate_indices = list(point_index.query(point_geom.buffer(point_distance_threshold)))
                    
                    # Filter candidates to only those from adjacent segment
                    for j in candidate_indices:
                        if j in adj_segment_point_indices:
                            adj_point_geom = point_geometries[j]
                            adj_point_idx = points_df.index[j]
                            
                            # Check distance
                            if point_geom.distance(adj_point_geom) <= point_distance_threshold:
                                adjacency_results.append((point_idx, adj_point_idx))
                
                return adjacency_results
            
            # Use multiprocessing for parallelization
            if n_jobs is None:
                n_jobs = max(1, multiprocessing.cpu_count() - 1)
                
            self.logger.info(f"Processing segment pairs in parallel using {n_jobs} processes")
            
            # Split work into chunks for better parallel performance
            chunk_size = max(1, len(segment_pairs) // (n_jobs * 10))  # Aim for ~10 chunks per job
            
            # Create pool and process chunks in parallel
            with multiprocessing.Pool(processes=n_jobs) as pool:
                all_results = []
                # Process in chunks with progress tracking
                for chunk_start in range(0, len(segment_pairs), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(segment_pairs))
                    chunk = segment_pairs[chunk_start:chunk_end]
                    
                    # Process the chunk
                    chunk_results = pool.map(process_segment_pair, chunk)
                    
                    # Flatten results
                    for result in chunk_results:
                        all_results.extend(result)
                        
                    # Log progress
                    progress = min(100, int((chunk_end / len(segment_pairs)) * 100))
                    self.logger.info(f"Progress: {progress}% ({chunk_end}/{len(segment_pairs)} pairs)")
            
            # Build adjacency dictionary from results
            self.logger.info(f"Building final adjacency structure from {len(all_results)} point pairs")
            point_adjacency = {idx: [] for idx in points_df.index}
            
            for point1, point2 in all_results:
                point_adjacency[point1].append(point2)
                point_adjacency[point2].append(point1)  # Add bidirectional adjacency
            
            # Add adjacency information to the points dataframe
            self.logger.info("Adding adjacency information to points dataframe")
            points_df['point_adjacent_ids'] = points_df.index.map(point_adjacency)
            points_df['point_adjacency_count'] = points_df['point_adjacent_ids'].apply(len)
            
            # Log adjacency statistics
            avg_adjacency = points_df['point_adjacency_count'].mean()
            max_adjacency = points_df['point_adjacency_count'].max()
            isolated_count = (points_df['point_adjacency_count'] == 0).sum()
            
            self.logger.info(f"Point adjacency statistics:")
            self.logger.info(f"  - Average adjacent points per point: {avg_adjacency:.2f}")
            self.logger.info(f"  - Maximum adjacent points: {max_adjacency}")
            self.logger.info(f"  - Isolated points: {isolated_count} ({isolated_count/len(points_df)*100:.1f}%)")
            
            return points_df
            
        except Exception as e:
            self.logger.error(f"Error in compute_point_adjacency_parallel: {e}")
            self.logger.error(f"Error details: {str(e)}")
            return segmentized_points

    def process(self, i, o=None, nta_path=None, neighborhood=None, 
               segmentation_distance=50, compute_adj=True, adj_tolerance=0.1,
               point_adjacency=True, point_distance_threshold=None):
        """
        Process sidewalk data to segmentize into points
        
        Args:
            i: Path to input sidewalk data
            o: Path to save output data (default: derived from input path)
            nta_path: Path to neighborhood data (optional)
            neighborhood: Name of neighborhood to focus on (optional)
            segmentation_distance: Distance between points in segmentation, feet (default: 50)
            compute_adj: Whether to compute adjacency relationships (default: True)
            adj_tolerance: Distance tolerance for considering geometries adjacent (default: 0.1 feet)
            point_adjacency: Whether to compute point-level adjacency (default: True)
            point_distance_threshold: Maximum distance between points to be considered adjacent (default: 1.2 * segmentation_distance)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Log processing parameters
            self.logger.info(f"Processing parameters:")
            self.logger.info(f"  - Input path: {i}")
            self.logger.info(f"  - Neighborhood filter: {'Yes, ' + neighborhood if neighborhood else 'No'}")
            self.logger.info(f"  - Segmentation distance: {segmentation_distance} feet")
            
            # Calculate intelligent default for point_distance_threshold if not specified
            if point_distance_threshold is None:
                # Use 1.2x the segmentation distance as the default threshold
                point_distance_threshold = 1.2 * segmentation_distance
                self.logger.info(f"  - Using auto-calculated point distance threshold: {point_distance_threshold:.2f} feet (1.2 × segmentation distance)")
            else:
                self.logger.info(f"  - Using manual point distance threshold: {point_distance_threshold} feet")
            
            # Set default output path if not provided
            if o is None:
                input_dir = os.path.dirname(i)
                input_basename = os.path.basename(i).split('.')[0]
                o = os.path.join(input_dir, f"{input_basename}_segmentized.parquet")
                self.logger.info(f"Using default output path: {o}")
            else:
                self.logger.info(f"Output path: {o}")
            
            self.logger.info("Starting sidewalk segmentization process")
            start_time = pd.Timestamp.now()
            
            # Load sidewalk data
            use_sidewalk_widths = "sidewalkwidths" in i.lower()
            self.logger.info(f"Data source identified as: {'sidewalk widths' if use_sidewalk_widths else 'standard sidewalks'}")
            sidewalks = self.read_geodataframe(
                i, 
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
            
            # Compute adjacency relationships if requested
            if compute_adj:
                self.logger.info("Computing segment-level adjacency relationships")
                sidewalks = self.compute_adjacency(sidewalks, tolerance=adj_tolerance)
                if sidewalks is None:
                    return False
            
            # Segmentize and extract points
            segmentized = self.segmentize_and_extract_points(sidewalks, distance=segmentation_distance)
            if segmentized is None:
                return False
            self.logger.info(f"Segmentized into {len(segmentized)} points")

            # cleanup 
            segmentized = self.cleanup(segmentized, "../../data/nyc/_raw/Sidewalk.geojson")
                
            # Compute point-level adjacency if requested and segment adjacency was computed
            if compute_adj and point_adjacency:
                self.logger.info("Computing point-level adjacency relationships")
                segmentized = self.compute_point_adjacency(
                    segmentized, 
                    sidewalks,
                    point_distance_threshold=point_distance_threshold
                )
                
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
            self.logger.info(f"Saving {len(result)} segmentized points to {o}")
            success = self.save_geoparquet(result, o)
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
#   --i="../data/sidewalkwidths_nyc.geojson" \
#   --o="../data/segmentized_nyc_sidewalks.parquet" \
#   --nta_path="../data/nynta2020_24b/nynta2020.shp" \
#   --neighborhood="Greenpoint" \
#   --segmentation_distance=50