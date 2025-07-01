import cudf
import cuspatial
import cupy as cp
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from data.nyc.c import PROJ_FT
import inspect
from typing import Dict, List, Set, Tuple, Optional, Union

class SegmentizeGPUImplementations:
    def __init__(self, parent):
        self.parent = parent
        self.logger = parent.logger
        
    def sidewalk_network_filter(self, segmentized_points, og_sidewalk_file_path):
        """
        GPU implementation of sidewalk_network_filter using cuSpatial for point-in-polygon operations.
        
        Parameters:
        -----------
        segmentized_points : GeoDataFrame
            Points to filter
        og_sidewalk_file_path : str
            Path to original sidewalk polygons
            
        Returns:
        --------
        GeoDataFrame
            Filtered points
        """
        self.logger.info("Using RAPIDS/cuSpatial for sidewalk_network_filter")

        # Load original sidewalks and convert to RAPIDS GeoSeries
        og_sidewalks = gpd.read_file(og_sidewalk_file_path).to_crs(PROJ_FT)['geometry']
        self.logger.info(f"Loaded {len(og_sidewalks)} polygons from sidewalk file")

        # Convert segmentized points to RAPIDS GeoSeries
        segmentized_points_cuspatial = cuspatial.from_geopandas(segmentized_points.geometry)
        self.logger.info(f"Converted {len(segmentized_points)} points to cuspatial format")

        # Convert entire sidewalk dataset to cuSpatial at once
        self.logger.info("Converting all polygons to cuSpatial format at once")
        og_sidewalks_cuspatial = cuspatial.from_geopandas(og_sidewalks)
        
        # Process in batches of 30 polygons due to cuSpatial limitation
        batch_size = 30  # cuSpatial maximum
        num_streams = 256  # Number of parallel streams to use
        len_before = len(segmentized_points)
        
        # Create a master mask for results
        master_mask = np.zeros(len(segmentized_points), dtype=bool)
        self.logger.info(f"Processing {len(og_sidewalks)} polygons with {num_streams} parallel streams")

        # Calculate number of batches
        num_batches = (len(og_sidewalks) + batch_size - 1) // batch_size
        
        # Process in super-batches to maximize GPU utilization
        for super_batch_start in range(0, num_batches, num_streams):
            super_batch_end = min(super_batch_start + num_streams, num_batches)
            
            # Create streams for parallel execution
            num_streams_to_use = super_batch_end - super_batch_start
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams_to_use)]
            batch_results = []
            
            # Launch work on multiple streams
            for stream_idx in range(num_streams_to_use):
                batch_num = super_batch_start + stream_idx + 1
                batch_start = (batch_num - 1) * batch_size
                batch_end = min(batch_start + batch_size, len(og_sidewalks))
                
                self.logger.debug(f"Queueing polygon batch {batch_num}/{num_batches} (indices {batch_start}:{batch_end})")
                
                # Process in stream context
                with streams[stream_idx]:
                    try:
                        # Extract the batch using iloc directly on the cuSpatial GeoSeries
                        polygon_batch_cuspatial = og_sidewalks_cuspatial.iloc[batch_start:batch_end]
                        
                        # Perform intersection check for this batch
                        batch_mask = cuspatial.point_in_polygon(
                            segmentized_points_cuspatial,
                            polygon_batch_cuspatial
                        )
                        
                        # Store result for this batch with its stream
                        batch_results.append((batch_mask, streams[stream_idx], batch_num))
                    except Exception as e:
                        self.logger.error(f"Error processing batch {batch_num} in stream {stream_idx}: {e}")
            
            # Synchronize and process results
            for batch_mask, stream, batch_num in batch_results:
                # Wait for this stream to complete
                stream.synchronize()
                
                # Convert to numpy and update the master mask with logical OR
                batch_mask_np = batch_mask.sum(axis=1).to_numpy().astype(bool)
                master_mask = np.logical_or(master_mask, batch_mask_np)
                
                points_in_batch = np.sum(batch_mask_np)
                self.logger.debug(f"Batch {batch_num} found {points_in_batch} points inside polygons")
                # Log running total of points contained
                self.logger.debug(f"Total points in polygons so far: {np.sum(master_mask)}")
            
            # log progress at end of super-batch
            self.logger.info(f"Super-batch {super_batch_start} processed, total points in polygons: {np.sum(master_mask)}")
        
        
        # Filter points based on the master mask
        segmentized_points = segmentized_points[master_mask]
        len_after = len(segmentized_points)

        # Ensure we return a GeoDataFrame, not a GeoSeries
        if isinstance(segmentized_points, gpd.GeoSeries):
            self.logger.info("Converting filtered GeoSeries back to GeoDataFrame")
            segmentized_points = gpd.GeoDataFrame(
                geometry=segmentized_points,
                crs=segmentized_points.crs
            )

        self.logger.info(f"Segmentized points cleaned up, {len_before} -> {len_after}")
        return segmentized_points
        
    def compute_point_adjacency(self, segmentized_points: gpd.GeoDataFrame, 
                               source_gdf: gpd.GeoDataFrame, 
                               point_distance_threshold: float = 10, 
                               batch_size: int = 1000) -> gpd.GeoDataFrame:
        """
        Compute adjacency using GPU acceleration via RAPIDS libraries.
        
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
        if 'adjacent_ids' not in source_gdf.columns:
            self.logger.warning("No adjacency information found in source data, skipping point adjacency")
            return segmentized_points
            
        self.logger.info(f"Computing point-level adjacency (threshold: {point_distance_threshold}ft)")
        
        # Reset index to get segment ID as level_0 column
        points_df = segmentized_points.reset_index()
        
        # Set the active geometry column explicitly - use existing geometry column first
        orig_geom_name = segmentized_points.geometry.name
        geometry_column = None
        
        # Try the original geometry column name first
        if orig_geom_name and orig_geom_name in points_df.columns:
            self.logger.info(f"Using original geometry column '{orig_geom_name}'")
            points_df = points_df.set_geometry(orig_geom_name)
            geometry_column = orig_geom_name
        # Then try standard column names
        elif 'geometry' in points_df.columns:
            self.logger.info("Using 'geometry' column")
            points_df = points_df.set_geometry('geometry')
            geometry_column = 'geometry'
        # Then try column 0 (legacy support)
        elif 0 in points_df.columns:
            self.logger.info("Using column '0' as geometry")
            points_df = points_df.set_geometry(0)
            geometry_column = 0
        # Last resort, scan all columns for Point objects
        else:
            for col in points_df.columns:
                if len(points_df) > 0 and isinstance(points_df[col].iloc[0], Point):
                    self.logger.info(f"Found column '{col}' containing Point objects")
                    points_df = points_df.set_geometry(col)
                    geometry_column = col
                    break
        
        if geometry_column is None:
            self.logger.error("Could not find geometry column in segmentized points")
            self.logger.debug(f"Available columns: {points_df.columns.tolist()}")
            return segmentized_points
        
        # Pre-process data for faster lookup
        segment_to_points: Dict[int, List[Tuple[int, Point]]] = {}
        for idx, row in points_df.iterrows():
            segment_id = row['level_0']
            if segment_id not in segment_to_points:
                segment_to_points[segment_id] = []
            segment_to_points[segment_id].append((idx, row[geometry_column]))
        
        # Get all segment pairs that need processing
        segment_pairs: List[Tuple[int, int]] = []
        for segment_id in source_gdf.index:
            if segment_id not in segment_to_points:
                continue
                
            adjacent_segment_ids = source_gdf.loc[segment_id, 'adjacent_ids']
            if not adjacent_segment_ids:
                continue
                
            for adj_segment_id in adjacent_segment_ids:
                if adj_segment_id in segment_to_points:
                    # Process both directions to ensure complete adjacency
                    segment_pairs.append((segment_id, adj_segment_id))
        
        self.logger.info(f"Processing {len(segment_pairs)} segment pairs")
        
        # Create a set to collect adjacency pairs 
        all_adjacency_pairs: Set[Tuple[int, int]] = set()
        total_pairs = len(segment_pairs)
        
        # For large collections, use vectorized operation if possible
        has_vectorized = False
        try:
            # Check for vectorized distance calculation support
            if hasattr(cuspatial, "pairwise_point_distance"):
                has_vectorized = True
        except Exception:
            has_vectorized = False
            
        # Process in batches
        for batch_start in range(0, total_pairs, batch_size):
            batch_end = min(batch_start + batch_size, total_pairs)
            batch = segment_pairs[batch_start:batch_end]
            
            # Only log progress every 25% to reduce verbosity
            progress_pct = int((batch_end / total_pairs) * 100)
            if batch_start == 0 or progress_pct % 25 == 0 or batch_end == total_pairs:
                self.logger.info(f"Point adjacency progress: {progress_pct}% ({batch_end}/{total_pairs})")
            
            # Process each segment pair in the batch
            for pair_idx in range(len(batch)):
                segment_id, adj_segment_id = batch[pair_idx]
                
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
                        # Fall back to CPU calculation for this segment pair
                        for i, (idx1, geom1) in enumerate(segment_points):
                            for j, (idx2, geom2) in enumerate(adj_segment_points):
                                if geom1.distance(geom2) <= point_distance_threshold:
                                    all_adjacency_pairs.add((idx1, idx2))
                                    all_adjacency_pairs.add((idx2, idx1))
                    
        # Convert set to list and build adjacency dictionary
        self.logger.info(f"Building adjacency structure from {len(all_adjacency_pairs)//2} point pairs")
        point_adjacency: Dict[int, List[int]] = {idx: [] for idx in points_df.index}
        
        for point1, point2 in all_adjacency_pairs:
            if point2 not in point_adjacency[point1]:
                point_adjacency[point1].append(point2)
        
        # After processing cross-segment adjacencies, add within-segment adjacencies
        self.logger.info("Adding within-segment adjacency relationships")
        
        # Group points by segment ID
        segment_point_groups = points_df.groupby('level_0')
        
        # Process each segment
        for segment_id, group in segment_point_groups:
            points = group[geometry_column].tolist()
            indices = group.index.tolist()
            
            # For smaller segments, use a direct approach
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    if points[i].distance(points[j]) <= point_distance_threshold:
                        idx1 = indices[i]
                        idx2 = indices[j]
                        
                        # Add bidirectional adjacency
                        if idx2 not in point_adjacency[idx1]:
                            point_adjacency[idx1].append(idx2)
                        if idx1 not in point_adjacency[idx2]:
                            point_adjacency[idx2].append(idx1)
        
        # Add adjacency information to the points dataframe
        points_df['point_adjacent_ids'] = points_df.index.map(point_adjacency)
        points_df['point_adjacency_count'] = points_df['point_adjacent_ids'].apply(len)
        
        # Log adjacency statistics (condensed)
        avg_adjacency = points_df['point_adjacency_count'].mean()
        max_adjacency = points_df['point_adjacency_count'].max()
        isolated_count = (points_df['point_adjacency_count'] == 0).sum()
        
        self.logger.info(f"Point adjacency stats: avg={avg_adjacency:.2f}, max={max_adjacency}, isolated={isolated_count} ({isolated_count/len(points_df)*100:.1f}%)")
        
        return points_df