import os
import geopandas as gpd
import pandas as pd
import numpy as np
import multiprocessing
from shapely.strtree import STRtree
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any, Union
from logging import Logger
from shapely.geometry import Point, Polygon

class SegmentizeCPUFallbacks:
    """
    CPU fallback implementations for SidewalkSegmentizer methods that can use GPU acceleration.
    
    This class provides CPU-based implementations that are used when RAPIDS/GPU
    acceleration is not available.
    """
    
    def __init__(self, parent: Any) -> None:
        """
        Initialize with reference to parent SidewalkSegmentizer
        
        Parameters:
        -----------
        parent : SidewalkSegmentizer
            The parent class that will use these methods
        """
        self.parent = parent
        self.logger: Logger = parent.logger
    
    def sidewalk_network_filter_cpu(self, segmentized_points: gpd.GeoDataFrame, og_sidewalk_file_path: str) -> gpd.GeoDataFrame:
        """
        CPU implementation of sidewalk network filter.
        
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
        from data.nyc.c import PROJ_FT
        
        self.logger.info("Using CPU implementation for sidewalk_network_filter")
        
        # Load original sidewalks
        og_sidewalks = gpd.read_file(og_sidewalk_file_path).to_crs(PROJ_FT)['geometry']
        self.logger.info(f"Loaded {len(og_sidewalks)} polygons from sidewalk file")
        
        # Create spatial index for sidewalks
        self.logger.info("Building spatial index for sidewalks")
        sidewalk_sindex = og_sidewalks.sindex
        
        len_before = len(segmentized_points)
        self.logger.info(f"Processing {len(segmentized_points)} points")
        
        # Process in chunks for better memory management
        chunk_size = 10000
        master_mask = np.zeros(len(segmentized_points), dtype=bool)
        
        for chunk_start in range(0, len(segmentized_points), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(segmentized_points))
            self.logger.info(f"Processing points {chunk_start}:{chunk_end}")
            
            chunk = segmentized_points.iloc[chunk_start:chunk_end]
            
            # Find candidate polygons for each point
            for i, point in enumerate(chunk.geometry):
                point_idx = i + chunk_start
                # Query the spatial index for candidate polygons
                candidates = list(sidewalk_sindex.intersection(point.bounds))
                
                # Test each candidate
                for candidate_idx in candidates:
                    if og_sidewalks.iloc[candidate_idx].contains(point):
                        master_mask[point_idx] = True
                        break
            
            # Log progress
            progress = min(100, int((chunk_end / len(segmentized_points)) * 100))
            self.logger.info(f"Progress: {progress}% ({chunk_end}/{len(segmentized_points)} points)")
        
        # Filter points based on the master mask
        segmentized_points = segmentized_points[master_mask]
        len_after = len(segmentized_points)
        
        self.logger.info(f"Segmentized points cleaned up, {len_before} -> {len_after}")
        return segmentized_points
    
    def compute_point_adjacency_parallel(self, 
                                         segmentized_points: gpd.GeoDataFrame, 
                                         source_gdf: gpd.GeoDataFrame, 
                                         point_distance_threshold: float = 10, 
                                         n_jobs: Optional[int] = None) -> gpd.GeoDataFrame:
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
            def process_segment_pair(pair_data: Tuple[int, int]) -> List[Tuple[int, int]]:
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
            point_adjacency: Dict[int, List[int]] = {idx: [] for idx in points_df.index}
            
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