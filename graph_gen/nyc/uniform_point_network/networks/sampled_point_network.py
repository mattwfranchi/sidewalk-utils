import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
from shapely.strtree import STRtree
from typing import Dict, List, Set, Tuple, Optional, Union
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict
import warnings
import sys
sys.path.append('/share/ju/sidewalk_utils')
from utils.logger import get_logger

# Import RAPIDS libraries for GPU acceleration
import cupy as cp
import cudf
import cugraph
import cuspatial
import fire
import random

# Import centerline helper
from centerline_helper import CenterlineHelper

# Try to import project constants
try:
    from data.nyc.c import PROJ_FT
except ImportError:
    print("WARNING: PROJ_FT not found. Using default NYC projection.")
    PROJ_FT = 'EPSG:2263'  # Default NYC projection


class SampledPointNetwork:
    """
    Geometric sampling-based point network generator that creates spatially uniform point distributions
    along block-level sidewalk centerlines using buffer zones and minimum distance thresholds.
    
    This class implements an alternative to H3-based approaches by using geometric sampling with
    buffer zones to ensure spatial uniformity and prevent MAUP (Modifiable Areal Unit Problem).
    
    Key features:
    - **Block-Level Processing**: Converts sidewalk polygons to centerlines for coherent network connectivity
    - **Buffer-based Sampling**: Creates exclusion zones around placed points
    - **Minimum Distance Thresholds**: Maintains spatial separation between points
    - **Network Topology Respect**: Preserves the structure and connectivity of the network
    - **Spatial Uniformity**: Ensures even distribution of points across the network
    - **MAUP Prevention**: Avoids artifacts from arbitrary spatial units
    - **Improved Adjacency**: Enhanced intersection detection for better connectivity
    
    Sampling Strategy:
    - **Uniform Sampling**: Points placed at regular intervals along centerlines
    """
    
    def __init__(self):
        # Use proper logger from logger.py
        self.logger = get_logger("SampledPointNetwork")
        self.logger.setLevel("INFO")
        
        # Set up GPU memory pool for better memory management
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        self.logger.info("GPU memory pool initialized for optimized memory management")
        
        # Load NYC census tracts for context
        try:
            self.ct_nyc = gpd.read_file("/share/ju/sidewalk_utils/data/nyc/geo/ct-nyc-2020.geojson").to_crs(PROJ_FT)
            self.logger.info(f"Loaded {len(self.ct_nyc)} NYC census tracts")
        except Exception as e:
            self.logger.error(f"Could not load NYC census tracts: {e}")
            self.ct_nyc = None

    def _cleanup_gpu_memory(self):
        """
        Clean up GPU memory after large operations.
        This helps prevent memory fragmentation and OOM errors.
        """
        try:
            # Clear CuPy memory pool
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            # Clear cuSpatial data if it exists
            if hasattr(self, '_original_segments_cuspatial'):
                del self._original_segments_cuspatial
                self.logger.info("Cleared cuSpatial segments data")
            
            # Clear spatial index if it exists
            if hasattr(self, '_segments_spatial_index'):
                del self._segments_spatial_index
                self.logger.info("Cleared spatial index")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("GPU memory cleaned up")
        except Exception as e:
            self.logger.warning(f"GPU memory cleanup failed: {e}")
    
    def _get_gpu_memory_info(self):
        """
        Get current GPU memory usage information.
        """
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            total = mempool.get_limit()
            used = mempool.used_bytes()
            free = total - used
            
            self.logger.info(f"GPU Memory: {used/1024**3:.2f}GB used, {free/1024**3:.2f}GB free, {total/1024**3:.2f}GB total")
            
            return {
                'used_gb': used / 1024**3,
                'free_gb': free / 1024**3,
                'total_gb': total / 1024**3
            }
        except Exception as e:
            self.logger.warning(f"Could not get GPU memory info: {e}")
            return None
    
    def generate_sampled_network(self, segments_gdf: gpd.GeoDataFrame,
                                buffer_distance: float = 100.0,
                                sampling_interval: float = 50.0,
                                strategy: str = "uniform",
                                preserve_intersections: bool = True,
                                min_segment_length: float = 10.0,
                                max_points_per_segment: int = 100,
                                pedestrian_ramps_path: Optional[str] = None,
                                compute_adjacency: bool = False,
                                walkshed_distance: float = 328.0,
                                crosswalk_distance: float = 100.0,
                                adjacency_centerlines: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """
        Generate geometrically sampled point network from sidewalk segments using uniform sampling.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input sidewalk segments with LineString geometries (must include 'parent_id' column)
        buffer_distance : float, default=100.0
            Minimum distance between points in feet (exclusion zone radius)
        sampling_interval : float, default=50.0
            Initial spacing between candidate points in feet
        strategy : str, default="uniform"
            Sampling strategy (only "uniform" is supported)
        preserve_intersections : bool, default=True
            Whether to preserve intersection points in the network (ignored for uniform sampling)
        min_segment_length : float, default=10.0
            Minimum segment length to consider for sampling
        max_points_per_segment : int, default=100
            Maximum number of points per segment (prevents excessive sampling)
        pedestrian_ramps_path : Optional[str], default=None
            Path to pedestrian ramps GeoJSON file (defaults to nyc_pedestrian_ramps.geojson)
            
        Returns:
        --------
        GeoDataFrame
            Sampled point network with spatial buffers and topology information
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING GEOMETRIC SAMPLING NETWORK GENERATION")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Load and process pedestrian ramps
        self.logger.info("Step 1: Loading and processing pedestrian ramps...")
        pedestrian_ramps_points = self._load_and_process_pedestrian_ramps(
            pedestrian_ramps_path, segments_gdf
        )
        
        # Step 2: Validate and prepare input segments
        self.logger.info("Step 2: Validating and preparing input segments...")
        validated_segments = self._validate_and_prepare_segments(segments_gdf, min_segment_length)
        
        if len(validated_segments) == 0:
            self.logger.error("No valid segments found after validation")
            return self._create_empty_result(segments_gdf.crs)
        
        # Step 3: Initialize sampling parameters
        self.logger.info("Step 3: Initializing sampling parameters...")
        # Only uniform strategy is supported
        if strategy != "uniform":
            self.logger.warning(f"WARNING: Strategy '{strategy}' not supported. Using 'uniform'")
            strategy = "uniform"
        
        # Ensure buffer distance is larger than sampling interval
        if buffer_distance <= sampling_interval:
            self.logger.warning(f"WARNING: Buffer distance ({buffer_distance}) should be larger than sampling interval ({sampling_interval})")
            buffer_distance = sampling_interval * 2.0
            self.logger.info(f"Adjusted buffer distance to {buffer_distance} feet")
        
        sampling_params = {
            'buffer_distance': buffer_distance,
            'sampling_interval': sampling_interval,
            'strategy': strategy,
            'preserve_intersections': preserve_intersections,
            'buffer_distance_squared': buffer_distance ** 2,  # Pre-calculate for efficiency
            'max_iterations': 1000  # Prevent infinite loops
        }
        
        self.logger.info(f"Sampling parameters initialized:")
        self.logger.info(f"  - Buffer distance: {buffer_distance} feet")
        self.logger.info(f"  - Sampling interval: {sampling_interval} feet")
        self.logger.info(f"  - Strategy: {strategy}")
        self.logger.info(f"  - Preserve intersections: {preserve_intersections}")
        
        # Store adjacency parameters
        self._compute_adjacency = compute_adjacency
        self._walkshed_distance = walkshed_distance
        self._crosswalk_distance = crosswalk_distance
        
        # Step 4: Generate candidate points
        self.logger.info("Step 4: Generating candidate points along segments...")
        candidate_points = self._generate_candidate_points(
            validated_segments, sampling_params, max_points_per_segment
        )
        
        if not candidate_points:
            self.logger.error("No candidate points generated")
            return self._create_empty_result(segments_gdf.crs)
        
        # Step 5: Apply parent_id-aware buffer filtering
        self.logger.info("Step 5: Applying parent_id-aware buffer filtering...")
        filtered_points = self._apply_parent_id_buffer_filtering(
            candidate_points, pedestrian_ramps_points, sampling_params
        )
        
        if not filtered_points:
            self.logger.error("No points survived buffer filtering")
            return self._create_empty_result(segments_gdf.crs)
        
        # Step 6: Handle intersections (if enabled)
        if preserve_intersections:
            self.logger.info("Step 6: Processing intersection points...")
            # For uniform sampling, we don't need special intersection processing
            self.logger.info("Uniform sampling - skipping intersection processing")
            intersection_processed = filtered_points
        else:
            intersection_processed = filtered_points
        
        # Step 7: Compute network topology
        self.logger.info("Step 7: Computing network topology...")
        topology_enhanced = self._compute_network_topology_gpu(
            intersection_processed, sampling_params
        )
        
        # Step 8: Compute adjacency relationships (optional)
        if compute_adjacency:
            self.logger.info("Step 8: Computing adjacency relationships...")
            
            # Debug pedestrian_ramps_points in main flow
            self.logger.info(f"DEBUG: Main flow - pedestrian_ramps_points: {'exists' if pedestrian_ramps_points else 'None'}")
            if pedestrian_ramps_points:
                self.logger.info(f"DEBUG: Main flow - Have {len(pedestrian_ramps_points)} pedestrian ramps points")
            
            # Generate crosswalk centerlines between intersection points
            if pedestrian_ramps_points:
                self.logger.info("Step 8a: Generating crosswalk centerlines...")
                crosswalk_centerlines = self._generate_intersection_centerlines(pedestrian_ramps_points, crosswalk_distance)
                
                # Debug crosswalk generation result
                self.logger.info(f"DEBUG: Main flow - Generated {len(crosswalk_centerlines)} crosswalk centerlines")
                
                # Combine existing centerlines with crosswalk centerlines
                if len(crosswalk_centerlines) > 0:
                    if adjacency_centerlines is not None:
                        # Combine filtered centerlines with crosswalk centerlines
                        combined_centerlines = pd.concat([adjacency_centerlines, crosswalk_centerlines], ignore_index=True)
                        self.logger.info(f"Combined {len(adjacency_centerlines)} existing centerlines with {len(crosswalk_centerlines)} crosswalk centerlines")
                    else:
                        # Use crosswalk centerlines only
                        combined_centerlines = crosswalk_centerlines
                        self.logger.info(f"Using {len(crosswalk_centerlines)} crosswalk centerlines for adjacency computation")
                else:
                    # No crosswalk centerlines generated, use existing centerlines
                    combined_centerlines = adjacency_centerlines if adjacency_centerlines is not None else segments_gdf
                    self.logger.info(f"No crosswalk centerlines generated, using {'filtered centerlines' if adjacency_centerlines is not None else 'neighborhood segments'}")
            else:
                # No intersection points, use existing centerlines
                combined_centerlines = adjacency_centerlines if adjacency_centerlines is not None else segments_gdf
                self.logger.info(f"No intersection points, using {'filtered centerlines' if adjacency_centerlines is not None else 'neighborhood segments'}")
            
            adjacency_enhanced = self.compute_adjacency_relationships(
                topology_enhanced, combined_centerlines, 
                walkshed_distance=walkshed_distance,
                crosswalk_distance=crosswalk_distance
            )
        else:
            adjacency_enhanced = topology_enhanced
        
        # Step 9: Create final GeoDataFrame
        self.logger.info("Step 9: Creating final sampled point network...")
        result = self._create_final_network(adjacency_enhanced, segments_gdf.crs)
        
        # Store the combined centerlines as an attribute of the result for inspection files
        if compute_adjacency and 'combined_centerlines' in locals():
            result.attrs['combined_centerlines_with_crosswalks'] = combined_centerlines
            self.logger.info(f"DEBUG: Stored {len(combined_centerlines)} combined centerlines (including crosswalks) in result attributes")
        else:
            result.attrs['combined_centerlines_with_crosswalks'] = None
            self.logger.info("DEBUG: No combined centerlines with crosswalks to store")
        
        # Log final statistics
        total_time = time.time() - start_time
        self._log_final_statistics(result, candidate_points, total_time)
        
        return result
    
    def _validate_and_prepare_segments(self, segments_gdf: gpd.GeoDataFrame, 
                                      min_segment_length: float) -> gpd.GeoDataFrame:
        """
        Validate and prepare input segments for sampling.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input sidewalk segments
        min_segment_length : float
            Minimum segment length threshold
            
        Returns:
        --------
        GeoDataFrame
            Validated and prepared segments
        """
        self.logger.info(f"Validating {len(segments_gdf)} input segments...")
        
        # Remove invalid geometries
        valid_geoms = segments_gdf[segments_gdf.geometry.is_valid & ~segments_gdf.geometry.is_empty]
        
        # Filter by minimum length
        long_enough = valid_geoms[valid_geoms.geometry.length >= min_segment_length]
        
        # Add segment metadata
        result = long_enough.copy()
        result['segment_id'] = range(len(result))
        result['segment_length'] = result.geometry.length
        result['segment_type'] = result.geometry.geom_type
        
        # Handle MultiLineString by exploding to individual LineStrings
        if (result.geometry.geom_type == 'MultiLineString').any():
            result = result.explode(index_parts=True).reset_index(drop=True)
            result['segment_id'] = range(len(result))
        
        # Validate parent_id field exists
        if 'parent_id' not in result.columns:
            self.logger.error("ERROR: 'parent_id' column not found in segments data")
            self.logger.error("This field is required to prevent cross-street buffer bleeding")
            raise ValueError("Missing required 'parent_id' column in segments data")
        
        self.logger.info(f"After validation: {len(result)} segments ({len(segments_gdf) - len(result)} filtered out)")
        self.logger.info(f"  - Average segment length: {result['segment_length'].mean():.2f} feet")
        self.logger.info(f"  - Total network length: {result['segment_length'].sum():.2f} feet")
        self.logger.info(f"  - Unique parent_ids: {result['parent_id'].nunique()}")
        
        return result
    
    def _load_and_process_pedestrian_ramps(self, pedestrian_ramps_path: Optional[str], 
                                          segments_gdf: gpd.GeoDataFrame) -> List[Dict]:
        """
        Load pedestrian ramps and create initial intersection points using vectorized operations.
        
        Parameters:
        -----------
        pedestrian_ramps_path : Optional[str]
            Path to pedestrian ramps GeoJSON file
        segments_gdf : GeoDataFrame
            Input segments for CRS reference
            
        Returns:
        --------
        List[Dict]
            List of pedestrian ramp point records
        """
        if pedestrian_ramps_path is None:
            # Default path - user can modify this
            pedestrian_ramps_path = "/share/ju/sidewalk_utils/data/nyc/geo/nyc_pedestrian_ramps.geojson"
        
        try:
            # Load pedestrian ramps
            self.logger.info(f"Loading pedestrian ramps from: {pedestrian_ramps_path}")
            ramps_gdf = gpd.read_file(pedestrian_ramps_path)
            
            # Debug: Check ramps data structure
            self.logger.info(f"Ramps GDF shape: {ramps_gdf.shape}")
            self.logger.info(f"Ramps GDF columns: {list(ramps_gdf.columns)}")
            self.logger.info(f"Ramps GDF CRS: {ramps_gdf.crs}")
            self.logger.info(f"Ramps GDF first few rows info:")
            self.logger.info(f"  - Index: {ramps_gdf.index[:5].tolist()}")
            self.logger.info(f"  - Geometry types: {ramps_gdf.geometry.geom_type.value_counts().to_dict()}")
            
            # Convert to same CRS as segments
            if ramps_gdf.crs != segments_gdf.crs:
                self.logger.info(f"Converting ramps CRS from {ramps_gdf.crs} to {segments_gdf.crs}")
                ramps_gdf = ramps_gdf.to_crs(segments_gdf.crs)
            
            self.logger.info(f"Loaded {len(ramps_gdf)} pedestrian ramps")

            # filter the ramps to only include those that are within 100 feet of a segment
            original_count = len(ramps_gdf)
            self.logger.info("Filtering pedestrian ramps to only include those within 100 feet of a segment...")
            ramps_gdf = self._filter_pedestrian_ramps(ramps_gdf, segments_gdf)
            self.logger.info(f"Filtered {original_count} pedestrian ramps to {len(ramps_gdf)}")
            
            # Debug: Check segments data structure
            self.logger.info(f"Segments GDF shape: {segments_gdf.shape}")
            self.logger.info(f"Segments GDF columns: {list(segments_gdf.columns)}")
            self.logger.info(f"Segments parent_id sample: {segments_gdf['parent_id'].head().tolist()}")
            
            # Prepare segments for spatial join - add index column for later reference
            segments_with_idx = segments_gdf.copy()
            segments_with_idx['segment_idx'] = segments_with_idx.index
            
            # Use vectorized spatial join to find nearest segments
            self.logger.info("Using sjoin_nearest for vectorized nearest neighbor search...")
            
            # Perform spatial join to find nearest segment for each ramp
            joined = gpd.sjoin_nearest(
                ramps_gdf, 
                segments_with_idx, 
                how='left',
                distance_col='distance_to_segment'
            )
            
            self.logger.info(f"Spatial join completed. Joined shape: {joined.shape}")
            
            # Check for any failed joins
            failed_joins = joined[joined['parent_id'].isna()]
            if len(failed_joins) > 0:
                self.logger.warning(f"WARNING: {len(failed_joins)} ramps failed to join with segments")
                # Drop failed joins
                joined = joined.dropna(subset=['parent_id'])
                self.logger.info(f"Proceeding with {len(joined)} successfully joined ramps")
            
            # Convert to list of dictionaries
            ramp_points = []
            
            for idx, (_, row) in enumerate(joined.iterrows()):
                # Create point record for this ramp
                ramp_point = {
                    'geometry': row.geometry,
                    'point_id': f"ramp_{idx}",
                    'is_pedestrian_ramp': True,
                    'is_intersection': True,
                    'parent_id': row['parent_id'],
                    'source_segment_idx': row['segment_idx'],
                    'distance_to_segment': row['distance_to_segment'],
                    # Add required keys for consistency with regular sampled points
                    'distance_along_segment': 0.0,  # Ramps are at segment start
                    'segment_total_length': 0.0,  # Will be updated later if needed
                    'position_ratio': 0.0,  # Ramps are at start of segment
                    'buffer_zone': None,  # Will be computed during filtering
                    'network_neighbors': [],  # Will be populated during topology computation
                    'ramp_attributes': {key: value for key, value in row.items() 
                                      if key not in ['geometry', 'parent_id', 'segment_idx', 'distance_to_segment']}
                }
                
                ramp_points.append(ramp_point)
            
            self.logger.info(f"Processed {len(ramp_points)} pedestrian ramp points")
            return ramp_points
            
        except Exception as e:
            import traceback
            self.logger.error(f"ERROR: Failed to load pedestrian ramps from {pedestrian_ramps_path}: {e}")
            self.logger.error(f"ERROR: Full traceback: {traceback.format_exc()}")
            self.logger.info("Continuing without pedestrian ramps...")
            return []
    
    def _filter_pedestrian_ramps(self, ramps_gdf: gpd.GeoDataFrame, segments_gdf: gpd.GeoDataFrame, buffer_distance: float = 100.0) -> gpd.GeoDataFrame:
        """
        Filter pedestrian ramps to only include those within buffer_distance of a segment.
        
        Parameters:
        -----------
        ramps_gdf : GeoDataFrame
            Pedestrian ramps GeoDataFrame
        segments_gdf : GeoDataFrame
            Segments GeoDataFrame
        buffer_distance : float
            Maximum distance to consider a ramp near a segment
            
        Returns:
        --------
        GeoDataFrame
            Filtered pedestrian ramps
        """
        # Create a spatial index for the segments
        segment_tree = STRtree(list(segments_gdf.geometry))
        
        # Find ramps that are within buffer_distance of any segment
        filtered_ramps = []
        
        for idx, ramp_geom in enumerate(ramps_gdf.geometry):
            # Find segments within buffer distance
            nearby_segments = segment_tree.query(ramp_geom.buffer(buffer_distance))
            
            if len(nearby_segments) > 0:
                # Check actual distances to nearby segments
                min_distance = float('inf')
                for segment_idx in nearby_segments:
                    segment_geom = segments_gdf.iloc[segment_idx].geometry
                    distance = ramp_geom.distance(segment_geom)
                    min_distance = min(min_distance, distance)
                
                # Keep ramp if it's within buffer_distance
                if min_distance <= buffer_distance:
                    filtered_ramps.append(idx)
        
        # Return filtered GeoDataFrame
        if filtered_ramps:
            return ramps_gdf.iloc[filtered_ramps].reset_index(drop=True)
        else:
            # Return empty GeoDataFrame with same columns
            return ramps_gdf.iloc[:0].copy()

    def _generate_candidate_points(self, segments_gdf: gpd.GeoDataFrame,
                                  sampling_params: Dict,
                                  max_points_per_segment: int) -> List[Dict]:
        """
        Generate candidate points along segments using uniform sampling.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Validated segments
        sampling_params : Dict
            Sampling parameters
        max_points_per_segment : int
            Maximum points per segment
            
        Returns:
        --------
        List[Dict]
            List of candidate point records
        """
        candidates = []
        interval = sampling_params['sampling_interval']
        
        for idx, (_, row) in enumerate(segments_gdf.iterrows()):
            if idx % 1000 == 0:
                self.logger.info(f"  Processing segment {idx + 1}/{len(segments_gdf)} ({idx/len(segments_gdf)*100:.1f}%)")
            
            try:
                segment_candidates = self._generate_points_along_linestring(
                    row.geometry, interval, idx, row, max_points_per_segment
                )
                candidates.extend(segment_candidates)
            except Exception as e:
                self.logger.error(f"Error processing segment {idx}: {e}")
                continue
        
        self.logger.info(f"Generated {len(candidates)} uniform candidate points")
        return candidates
    
    def _generate_uniform_candidates(self, segments_gdf: gpd.GeoDataFrame,
                                    sampling_params: Dict,
                                    max_points_per_segment: int) -> List[Dict]:
        """
        Generate uniformly spaced candidate points along segments.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input segments
        sampling_params : Dict
            Sampling parameters
        max_points_per_segment : int
            Maximum points per segment
            
        Returns:
        --------
        List[Dict]
            List of candidate point records
        """
        candidates = []
        interval = sampling_params['sampling_interval']
        
        for idx, (_, row) in enumerate(segments_gdf.iterrows()):
            if idx % 1000 == 0:
                self.logger.info(f"  Processing segment {idx + 1}/{len(segments_gdf)} ({idx/len(segments_gdf)*100:.1f}%)")
            
            try:
                segment_candidates = self._generate_points_along_linestring(
                    row.geometry, interval, idx, row, max_points_per_segment
                )
                candidates.extend(segment_candidates)
            except Exception as e:
                self.logger.error(f"Error processing segment {idx}: {e}")
                continue
        
        self.logger.info(f"Generated {len(candidates)} uniform candidate points")
        return candidates
    

    
    def _generate_points_along_linestring(self, linestring: LineString,
                                         interval: float,
                                         segment_idx: int,
                                         segment_row: pd.Series,
                                         max_points: int) -> List[Dict]:
        """
        Generate points along a single LineString at specified intervals.
        
        Parameters:
        -----------
        linestring : LineString
            Input LineString geometry
        interval : float
            Interval between points
        segment_idx : int
            Index of the source segment
        segment_row : pd.Series
            Source segment data
        max_points : int
            Maximum number of points to generate
            
        Returns:
        --------
        List[Dict]
            List of point records
        """
        points = []
        length = linestring.length
        
        if length < interval:
            # For very short segments, place one point at the midpoint
            midpoint = linestring.interpolate(length / 2.0)
            record = {
                'geometry': midpoint,
                'source_segment_idx': segment_idx,
                'distance_along_segment': length / 2.0,
                'segment_total_length': length,
                'position_ratio': 0.5,
                'point_id': None,  # Will be assigned later
                'is_intersection': False,  # Will be updated if needed
                'buffer_zone': None,  # Will be computed during filtering
                'network_neighbors': [],  # Will be populated during topology computation
            }
            
            # Add all original segment attributes with prefix
            for col in segment_row.index:
                if col != 'geometry':
                    record[f'source_{col}'] = segment_row[col]
            
            # Ensure parent_id is directly accessible (not just as source_parent_id)
            if 'parent_id' in segment_row.index:
                record['parent_id'] = segment_row['parent_id']
            points.append(record)
        else:
            # Generate points at regular intervals
            num_points = min(int(length / interval) + 1, max_points)
            
            for i in range(num_points):
                distance = min(i * interval, length)
                point_geom = linestring.interpolate(distance)
                
                record = {
                    'geometry': point_geom,
                    'source_segment_idx': segment_idx,
                    'distance_along_segment': distance,
                    'segment_total_length': length,
                    'position_ratio': distance / length,
                    'point_id': None,  # Will be assigned later
                    'is_intersection': False,  # Will be updated if needed
                    'buffer_zone': None,  # Will be computed during filtering
                    'network_neighbors': [],  # Will be populated during topology computation
                }
                
                # Add all original segment attributes with prefix
                for col in segment_row.index:
                    if col != 'geometry':
                        record[f'source_{col}'] = segment_row[col]
                
                # Ensure parent_id is directly accessible (not just as source_parent_id)
                if 'parent_id' in segment_row.index:
                    record['parent_id'] = segment_row['parent_id']
                points.append(record)
        
        return points
    

    
    def _create_point_record(self, point_geom: Point,
                            segment_idx: int,
                            segment_row: pd.Series,
                            distance: float,
                            total_length: float,
                            position_ratio: float) -> Dict:
        """
        Create a standardized point record with all necessary metadata.
        
        Parameters:
        -----------
        point_geom : Point
            Point geometry
        segment_idx : int
            Source segment index
        segment_row : pd.Series
            Source segment data
        distance : float
            Distance along segment
        total_length : float
            Total segment length
        position_ratio : float
            Position as ratio of total length
            
        Returns:
        --------
        Dict
            Point record with metadata
        """
        record = {
            'geometry': point_geom,
            'source_segment_idx': segment_idx,
            'distance_along_segment': distance,
            'segment_total_length': total_length,
            'position_ratio': position_ratio,
            'point_id': None,  # Will be assigned later
            'is_intersection': False,  # Will be updated if needed
            'buffer_zone': None,  # Will be computed during filtering
            'network_neighbors': [],  # Will be populated during topology computation
        }
        
        # Add all original segment attributes with prefix
        for col in segment_row.index:
            if col != 'geometry':
                record[f'source_{col}'] = segment_row[col]
        
        # Ensure parent_id is directly accessible (not just as source_parent_id)
        if 'parent_id' in segment_row.index:
            record['parent_id'] = segment_row['parent_id']
        
        return record
    

    

    

    
    def _apply_parent_id_buffer_filtering(self, candidate_points: List[Dict],
                                         pedestrian_ramps_points: List[Dict],
                                         sampling_params: Dict) -> List[Dict]:
        """
        Apply parent_id-aware buffer filtering to maintain minimum distance within parent_id groups.
        
        Parameters:
        -----------
        candidate_points : List[Dict]
            List of candidate point records
        pedestrian_ramps_points : List[Dict]
            List of pedestrian ramp point records (already accepted)
        sampling_params : Dict
            Sampling parameters
            
        Returns:
        --------
        List[Dict]
            Filtered point records
        """
        self.logger.info(f"Applying parent_id-aware buffer filtering...")
        self.logger.info(f"  - Starting with {len(pedestrian_ramps_points)} pedestrian ramp points")
        self.logger.info(f"  - Processing {len(candidate_points)} candidate points")
        
        # Start with all pedestrian ramp points (these are automatically accepted)
        accepted_points = pedestrian_ramps_points.copy()
        
        # Assign point IDs to pedestrian ramps
        for i, point in enumerate(accepted_points):
            point['point_id'] = i
            point['buffer_zone'] = point['geometry'].buffer(sampling_params['buffer_distance'])
        
        # Group candidate points by parent_id
        candidates_by_parent_id = defaultdict(list)
        for candidate in candidate_points:
            parent_id = candidate.get('parent_id')
            candidates_by_parent_id[parent_id].append(candidate)
        
        # Process each parent_id group separately
        for parent_id, parent_candidates in candidates_by_parent_id.items():
            self.logger.debug(f"  Processing parent_id {parent_id} with {len(parent_candidates)} candidates")
            
            # Get accepted points for this parent_id only
            parent_accepted_points = [
                pt for pt in accepted_points 
                if pt.get('parent_id') == parent_id
            ]
            
            # Apply buffer filtering within this parent_id group
            filtered_parent_points = self._apply_buffer_filtering_parent_id(
                parent_candidates, parent_accepted_points, sampling_params
            )
            
            # Add filtered points to the main accepted list
            for point in filtered_parent_points:
                point['point_id'] = len(accepted_points)
                accepted_points.append(point)
        
        self.logger.info(f"Parent_id-aware buffer filtering complete:")
        self.logger.info(f"  - Pedestrian ramp points: {len(pedestrian_ramps_points)}")
        self.logger.info(f"  - Additional sampled points: {len(accepted_points) - len(pedestrian_ramps_points)}")
        self.logger.info(f"  - Total accepted points: {len(accepted_points)}")
        
        return accepted_points
    

    

    
    def _apply_buffer_filtering_parent_id(self, candidate_points: List[Dict],
                                         existing_points: List[Dict],
                                         sampling_params: Dict) -> List[Dict]:
        """
        GPU-based buffer filtering within a parent_id group.
        
        Parameters:
        -----------
        candidate_points : List[Dict]
            Candidate points for this parent_id
        existing_points : List[Dict]
            Already accepted points for this parent_id
        sampling_params : Dict
            Sampling parameters
            
        Returns:
        --------
        List[Dict]
            Filtered points for this parent_id
        """
        buffer_distance = sampling_params['buffer_distance']
        buffer_distance_squared = sampling_params['buffer_distance_squared']
        
        # Extract coordinates for GPU processing
        candidate_coords = np.array([[pt['geometry'].x, pt['geometry'].y] for pt in candidate_points])
        existing_coords = np.array([[pt['geometry'].x, pt['geometry'].y] for pt in existing_points])
        
        # Transfer to GPU
        candidate_coords_gpu = cp.asarray(candidate_coords)
        existing_coords_gpu = cp.asarray(existing_coords) if len(existing_points) > 0 else None
        
        # Sort candidates by segment and position
        sort_keys = [(pt['source_segment_idx'], pt['distance_along_segment']) for pt in candidate_points]
        sort_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
        
        sorted_candidate_coords = candidate_coords_gpu[sort_indices]
        
        # GPU-accelerated filtering
        accepted_indices = []
        
        for i in range(len(sorted_candidate_coords)):
            current_point = sorted_candidate_coords[i:i+1]  # Keep 2D shape (1, 2)
            
            # Check distance to existing points (pedestrian ramps)
            too_close = False
            if existing_coords_gpu is not None:
                diffs = existing_coords_gpu - current_point
                distances_squared = cp.sum(diffs * diffs, axis=1)
                if cp.min(distances_squared) < buffer_distance_squared:
                    too_close = True
            
            # Check distance to already accepted candidates
            if not too_close and accepted_indices:
                accepted_coords = sorted_candidate_coords[accepted_indices]
                diffs = accepted_coords - current_point
                distances_squared = cp.sum(diffs * diffs, axis=1)
                if cp.min(distances_squared) < buffer_distance_squared:
                    too_close = True
            
            if not too_close:
                accepted_indices.append(i)
        
        # Transfer results back to CPU
        accepted_indices_cpu = cp.asnumpy(cp.array(accepted_indices))
        
        # Create filtered point list
        accepted_points = []
        for i in accepted_indices_cpu:
            orig_idx = sort_indices[i]
            point = candidate_points[orig_idx].copy()
            point['buffer_zone'] = point['geometry'].buffer(buffer_distance)
            point['parent_id'] = point.get('source_parent_id', point.get('parent_id'))
            accepted_points.append(point)
        
        return accepted_points
    

    

    

    
    def _process_intersection_points_direct(self, filtered_points: List[Dict],
                                           segments_gdf: gpd.GeoDataFrame,
                                           sampling_params: Dict) -> List[Dict]:
        """
        Process intersection points to ensure they are properly represented.
        Uses GPU acceleration for much faster processing.
        
        Parameters:
        -----------
        filtered_points : List[Dict]
            Filtered point records
        segments_gdf : GeoDataFrame
            Original segments
        sampling_params : Dict
            Sampling parameters
            
        Returns:
        --------
        List[Dict]
            Points with intersection information
        """
        # For uniform sampling, we don't need special intersection processing
        self.logger.info("Uniform sampling - skipping intersection processing")
        return filtered_points
    

    
    def _remove_duplicate_points_vectorized(self, points: List[Point]) -> List[Point]:
        """
        Remove duplicate points using vectorized GPU operations.
        
        Parameters:
        -----------
        points : List[Point]
            List of points that may contain duplicates
            
        Returns:
        --------
        List[Point]
            List of unique points
        """
        if not points or len(points) <= 1:
            return points
        
        # Extract coordinates
        coords = cp.array([[pt.x, pt.y] for pt in points])
        
        # Use CuPy for vectorized distance calculations
        unique_indices = [0]  # Always keep the first point
        tolerance = 1.0  # 1 foot tolerance
        tolerance_squared = tolerance * tolerance
        
        # Process in larger batches for better GPU utilization
        batch_size = 10000
        
        for i in range(1, len(coords), batch_size):
            end_idx = min(i + batch_size, len(coords))
            batch_coords = coords[i:end_idx]
            
            # Calculate distances from batch to all unique points
            unique_coords = coords[unique_indices]
            
            # Vectorized distance calculation: (batch_size, num_unique)
            diffs = batch_coords[:, cp.newaxis, :] - unique_coords[cp.newaxis, :, :]
            distances_squared = cp.sum(diffs * diffs, axis=2)
            
            # Find minimum distance for each batch point
            min_distances_squared = cp.min(distances_squared, axis=1)
            
            # Keep points that are not too close to existing unique points
            unique_batch_mask = min_distances_squared >= tolerance_squared
            
            # Add indices of unique batch points
            batch_indices = cp.arange(i, end_idx)
            unique_batch_indices = batch_indices[unique_batch_mask]
            unique_indices.extend(cp.asnumpy(unique_batch_indices).tolist())
        
        # Convert back to CPU and return unique points
        return [points[i] for i in unique_indices]
    

    

    

    
    def _compute_network_topology_gpu(self, points: List[Dict],
                                     sampling_params: Dict) -> List[Dict]:
        """
        GPU-accelerated network topology computation using CuPy for vectorized operations.
        
        Parameters:
        -----------
        points : List[Dict]
            Point records
        sampling_params : Dict
            Sampling parameters
            
        Returns:
        --------
        List[Dict]
            Points with topology information
        """
        self.logger.info("Using GPU-accelerated topology computation...")
        
        topology_buffer = sampling_params['buffer_distance'] * 2.0  # Larger buffer for topology
        topology_buffer_squared = topology_buffer ** 2
        
        # Extract coordinates and metadata for GPU processing
        coords = np.array([[pt['geometry'].x, pt['geometry'].y] for pt in points])
        segment_indices = np.array([pt['source_segment_idx'] for pt in points])
        point_ids = np.array([pt['point_id'] for pt in points])
        is_intersection = np.array([pt.get('is_intersection', False) for pt in points])
        
        # Transfer to GPU
        coords_gpu = cp.asarray(coords)
        segment_indices_gpu = cp.asarray(segment_indices)
        point_ids_gpu = cp.asarray(point_ids)
        is_intersection_gpu = cp.asarray(is_intersection)
        
        # Use vectorized distance calculations for topology
        self.logger.info("Computing vectorized distances for topology...")
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        topology_info = {}
        
        # Initialize topology info for all points
        all_point_ids = cp.asnumpy(point_ids_gpu)
        for point_id in all_point_ids:
            topology_info[point_id] = []
        
        for start_idx in range(0, len(coords_gpu), batch_size):
            end_idx = min(start_idx + batch_size, len(coords_gpu))
            batch_coords = coords_gpu[start_idx:end_idx]
            
            if start_idx % 5000 == 0:
                self.logger.info(f"  Processing topology batch {start_idx//batch_size + 1}/{(len(coords_gpu) + batch_size - 1)//batch_size}")
            
            # Calculate distances from batch to all other points
            # Shape: (batch_size, total_points)
            diffs = batch_coords[:, cp.newaxis, :] - coords_gpu[cp.newaxis, :, :]
            distances_squared = cp.sum(diffs * diffs, axis=2)
            
            # Find points within topology buffer (excluding self)
            within_buffer_mask = distances_squared < topology_buffer_squared
            
            # Remove self-connections
            for i in range(len(batch_coords)):
                within_buffer_mask[i, start_idx + i] = False
            
            # Process each point in the batch
            for i in range(len(batch_coords)):
                point_idx = start_idx + i
                point_id = int(point_ids_gpu[point_idx])
                
                # Get indices of points within buffer
                neighbor_indices = cp.where(within_buffer_mask[i])[0]
                
                if len(neighbor_indices) > 0:
                    # Get distances and neighbor info
                    neighbor_distances = cp.sqrt(distances_squared[i, neighbor_indices])
                    neighbor_point_ids = point_ids_gpu[neighbor_indices]
                    neighbor_segment_indices = segment_indices_gpu[neighbor_indices]
                    neighbor_is_intersection = is_intersection_gpu[neighbor_indices]
                    
                    # Sort by distance
                    sort_indices = cp.argsort(neighbor_distances)
                    sorted_distances = neighbor_distances[sort_indices]
                    sorted_point_ids = neighbor_point_ids[sort_indices]
                    sorted_segment_indices = neighbor_segment_indices[sort_indices]
                    sorted_is_intersection = neighbor_is_intersection[sort_indices]
                    
                    # Limit to 10 nearest neighbors
                    max_neighbors = min(10, len(sort_indices))
                    sorted_distances = sorted_distances[:max_neighbors]
                    sorted_point_ids = sorted_point_ids[:max_neighbors]
                    sorted_segment_indices = sorted_segment_indices[:max_neighbors]
                    sorted_is_intersection = sorted_is_intersection[:max_neighbors]
                    
                    # Convert to CPU for relationship classification
                    distances_cpu = cp.asnumpy(sorted_distances)
                    point_ids_cpu = cp.asnumpy(sorted_point_ids)
                    segment_indices_cpu = cp.asnumpy(sorted_segment_indices)
                    is_intersection_cpu = cp.asnumpy(sorted_is_intersection)
                    
                    # Create neighbor records
                    neighbors = []
                    for j in range(len(distances_cpu)):
                        relationship = self._classify_relationship_gpu(
                            segment_indices_gpu[point_idx], is_intersection_gpu[point_idx],
                            segment_indices_cpu[j], is_intersection_cpu[j]
                        )
                        
                        neighbors.append({
                            'point_id': int(point_ids_cpu[j]),
                            'distance': float(distances_cpu[j]),
                            'relationship': relationship
                        })
                    
                    topology_info[point_id] = neighbors
        
        # Update the points with topology information
        for i, point in enumerate(points):
            point_id = point['point_id']
            point['network_neighbors'] = topology_info.get(point_id, [])
        
        # Clean up GPU memory after large operation
        self._cleanup_gpu_memory()
        
        self.logger.info(f"GPU-accelerated topology computation complete for {len(points)} points")
        return points
    
    def compute_adjacency_relationships(self, points: List[Dict], 
                                      segments_gdf: gpd.GeoDataFrame,
                                      walkshed_distance: float = 328.0,  # 100 meters in feet
                                      crosswalk_distance: float = 100.0,
                                      batch_params: Optional[Dict] = None) -> List[Dict]:
        """
        Compute adjacency relationships using simplified GPU-accelerated approach.
        
        This method consolidates all adjacency computation into a single GPU pipeline:
        1. Convert all data to GPU format once
        2. Compute nearest linestrings for all points
        3. Find adjacencies in a single vectorized operation
        4. Return results directly
        
        IMPORTANT INDEXING DIFFERENCES:
        - cuDF DataFrames: Use .iloc[] for integer indexing, .loc[] for label indexing
        - cuDF .idxmin()/.idxmax(): Returns the index value, not position
        - cuDF .values: Returns CuPy arrays, use .get() to extract scalars
        - cuSpatial quadtree: key_to_point maps point_id to quadtree index, not DataFrame index
        - CuPy arrays: Use standard numpy-style indexing
        
        Parameters:
        -----------
        points : List[Dict]
            Point records from the sampled network
        segments_gdf : GeoDataFrame
            Original sidewalk segments
        walkshed_distance : float, default=328.0
            Walkshed radius in feet (100 meters) - kept for compatibility
        crosswalk_distance : float, default=100.0
            Maximum distance for crosswalk creation in feet
        batch_params : Optional[Dict], default=None
            Batching parameters for memory management
            
        Returns:
        --------
        List[Dict]
            Points with adjacency information added
        """
        self.logger.info("=" * 60)
        self.logger.info("COMPUTING ADJACENCY RELATIONSHIPS (SIMPLIFIED GPU)")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Convert all data to GPU format once
        self.logger.info("Step 1: Converting data to GPU format...")
        points_data = []
        for point in points:
            points_data.append({
                'point_id': int(point['point_id']),
                'x': float(point['geometry'].x),
                'y': float(point['geometry'].y),
                'parent_id': str(point.get('parent_id', '')),
                'source_segment_idx': int(point['source_segment_idx']),
                'distance_along_segment': float(point.get('distance_along_segment', 0.0)),
                'is_intersection': bool(point.get('is_intersection', False)),
                'is_pedestrian_ramp': bool(point.get('is_pedestrian_ramp', False))
            })

        # Create points GeoDataFrame and convert to cuSpatial
        x, y = zip(*[(point['x'], point['y']) for point in points_data])
        points_gdf = gpd.GeoDataFrame(points_data, geometry=gpd.points_from_xy(x, y))
        points_df = cuspatial.from_geopandas(points_gdf)
        
        # Convert segments to CuDF with all necessary columns
        segments_data = []
        for idx, (_, row) in enumerate(segments_gdf.iterrows()):
            try:
                coords = list(row.geometry.coords)
                if len(coords) >= 2:
                    start_point = coords[0]
                    end_point = coords[-1]
                    
                    segments_data.append({
                        'segment_id': int(idx),
                        'parent_id': str(row.get('parent_id', '')),
                        'segment_type': str(row.get('segment_type', 'sidewalk')),
                        'is_crosswalk': bool(row.get('is_crosswalk', False)),
                        'length': float(row.geometry.length),
                        'start_x': float(start_point[0]),
                        'start_y': float(start_point[1]),
                        'end_x': float(end_point[0]),
                        'end_y': float(end_point[1])
                    })
            except Exception as e:
                self.logger.warning(f"Skipping segment {idx} due to conversion error: {e}")
                continue
        
        if not segments_data:
            self.logger.error("No valid segments found for GPU conversion")
            raise ValueError("No valid segments could be converted to GPU format")
        
        # Create segments GeoDataFrame and convert to cuSpatial
        linestrings = segments_gdf.geometry
        segments_gdf = gpd.GeoDataFrame(segments_data, geometry=linestrings)
        segments_df = cuspatial.from_geopandas(segments_gdf)
        
        self.logger.info(f"Converted {len(points_df)} points and {len(segments_df)} segments to GPU format")
        
        # Step 2: Compute adjacency relationships in single GPU operation
        self.logger.info("Step 2: Computing adjacency relationships...")
        adjacency_results = self._compute_adjacency_gpu_simplified(
            points_df, segments_df, crosswalk_distance
        )
        
        # Step 3: Update points with adjacency information
        self.logger.info("Step 3: Updating points with adjacency information...")
        adjacency_lookup = {result['point_id']: result for result in adjacency_results}
        
        self.logger.info(f"Updating {len(points)} points with {len(adjacency_results)} adjacency results")
        
        # Update points with adjacency information
        updated_points = []
        points_with_adjacency = 0
        
        # Progress logging for point updates
        total_points = len(points)
        log_interval = max(1, total_points // 10)  # Log every 10% progress
        
        for i, point in enumerate(points):
            # Progress logging
            if i % log_interval == 0:
                progress_pct = (i / total_points) * 100
                self.logger.info(f"  Updating points: {i}/{total_points} ({progress_pct:.1f}%)")
            
            point_id = point['point_id']
            adjacency_info = adjacency_lookup.get(point_id, {
                'adjacent_points': '',
                'adjacent_distances': '',
                'adjacency_count': 0
            })
            
            # Add adjacency information to point
            point['adjacent_points'] = adjacency_info['adjacent_points']
            point['adjacent_distances'] = adjacency_info['adjacent_distances']
            point['adjacency_count'] = adjacency_info['adjacency_count']
            
            if adjacency_info['adjacency_count'] > 0:
                points_with_adjacency += 1
            
            updated_points.append(point)
        
        self.logger.info(f"Updated {points_with_adjacency}/{len(points)} points with adjacencies")
        
        # Log final statistics
        total_time = time.time() - start_time
        
        return updated_points

    def _compute_adjacency_gpu_simplified(self, points_df: cudf.DataFrame,
                                         segments_df: cudf.DataFrame,
                                         crosswalk_distance: float) -> List[Dict]:
        """
        Simplified GPU adjacency computation that does everything in one operation.
        
        Parameters:
        -----------
        points_df : cudf.DataFrame
            Points in GPU format
        segments_df : cudf.DataFrame
            Segments in GPU format
        crosswalk_distance : float
            Maximum distance for crosswalk creation
            
        Returns:
        --------
        List[Dict]
            Adjacency results for all points
        """
        self.logger.info("Computing adjacency relationships using simplified GPU pipeline...")
        
        # Step 1: Compute nearest linestrings for all points using cuSpatial
        self.logger.info("Computing nearest linestrings using cuSpatial quadtree...")
        try:
            self.logger.info("Converting data to CPU for cuSpatial operations...")
            # Convert to CPU for cuSpatial operations
            points_df_cpu = points_df.to_geopandas()
            segments_df_cpu = segments_df.to_geopandas()
            
            # Get bounds
            self.logger.info("Computing spatial bounds...")
            points_bounds = points_df_cpu.total_bounds
            segments_bounds = segments_df_cpu.total_bounds
            min_x = min(points_bounds[0], segments_bounds[0])
            min_y = min(points_bounds[1], segments_bounds[1])
            max_x = max(points_bounds[2], segments_bounds[2])
            max_y = max(points_bounds[3], segments_bounds[3])

            # Build quadtree
            self.logger.info("Building cuSpatial quadtree...")
            scale = max(max_x - min_x, max_y - min_y) / (1 << 8)
            max_depth = 8
            max_size = 10
            
            key_to_point, quadtree = cuspatial.quadtree_on_points(
                points_df.geometry,
                min_x, max_x, min_y, max_y,
                scale, max_depth, max_size
            )

            # Get segment bounding boxes
            self.logger.info("Computing segment bounding boxes...")
            line_bboxes = cuspatial.linestring_bounding_boxes(segments_df.geometry, expansion_radius=1.0)
            
            # Join quadtree with segment bounding boxes
            self.logger.info("Joining quadtree with segment bounding boxes...")
            linestring_quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
                quadtree, line_bboxes,
                min_x, max_x, min_y, max_y,
                scale, max_depth
            )
            
            # Compute nearest linestring for each point
            self.logger.info("Computing nearest linestrings for all points...")
            nearest_results = cuspatial.quadtree_point_to_nearest_linestring(
                linestring_quad_pairs,
                quadtree,
                key_to_point,
                points_df.geometry,
                segments_df.geometry
            )

            self.logger.info(f"Computed nearest linestrings for {len(nearest_results)} point-segment pairs")
            nearest_data = {
                'nearest_results': nearest_results,
                'key_to_point': key_to_point
            }
            
        except Exception as e:
            self.logger.error(f"Error computing nearest linestrings: {e}")
            nearest_data = None
        if nearest_data is None:
            self.logger.error("Failed to compute nearest linestrings")
            return []
        
        nearest_results = nearest_data['nearest_results']
        key_to_point = nearest_data['key_to_point']
        self.logger.info(f"Found nearest linestrings for {len(nearest_results)} point-segment pairs")
        
        # Step 2: Compute all adjacencies in a single vectorized operation
        self.logger.info("Computing adjacency relationships for all points...")
        self.logger.info("Step 1: Pre-computing data structures...")
        
        # Create quadtree mapping for efficient lookups. key should be the point_id, value should be the quadtree index.
        quadtree_mapping = key_to_point.to_dict()
        # reverse the keys and values
        quadtree_mapping = {v: k for k, v in quadtree_mapping.items()}
        
        # Group points by type
        segment_points_mask = ~(points_df['is_intersection'].values | 
                              (points_df['is_pedestrian_ramp'].values if 'is_pedestrian_ramp' in points_df.columns else cp.zeros(len(points_df), dtype=bool)))
        intersection_points_mask = ~segment_points_mask
        
        segment_point_indices = cp.where(segment_points_mask)[0]
        intersection_point_indices = cp.where(intersection_points_mask)[0]
        
        self.logger.info(f"Grouped points: {len(segment_point_indices)} segment points, {len(intersection_point_indices)} intersection points")
        
        # Debug: Check intersection point details
        if len(intersection_point_indices) > 0:
            self.logger.info("Debug: Sample intersection points:")

            sample_intersection_indices = random.sample(intersection_point_indices.tolist(), min(3, len(intersection_point_indices)))
            for idx in sample_intersection_indices:
                point_id = int(points_df['point_id'].iloc[idx])
                is_intersection = bool(points_df['is_intersection'].iloc[idx])
                is_pedestrian_ramp = bool(points_df['is_pedestrian_ramp'].iloc[idx]) if 'is_pedestrian_ramp' in points_df.columns else False
                x = float(points_df['x'].iloc[idx])
                y = float(points_df['y'].iloc[idx])
                self.logger.info(f"  Point {point_id}: is_intersection={is_intersection}, is_pedestrian_ramp={is_pedestrian_ramp}, coords=({x:.2f}, {y:.2f})")
        else:
            self.logger.warning("WARNING: No intersection points found! Check if points are being marked as intersections correctly.")
        
        # Step 2: Process segment adjacencies in batch
        self.logger.info("Step 2: Computing segment adjacencies...")
        segment_adjacencies = self._compute_segment_adjacencies_batch(
            points_df, segment_point_indices
        )
        
        # Step 3: Process intersection adjacencies in batch
        self.logger.info("Step 3: Computing intersection adjacencies...")
        intersection_adjacencies = self._compute_intersection_adjacencies_batch(
            points_df, intersection_point_indices, crosswalk_distance
        )
        
        # Step 4: Combine results
        self.logger.info("Step 4: Combining adjacency results...")
        
        # Create adjacency results in the same order as the original points DataFrame
        # This ensures the indices match correctly
        adjacency_results = []
        
        # Initialize all results with empty adjacencies, maintaining DataFrame order
        for i in range(len(points_df)):
            point_id = int(points_df['point_id'].iloc[i])
            adjacency_results.append({
                'point_id': point_id,
                'adjacent_points': '',
                'adjacent_distances': '',
                'adjacency_count': 0
            })
        
        # Create a proper mapping from point_id to DataFrame index
        # This ensures we're using the correct indices that match adjacency_results
        point_id_to_df_idx = {}
        for i in range(len(points_df)):
            point_id = int(points_df['point_id'].iloc[i])
            point_id_to_df_idx[point_id] = i
        
        # Update with segment adjacencies
        for point_id, adjacencies in segment_adjacencies.items():
            if point_id in point_id_to_df_idx:
                idx = point_id_to_df_idx[point_id]
                adjacency_results[idx]['adjacent_points'] = ','.join(map(str, adjacencies['points']))
                adjacency_results[idx]['adjacent_distances'] = ','.join(map(str, adjacencies['distances']))
                adjacency_results[idx]['adjacency_count'] = len(adjacencies['points'])
        
        # Update with intersection adjacencies
        intersection_update_count = 0
        for point_id, adjacencies in intersection_adjacencies.items():
            if point_id in point_id_to_df_idx:
                idx = point_id_to_df_idx[point_id]
                existing_points = adjacency_results[idx]['adjacent_points']
                existing_distances = adjacency_results[idx]['adjacent_distances']
                
                # Combine with existing adjacencies
                new_points = ','.join(map(str, adjacencies['points']))
                new_distances = ','.join(map(str, adjacencies['distances']))
                
                if existing_points:
                    adjacency_results[idx]['adjacent_points'] = f"{existing_points},{new_points}"
                    adjacency_results[idx]['adjacent_distances'] = f"{existing_distances},{new_distances}"
                else:
                    adjacency_results[idx]['adjacent_points'] = new_points
                    adjacency_results[idx]['adjacent_distances'] = new_distances
                
                adjacency_results[idx]['adjacency_count'] += len(adjacencies['points'])
                intersection_update_count += 1
        
        self.logger.info(f"Updated {intersection_update_count} points with intersection adjacencies")
        
        # Log summary statistics
        points_with_adjacencies = sum(1 for result in adjacency_results if result['adjacency_count'] > 0)
        total_adjacencies = sum(result['adjacency_count'] for result in adjacency_results)
        
        # Debug: Print first few adjacency results to verify correctness
        self.logger.info("Debug: Random sample of 5 adjacency results:")
        sample_indices = random.sample(range(len(adjacency_results)), min(5, len(adjacency_results)))
        for idx in sample_indices:
            result = adjacency_results[idx]
            self.logger.info(f"  Point {result['point_id']}: {result['adjacency_count']} adjacencies - {result['adjacent_points']}")
        
        self.logger.info(f"Adjacency computation summary:")
        self.logger.info(f"  - Total points processed: {len(points_df)}")
        self.logger.info(f"  - Points with adjacencies: {points_with_adjacencies}")
        self.logger.info(f"  - Total adjacencies found: {total_adjacencies}")
        self.logger.info(f"  - Average adjacencies per point: {total_adjacencies/len(points_df):.2f}")
        
        self.logger.info(f"Completed adjacency computation for {len(adjacency_results)} points")
        return adjacency_results

    def _compute_segment_adjacencies_batch(self, points_df: cudf.DataFrame,
                                         segment_point_indices: cp.ndarray) -> Dict:
        """
        Compute segment adjacencies for all segment points in batch.
        """
        segment_adjacencies = {}
        
        # Group points by segment for efficient processing
        segment_groups = {}
        for idx in segment_point_indices:
            point_id = int(points_df['point_id'].iloc[idx])
            segment_idx = int(points_df['source_segment_idx'].iloc[idx])
            distance = float(points_df['distance_along_segment'].iloc[idx])
            
            if segment_idx not in segment_groups:
                segment_groups[segment_idx] = []
            segment_groups[segment_idx].append((point_id, distance, idx))
        
        self.logger.info(f"Processing {len(segment_groups)} segments for segment adjacencies")
        
        # Debug: Print random sample of segment groups
        sample_segments = random.sample(list(segment_groups.items()), min(3, len(segment_groups)))
        for segment_idx, points in sample_segments:
            self.logger.info(f"  Segment {segment_idx}: {len(points)} points")
            # Show random sample of points in this segment
            sample_points = random.sample(points, min(3, len(points)))
            for point_id, distance, _ in sample_points:
                self.logger.info(f"    Point {point_id}: distance {distance:.2f}")
        
        # Process each segment
        for segment_idx, points in segment_groups.items():
            if len(points) <= 1:
                continue
                
            # Sort points by distance along segment
            points.sort(key=lambda x: x[1])
            
            # Find adjacencies for each point in this segment
            for i, (point_id, distance, _) in enumerate(points):
                adjacent_points = []
                adjacent_distances = []
                
                # Forward adjacency
                if i + 1 < len(points):
                    next_point_id, next_distance, _ = points[i + 1]
                    adjacent_points.append(next_point_id)
                    adjacent_distances.append(next_distance - distance)
                
                # Backward adjacency
                if i > 0:
                    prev_point_id, prev_distance, _ = points[i - 1]
                    adjacent_points.append(prev_point_id)
                    adjacent_distances.append(distance - prev_distance)
                
                if adjacent_points:
                    segment_adjacencies[point_id] = {
                        'points': adjacent_points,
                        'distances': adjacent_distances
                    }
                    
                    # Debug: Print random sample of adjacencies
                    if len(segment_adjacencies) <= 5:
                        self.logger.info(f"  Segment adjacency for point {point_id}: {adjacent_points} (distances: {adjacent_distances})")
        
        # Debug: Print random sample of segment adjacencies
        if segment_adjacencies:
            sample_adjacencies = random.sample(list(segment_adjacencies.items()), min(3, len(segment_adjacencies)))
            self.logger.info("Debug: Random sample of segment adjacencies:")
            for point_id, adjacencies in sample_adjacencies:
                self.logger.info(f"  Point {point_id}: {adjacencies['points']} (distances: {adjacencies['distances']})")
        
        return segment_adjacencies

    def _compute_intersection_adjacencies_batch(self, points_df: cudf.DataFrame,
                                             intersection_point_indices: cp.ndarray,
                                             crosswalk_distance: float) -> Dict:
        """
        Compute intersection adjacencies using direct point-to-point distance calculation.
        This avoids the complex segment-based lookup that was causing index mismatches.
        """
        intersection_adjacencies = {}
        
        self.logger.info(f"Processing {len(intersection_point_indices)} intersection points using direct distance calculation")
        
        # Extract coordinates for all points
        all_point_coords = []
        all_point_ids = []
        all_is_intersection = []
        
        for idx in range(len(points_df)):
            point_id = int(points_df['point_id'].iloc[idx])
            x = float(points_df['x'].iloc[idx])
            y = float(points_df['y'].iloc[idx])
            is_intersection = bool(points_df['is_intersection'].iloc[idx])
            
            all_point_coords.append((x, y))
            all_point_ids.append(point_id)
            all_is_intersection.append(is_intersection)
        
        # Separate intersection and regular points
        intersection_coords = []
        intersection_ids = []
        regular_coords = []
        regular_ids = []
        
        for i, (coords, point_id, is_intersection) in enumerate(zip(all_point_coords, all_point_ids, all_is_intersection)):
            if is_intersection:
                intersection_coords.append(coords)
                intersection_ids.append(point_id)
            else:
                regular_coords.append(coords)
                regular_ids.append(point_id)
        
        self.logger.info(f"DEBUG: Found {len(intersection_coords)} intersection points and {len(regular_coords)} regular points")
        
        # Process each intersection point
        processed_count = 0
        found_adjacencies_count = 0
        distance_stats = []
        
        for i, (intersection_coord, intersection_id) in enumerate(zip(intersection_coords, intersection_ids)):
            ix, iy = intersection_coord
            
            if processed_count < 3:  # Debug first 3 intersection points
                self.logger.info(f"  DEBUG: Processing intersection point {intersection_id} at ({ix:.2f}, {iy:.2f})")
            
            adjacent_points = []
            adjacent_distances = []
            
            # Find closest regular points within crosswalk_distance
            for regular_coord, regular_id in zip(regular_coords, regular_ids):
                rx, ry = regular_coord
                distance = ((rx - ix) ** 2 + (ry - iy) ** 2) ** 0.5
                distance_stats.append(distance)
                
                if distance <= crosswalk_distance and distance > 0:  # Exclude self
                    adjacent_points.append(regular_id)
                    adjacent_distances.append(distance)
            
            # Also find closest intersection points within crosswalk_distance (intersection-to-intersection connections)
            for other_intersection_coord, other_intersection_id in zip(intersection_coords, intersection_ids):
                if other_intersection_id != intersection_id:  # Don't connect to self
                    ox, oy = other_intersection_coord
                    distance = ((ox - ix) ** 2 + (oy - iy) ** 2) ** 0.5
                    distance_stats.append(distance)
                    
                    if distance <= crosswalk_distance and distance > 0:
                        adjacent_points.append(other_intersection_id)
                        adjacent_distances.append(distance)
            
            # Sort by distance and limit to reasonable number
            if adjacent_points:
                # Sort by distance
                sorted_pairs = sorted(zip(adjacent_distances, adjacent_points))
                sorted_distances, sorted_points = zip(*sorted_pairs)
                
                # Limit to top 5 closest points
                max_adjacencies = 10
                final_distances = list(sorted_distances[:max_adjacencies])
                final_points = list(sorted_points[:max_adjacencies])
                
                intersection_adjacencies[intersection_id] = {
                    'points': final_points,
                    'distances': final_distances
                }
                found_adjacencies_count += 1
                
                if processed_count < 3:  # Debug first few
                    # Count intersection vs regular adjacencies
                    intersection_adj_count = sum(1 for pt_id in final_points if pt_id in intersection_ids)
                    regular_adj_count = len(final_points) - intersection_adj_count
                    
                    self.logger.info(f"    DEBUG: Found {len(final_points)} adjacencies for point {intersection_id} ({intersection_adj_count} intersection, {regular_adj_count} regular)")
                    for j, (pt_id, dist) in enumerate(zip(final_points, final_distances)):
                        pt_type = "intersection" if pt_id in intersection_ids else "regular"
                        self.logger.info(f"      {j+1}: Point {pt_id} ({pt_type}) at distance {dist:.2f} feet")
            
            processed_count += 1
        
        # Debug: Show distance statistics
        if distance_stats:
            import numpy as np
            distance_stats = np.array(distance_stats)
            self.logger.info(f"Distance statistics for intersection points:")
            self.logger.info(f"  - Total distances checked: {len(distance_stats)}")
            self.logger.info(f"  - Min distance: {distance_stats.min():.2f} feet")
            self.logger.info(f"  - Max distance: {distance_stats.max():.2f} feet")
            self.logger.info(f"  - Mean distance: {distance_stats.mean():.2f} feet")
            self.logger.info(f"  - Median distance: {np.median(distance_stats):.2f} feet")
            self.logger.info(f"  - Distances <= {crosswalk_distance} feet: {np.sum(distance_stats <= crosswalk_distance)} ({np.sum(distance_stats <= crosswalk_distance)/len(distance_stats)*100:.1f}%)")
            self.logger.info(f"  - Distances <= 50 feet: {np.sum(distance_stats <= 50)} ({np.sum(distance_stats <= 50)/len(distance_stats)*100:.1f}%)")
            self.logger.info(f"  - Distances <= 25 feet: {np.sum(distance_stats <= 25)} ({np.sum(distance_stats <= 25)/len(distance_stats)*100:.1f}%)")
        
        self.logger.info(f"Intersection processing complete: {found_adjacencies_count}/{len(intersection_coords)} intersection points found adjacencies")
        
        # Calculate adjacency type statistics
        total_intersection_to_intersection = 0
        total_intersection_to_regular = 0
        
        for intersection_id, adjacencies in intersection_adjacencies.items():
            for adj_point_id in adjacencies['points']:
                if adj_point_id in intersection_ids:
                    total_intersection_to_intersection += 1
                else:
                    total_intersection_to_regular += 1
        
        self.logger.info(f"Adjacency type breakdown:")
        self.logger.info(f"  - Intersection-to-intersection: {total_intersection_to_intersection}")
        self.logger.info(f"  - Intersection-to-regular: {total_intersection_to_regular}")
        self.logger.info(f"  - Total adjacencies: {total_intersection_to_intersection + total_intersection_to_regular}")
        
        # Debug: Show sample intersection adjacencies
        if intersection_adjacencies:
            sample_intersection_adjacencies = random.sample(list(intersection_adjacencies.items()), min(3, len(intersection_adjacencies)))
            self.logger.info("Debug: Sample intersection adjacencies:")
            for point_id, adjacencies in sample_intersection_adjacencies:
                intersection_adj_count = sum(1 for pt_id in adjacencies['points'] if pt_id in intersection_ids)
                regular_adj_count = len(adjacencies['points']) - intersection_adj_count
                self.logger.info(f"  Point {point_id}: {len(adjacencies['points'])} total ({intersection_adj_count} intersection, {regular_adj_count} regular)")
        else:
            self.logger.warning("WARNING: No intersection adjacencies found!")
        
        return intersection_adjacencies

    def _classify_relationship_gpu(self, segment_idx1: int, is_intersection1: bool,
                                  segment_idx2: int, is_intersection2: bool) -> str:
        """
        GPU-optimized relationship classification using pre-computed metadata.
        
        Parameters:
        -----------
        segment_idx1 : int
            Source segment index of first point
        is_intersection1 : bool
            Whether first point is an intersection
        segment_idx2 : int
            Source segment index of second point
        is_intersection2 : bool
            Whether second point is an intersection
            
        Returns:
        --------
        str
            Relationship type
        """
        # Same segment
        if segment_idx1 == segment_idx2:
            return 'same_segment'
        
        # Both are intersections
        if is_intersection1 and is_intersection2:
            return 'intersection_intersection'
        
        # One is intersection
        if is_intersection1 or is_intersection2:
            return 'intersection_segment'
        
        # Different segments
        return 'cross_segment'
    

    
    def _create_final_network(self, topology_points: List[Dict], crs) -> gpd.GeoDataFrame:
        """
        Create the final GeoDataFrame with all point network data.
        
        Parameters:
        -----------
        topology_points : List[Dict]
            Points with topology information
        crs : CRS
            Coordinate reference system
            
        Returns:
        --------
        GeoDataFrame
            Final sampled point network
        """
        self.logger.info("Creating final point network GeoDataFrame...")
        
        # Prepare data for GeoDataFrame
        final_data = []
        
        for point in topology_points:
            # Create simplified record for GeoDataFrame
            record = {
                'geometry': point['geometry'],
                'point_id': point['point_id'],
                'source_segment_idx': point['source_segment_idx'],
                'distance_along_segment': point['distance_along_segment'],
                'segment_total_length': point['segment_total_length'],
                'position_ratio': point['position_ratio'],
                'is_intersection': point.get('is_intersection', False),
                'neighbor_count': len(point.get('network_neighbors', [])),
                'buffer_distance': point['buffer_zone'].area / np.pi ** 0.5 if point.get('buffer_zone') else None,
            }
            
            # Add parent_id directly (not as source_parent_id)
            if 'parent_id' in point:
                record['parent_id'] = point['parent_id']
            
            # Add source segment attributes
            for key, value in point.items():
                if key.startswith('source_') and key != 'source_segment_idx':
                    record[key] = value
            
            # Add neighbor information (simplified)
            neighbors = point.get('network_neighbors', [])
            if neighbors:
                record['nearest_neighbor_distance'] = neighbors[0]['distance']
                record['nearest_neighbor_id'] = neighbors[0]['point_id']
            else:
                record['nearest_neighbor_distance'] = None
                record['nearest_neighbor_id'] = None
            
            # Add adjacency information if available
            record['adjacent_points'] = point.get('adjacent_points', '')
            record['adjacent_distances'] = point.get('adjacent_distances', '')
            record['adjacency_count'] = point.get('adjacency_count', 0)
            
            final_data.append(record)
        
        # Create GeoDataFrame
        result = gpd.GeoDataFrame(final_data, crs=crs)
        
        # Add summary statistics as attributes
        if not topology_points:
            summary = {}
        else:
            neighbor_distances = []
            for point in topology_points:
                neighbors = point.get('network_neighbors', [])
                if neighbors:
                    neighbor_distances.append(neighbors[0]['distance'])
            
            summary = {
                'total_points': len(topology_points),
                'intersection_points': sum(1 for pt in topology_points if pt.get('is_intersection', False)),
                'avg_neighbor_distance': np.mean(neighbor_distances) if neighbor_distances else None,
                'min_neighbor_distance': np.min(neighbor_distances) if neighbor_distances else None,
                'max_neighbor_distance': np.max(neighbor_distances) if neighbor_distances else None,
                'unique_segments': len(set(pt['source_segment_idx'] for pt in topology_points)),
                'avg_points_per_segment': len(topology_points) / len(set(pt['source_segment_idx'] for pt in topology_points)),
            }
        result.attrs['sampling_summary'] = summary
        
        self.logger.info(f"Created final network with {len(result)} points")
        self.logger.info(f"Final columns: {list(result.columns)}")
        
        # Log adjacency column info
        if 'adjacent_points' in result.columns:
            non_empty_adjacencies = result[result['adjacent_points'] != '']
            self.logger.info(f"Points with adjacencies: {len(non_empty_adjacencies)}/{len(result)}")
            if len(non_empty_adjacencies) > 0:
                self.logger.info(f"Sample adjacent_points: {non_empty_adjacencies['adjacent_points'].iloc[0]}")
        else:
            self.logger.warning("adjacent_points column not found in final result!")
        
        return result
    
    def _compute_sampling_summary(self, points: List[Dict]) -> Dict:
        """
        Compute summary statistics for the sampling process.
        
        Parameters:
        -----------
        points : List[Dict]
            Point records
            
        Returns:
        --------
        Dict
            Summary statistics
        """
        if not points:
            return {}
        
        # Extract distances to nearest neighbors
        neighbor_distances = []
        for point in points:
            neighbors = point.get('network_neighbors', [])
            if neighbors:
                neighbor_distances.append(neighbors[0]['distance'])
        
        summary = {
            'total_points': len(points),
            'intersection_points': sum(1 for pt in points if pt.get('is_intersection', False)),
            'avg_neighbor_distance': np.mean(neighbor_distances) if neighbor_distances else None,
            'min_neighbor_distance': np.min(neighbor_distances) if neighbor_distances else None,
            'max_neighbor_distance': np.max(neighbor_distances) if neighbor_distances else None,
            'unique_segments': len(set(pt['source_segment_idx'] for pt in points)),
            'avg_points_per_segment': len(points) / len(set(pt['source_segment_idx'] for pt in points)),
        }
        
        return summary
    
    def _create_empty_result(self, crs) -> gpd.GeoDataFrame:
        """
        Create an empty result GeoDataFrame with the correct schema.
        
        Parameters:
        -----------
        crs : CRS
            Coordinate reference system
            
        Returns:
        --------
        GeoDataFrame
            Empty result with correct columns
        """
        return gpd.GeoDataFrame(
            columns=[
                'geometry', 'point_id', 'source_segment_idx', 'distance_along_segment',
                'segment_total_length', 'position_ratio', 'is_intersection',
                'neighbor_count', 'buffer_distance', 'nearest_neighbor_distance',
                'nearest_neighbor_id', 'adjacent_points', 'adjacent_distances', 'adjacency_count'
            ],
            crs=crs
        )
    
    def _log_final_statistics(self, result: gpd.GeoDataFrame, 
                             candidate_points: List[Dict], 
                             total_time: float):
        """
        Log final statistics about the sampling process.
        
        Parameters:
        -----------
        result : GeoDataFrame
            Final result
        candidate_points : List[Dict]
            Original candidate points
        total_time : float
            Total processing time
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("GEOMETRIC SAMPLING COMPLETED")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Processing time: {total_time:.2f} seconds")
        self.logger.info(f"Candidate points: {len(candidate_points)}")
        self.logger.info(f"Final sampled points: {len(result)}")
        self.logger.info(f"Filtering efficiency: {len(result)/len(candidate_points)*100:.1f}%")
        
        if len(result) > 0:
            summary = result.attrs.get('sampling_summary', {})
            
            self.logger.info(f"Network statistics:")
            self.logger.info(f"  - Total points: {summary.get('total_points', 'N/A')}")
            self.logger.info(f"  - Intersection points: {summary.get('intersection_points', 'N/A')}")
            self.logger.info(f"  - Unique segments: {summary.get('unique_segments', 'N/A')}")
            self.logger.info(f"  - Avg points per segment: {summary.get('avg_points_per_segment', 'N/A'):.2f}")
            
            if summary.get('avg_neighbor_distance'):
                self.logger.info(f"  - Avg neighbor distance: {summary['avg_neighbor_distance']:.2f} feet")
                self.logger.info(f"  - Min neighbor distance: {summary['min_neighbor_distance']:.2f} feet")
                self.logger.info(f"  - Max neighbor distance: {summary['max_neighbor_distance']:.2f} feet")
        
        self.logger.info("=" * 60)

    

    
    
    
    
    
    def process_segments_to_sampled_network(self, segments_input: str,
                                          centerlines_path: str = "/share/ju/sidewalk_utils/data/nyc/processed/nyc_sidewalk_centerlines.parquet",
                                          buffer_distance: float = 100.0,
                                          sampling_interval: float = 50.0,
                                          strategy: str = "uniform",
                                          output_path: Optional[str] = None,
                                          save_intermediate: bool = False,
                                          intermediate_dir: Optional[str] = None,
                                          pedestrian_ramps_path: Optional[str] = None,
                                          compute_adjacency: bool = True,
                                          walkshed_distance: float = 328.0,
                                          crosswalk_distance: float = 100.0) -> gpd.GeoDataFrame:
        """
        Master function to process segments into a complete sampled point network using uniform sampling.
        
        Parameters:
        -----------
        segments_input : str
            Path to neighborhood segments parquet file (for point sampling)
        centerlines_path : str, default="/share/ju/sidewalk_utils/data/nyc/processed/nyc_sidewalk_centerlines.parquet"
            Path to citywide centerlines geoparquet file (for adjacency computation)
        buffer_distance : float
            Minimum distance between points
        sampling_interval : float
            Initial spacing between candidate points
        strategy : str
            Sampling strategy (only "uniform" is supported)
        output_path : Optional[str]
            Path to save the final network
        save_intermediate : bool
            Whether to save intermediate results
        intermediate_dir : Optional[str]
            Directory for intermediate files
        pedestrian_ramps_path : Optional[str]
            Path to pedestrian ramps GeoJSON file
        compute_adjacency : bool, default=True
            Whether to compute adjacency relationships
        walkshed_distance : float, default=328.0
            Walkshed radius in feet (100 meters)
        crosswalk_distance : float, default=100.0
            Maximum distance for crosswalk creation in feet
            
        Returns:
        --------
        GeoDataFrame
            Complete sampled point network
        """
        start_time = time.time()
        
        # Step 1: Load and validate input
        self.logger.info("Step 1: Loading neighborhood segments for point sampling...")
        
        # Load neighborhood segments for point sampling
        segments_gdf = gpd.read_parquet(segments_input)
        self.logger.info(f"Loaded {len(segments_gdf)} neighborhood segments from: {segments_input}")
        
        # Load citywide centerlines for adjacency computation
        self.logger.info("Step 1b: Loading citywide centerlines for adjacency computation...")
        citywide_centerlines = self._load_centerlines(centerlines_path)
        if citywide_centerlines is None:
            self.logger.error("Failed to load citywide centerlines")
            return self._create_empty_result(PROJ_FT)
        
        # Load intersection points (pedestrian ramps) for centerline filtering
        intersection_points = None
        pedestrian_ramps_points = None  # Store for later use in inspection files
        if pedestrian_ramps_path:
            self.logger.info("Step 1b.1: Loading intersection points for centerline filtering...")
            ramps_gdf = gpd.read_file(pedestrian_ramps_path)
            ramps_gdf = ramps_gdf.to_crs(PROJ_FT)
            
            # Filter ramps to neighborhood area
            filtered_ramps = self._filter_pedestrian_ramps(ramps_gdf, segments_gdf, buffer_distance)
            
            # Convert to list of dicts for centerline filtering
            intersection_points = []
            for idx, row in filtered_ramps.iterrows():
                intersection_points.append({
                    'geometry': row.geometry,
                    'point_id': f"ramp_{idx}",
                    'is_intersection': True,
                    'is_pedestrian_ramp': True
                })
            
            # Store pedestrian ramps points for later use in inspection files
            pedestrian_ramps_points = self._load_and_process_pedestrian_ramps(pedestrian_ramps_path, segments_gdf)
            
            self.logger.info(f"Loaded {len(intersection_points)} intersection points for centerline filtering")
        
        # Filter citywide centerlines to only include those relevant to neighborhood segments AND intersection points
        self.logger.info("Step 1c: Filtering citywide centerlines to neighborhood area and intersection points...")
        filtered_centerlines = self._filter_centerlines_to_neighborhood(
            citywide_centerlines, segments_gdf, buffer_distance, 
            intersection_points=intersection_points, crosswalk_distance=crosswalk_distance
        )
        self.logger.info(f"Filtered citywide centerlines: {len(citywide_centerlines)} -> {len(filtered_centerlines)} segments")
        
        if save_intermediate and intermediate_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            segments_path = os.path.join(intermediate_dir, f"01_neighborhood_segments_{timestamp}.parquet")
            segments_gdf.to_parquet(segments_path)
            self.logger.info(f"Saved neighborhood segments to: {segments_path}")
            
            centerlines_path = os.path.join(intermediate_dir, f"01_filtered_centerlines_{timestamp}.parquet")
            filtered_centerlines.to_parquet(centerlines_path)
            self.logger.info(f"Saved filtered centerlines to: {centerlines_path}")
        
        # Step 2: Generate sampled network using neighborhood segments
        self.logger.info("Step 2: Generating sampled point network from neighborhood segments...")
        sampled_network = self.generate_sampled_network(
            segments_gdf=segments_gdf,
            buffer_distance=buffer_distance,
            sampling_interval=sampling_interval,
            strategy=strategy,
            pedestrian_ramps_path=pedestrian_ramps_path,
            compute_adjacency=compute_adjacency,
            walkshed_distance=walkshed_distance,
            crosswalk_distance=crosswalk_distance,
            adjacency_centerlines=filtered_centerlines  # Pass filtered centerlines for adjacency
        )
        
        if save_intermediate and intermediate_dir:
            intermediate_path = os.path.join(intermediate_dir, f"02_sampled_network_{timestamp}.parquet")
            sampled_network.to_parquet(intermediate_path)
            self.logger.info(f"Saved sampled network to: {intermediate_path}")
        
        # Step 3: Use sampled network directly (adjacency computation removed)
        self.logger.info("Step 3: Using sampled network directly (adjacency computation removed)...")
        final_network = sampled_network
        
        # Step 4: Save final result and intermediate files for inspection
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        if output_path:
            base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
            extension = output_path.rsplit('.', 1)[1] if '.' in output_path else 'parquet'
            dynamic_output_path = f"{base_path}_sampled_{timestamp}.{extension}"
            
            # Prepare for saving (remove only complex objects that can't be serialized)
            save_network = final_network.copy()
            # Remove complex objects that can't be serialized (but keep adjacency columns)
            if 'network_neighbors' in save_network.columns:
                save_network = save_network.drop(columns=['network_neighbors'])
            save_network.to_parquet(dynamic_output_path)
            self.logger.info(f"Saved final network to: {dynamic_output_path}")
        
        # Save intermediate files for inspection if requested
        if save_intermediate:
            # Determine output directory
            if intermediate_dir:
                output_dir = intermediate_dir
            elif output_path:
                output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
            else:
                output_dir = '.'
            
            # Save sampled network with centerlines for inspection
            self.logger.info(f"DEBUG: compute_adjacency={compute_adjacency}, pedestrian_ramps_points={'exists' if pedestrian_ramps_points else 'None'}, pedestrian_ramps_path={pedestrian_ramps_path}")
            
            # Check if we have combined centerlines from the main flow (including crosswalks)
            main_flow_centerlines = final_network.attrs.get('combined_centerlines_with_crosswalks')
            
            if main_flow_centerlines is not None and len(main_flow_centerlines) > 0:
                self.logger.info(f"DEBUG: Using combined centerlines from main flow ({len(main_flow_centerlines)} total centerlines including crosswalks)")
                combined_centerlines = main_flow_centerlines
                
                # Extract crosswalk centerlines for separate inspection file
                if 'is_crosswalk' in combined_centerlines.columns:
                    crosswalk_centerlines = combined_centerlines[combined_centerlines['is_crosswalk'] == True].copy()
                    self.logger.info(f"DEBUG: Extracted {len(crosswalk_centerlines)} crosswalk centerlines from main flow")
                else:
                    crosswalk_centerlines = gpd.GeoDataFrame(columns=['geometry'], crs=segments_gdf.crs)
                    self.logger.info("DEBUG: No is_crosswalk column found, creating empty crosswalk centerlines")
                
            elif compute_adjacency and (pedestrian_ramps_points or pedestrian_ramps_path):
                self.logger.info("DEBUG: Main flow centerlines not available, regenerating crosswalks for inspection...")
                
                # If we don't have pedestrian_ramps_points but have a path, load them now
                if not pedestrian_ramps_points and pedestrian_ramps_path:
                    self.logger.info(f"DEBUG: Loading pedestrian ramps points from path for inspection...")
                    pedestrian_ramps_points = self._load_and_process_pedestrian_ramps(pedestrian_ramps_path, segments_gdf)
                    self.logger.info(f"DEBUG: Loaded {len(pedestrian_ramps_points) if pedestrian_ramps_points else 0} pedestrian ramps points")
                
                # Debug pedestrian_ramps_points
                if pedestrian_ramps_points:
                    self.logger.info(f"DEBUG: Have {len(pedestrian_ramps_points)} pedestrian ramps points for crosswalk generation")
                else:
                    self.logger.warning(f"DEBUG: No pedestrian ramps points available for crosswalk generation")
                
                # Generate crosswalk centerlines for inspection
                crosswalk_centerlines = self._generate_intersection_centerlines(pedestrian_ramps_points, crosswalk_distance)
                
                # Debug crosswalk centerlines result
                self.logger.info(f"DEBUG: Generated {len(crosswalk_centerlines)} crosswalk centerlines")
                if len(crosswalk_centerlines) > 0:
                    self.logger.info(f"DEBUG: Sample crosswalk centerlines:")
                    for i, (_, row) in enumerate(crosswalk_centerlines.head(3).iterrows()):
                        self.logger.info(f"  {i+1}: {row['segment_id']} - {row['segment_length']:.2f} feet")
                
                # Combine original centerlines with crosswalk centerlines
                if len(crosswalk_centerlines) > 0:
                    self.logger.info(f"DEBUG: Combining {len(filtered_centerlines)} original centerlines with {len(crosswalk_centerlines)} crosswalk centerlines")
                    combined_centerlines = pd.concat([filtered_centerlines, crosswalk_centerlines], ignore_index=True)
                    self.logger.info(f"Combined {len(filtered_centerlines)} original centerlines with {len(crosswalk_centerlines)} crosswalk centerlines for inspection")
                    self.logger.info(f"DEBUG: Combined centerlines total: {len(combined_centerlines)}")
                else:
                    combined_centerlines = filtered_centerlines
                    self.logger.info(f"Using {len(filtered_centerlines)} original centerlines for inspection (no crosswalks generated)")
                    
            else:
                self.logger.info("DEBUG: No adjacency computation or pedestrian ramps, using filtered centerlines only")
                combined_centerlines = filtered_centerlines
                crosswalk_centerlines = gpd.GeoDataFrame(columns=['geometry'], crs=segments_gdf.crs)
            
            # Save combined centerlines for inspection
            centerlines_inspection_path = os.path.join(output_dir, f"03_combined_centerlines_{timestamp}.parquet")
            combined_centerlines['segment_id'] = combined_centerlines['segment_id'].astype(str)
            combined_centerlines.to_parquet(centerlines_inspection_path)
            self.logger.info(f"Saved combined centerlines for inspection to: {centerlines_inspection_path}")
            
            # Save crosswalk centerlines separately for inspection
            if len(crosswalk_centerlines) > 0:
                crosswalk_inspection_path = os.path.join(output_dir, f"03_crosswalk_centerlines_{timestamp}.parquet")
                crosswalk_centerlines.to_parquet(crosswalk_inspection_path)
                self.logger.info(f"Saved crosswalk centerlines for inspection to: {crosswalk_inspection_path}")
                
                # Save centerlines network WITH intersection point-to-point connections
                centerlines_with_intersections_path = os.path.join(output_dir, f"03_centerlines_with_intersections_{timestamp}.parquet")
                combined_centerlines.to_parquet(centerlines_with_intersections_path)
                self.logger.info(f"Saved centerlines network with intersection connections to: {centerlines_with_intersections_path}")
                
                # Calculate crosswalk count more carefully
                if main_flow_centerlines is not None:
                    original_count = len(main_flow_centerlines) - len(crosswalk_centerlines)
                    self.logger.info(f"DEBUG: 03_centerlines_with_intersections file contains {len(combined_centerlines)} total centerlines ({original_count} original + {len(crosswalk_centerlines)} crosswalks)")
                else:
                    self.logger.info(f"DEBUG: 03_centerlines_with_intersections file contains {len(combined_centerlines)} total centerlines (includes {len(crosswalk_centerlines)} crosswalks)")
            else:
                self.logger.info("No crosswalk centerlines found - saving centerlines without crosswalks")
                
                # Save centerlines as the network with intersections (no connections to add)
                centerlines_with_intersections_path = os.path.join(output_dir, f"03_centerlines_with_intersections_{timestamp}.parquet")
                combined_centerlines.to_parquet(centerlines_with_intersections_path)
                self.logger.info(f"Saved centerlines network (no intersection connections) to: {centerlines_with_intersections_path}")
                self.logger.info(f"DEBUG: 03_centerlines_with_intersections file contains {len(combined_centerlines)} centerlines only")
            
            # Save sampled network with centerlines for inspection
            network_with_centerlines_path = os.path.join(output_dir, f"03_network_with_centerlines_{timestamp}.parquet")
            
            # Create a combined GeoDataFrame with both points and centerlines
            # Convert points to a format that can be combined with centerlines
            points_for_inspection = final_network.copy()
            points_for_inspection['feature_type'] = 'point'
            points_for_inspection['feature_id'] = points_for_inspection['point_id'].astype(str)
            
            # Prepare centerlines for combination
            centerlines_for_inspection = combined_centerlines.copy()
            centerlines_for_inspection['feature_type'] = 'centerline'
            centerlines_for_inspection['feature_id'] = centerlines_for_inspection['segment_id'].astype(str)
            
            # Combine points and centerlines
            combined_inspection = pd.concat([points_for_inspection, centerlines_for_inspection], ignore_index=True)
            combined_inspection['parent_id'] = combined_inspection['parent_id'].astype(str)
            combined_inspection.to_parquet(network_with_centerlines_path)
            self.logger.info(f"Saved network with centerlines for inspection to: {network_with_centerlines_path}")
        else:
            # Save just the original centerlines for inspection even if adjacency is disabled
            centerlines_inspection_path = os.path.join(output_dir, f"03_original_centerlines_{timestamp}.parquet")
            filtered_centerlines.to_parquet(centerlines_inspection_path)
            self.logger.info(f"Saved original centerlines for inspection to: {centerlines_inspection_path}")
            
            # Save centerlines network with intersection connections (even if adjacency is disabled)
            centerlines_with_intersections_path = os.path.join(output_dir, f"03_centerlines_with_intersections_{timestamp}.parquet")
            filtered_centerlines.to_parquet(centerlines_with_intersections_path)
            self.logger.info(f"Saved centerlines network (no intersection connections) to: {centerlines_with_intersections_path}")
            
            # Save sampled network with original centerlines for inspection
            network_with_centerlines_path = os.path.join(output_dir, f"03_network_with_centerlines_{timestamp}.parquet")
            
            # Create a combined GeoDataFrame with both points and centerlines
            points_for_inspection = final_network.copy()
            points_for_inspection['feature_type'] = 'point'
            points_for_inspection['feature_id'] = points_for_inspection['point_id'].astype(str)
            
            # Prepare centerlines for combination
            centerlines_for_inspection = filtered_centerlines.copy()
            centerlines_for_inspection['feature_type'] = 'centerline'
            centerlines_for_inspection['feature_id'] = centerlines_for_inspection['segment_id'].astype(str)
            
            # Combine points and centerlines
            combined_inspection = pd.concat([points_for_inspection, centerlines_for_inspection], ignore_index=True)
            combined_inspection['parent_id'] = combined_inspection['parent_id'].astype(str)
            combined_inspection.to_parquet(network_with_centerlines_path)
            self.logger.info(f"Saved network with original centerlines for inspection to: {network_with_centerlines_path}")
        
        # Final statistics
        total_time = time.time() - start_time
        self.logger.info(f"\nTotal processing time: {total_time:.2f} seconds")
        self.logger.info(f"Final network: {len(final_network)} points")
        
        return final_network


    
    # Removed _load_segments method - no longer needed with block-level processing

    def _filter_centerlines_to_neighborhood(self, centerlines_gdf: gpd.GeoDataFrame, neighborhood_segments: gpd.GeoDataFrame, buffer_distance: float = 100.0, intersection_points: Optional[List[Dict]] = None, crosswalk_distance: float = 100.0) -> gpd.GeoDataFrame:
        """
        Filter citywide centerlines to only include those relevant to neighborhood segments AND intersection points.
        
        Parameters:
        -----------
        centerlines_gdf : GeoDataFrame
            Citywide centerlines GeoDataFrame
        neighborhood_segments : GeoDataFrame
            Neighborhood segments GeoDataFrame
        buffer_distance : float
            Maximum distance to consider a centerline relevant to neighborhood
        intersection_points : Optional[List[Dict]]
            List of intersection points (pedestrian ramps) to consider
        crosswalk_distance : float
            Maximum distance for intersection point connections
            
        Returns:
        --------
        GeoDataFrame
            Filtered centerlines relevant to neighborhood and intersection points
        """
        # Create a spatial index for the neighborhood segments
        segment_tree = STRtree(list(neighborhood_segments.geometry))
        
        # Find centerlines that are within buffer_distance of any neighborhood segment
        filtered_centerlines = set()
        
        # First, find centerlines near neighborhood segments
        for idx, centerline_geom in enumerate(centerlines_gdf.geometry):
            # Find neighborhood segments within buffer distance
            nearby_segments = segment_tree.query(centerline_geom.buffer(buffer_distance))
            
            if len(nearby_segments) > 0:
                # Check actual distances to nearby segments
                min_distance = float('inf')
                for segment_idx in nearby_segments:
                    segment_geom = neighborhood_segments.iloc[segment_idx].geometry
                    distance = centerline_geom.distance(segment_geom)
                    min_distance = min(min_distance, distance)
                
                # Keep centerline if it's within buffer_distance
                if min_distance <= buffer_distance:
                    filtered_centerlines.add(idx)
        
        # Then, find centerlines near intersection points (if provided)
        if intersection_points:
            self.logger.info(f"Adding centerlines near {len(intersection_points)} intersection points...")
            
            # Create spatial index for centerlines
            centerline_tree = STRtree(list(centerlines_gdf.geometry))
            
            for point in intersection_points:
                point_geom = point['geometry']
                
                # Find centerlines within crosswalk distance of this intersection point
                nearby_centerlines = centerline_tree.query(point_geom.buffer(crosswalk_distance))
                
                for centerline_idx in nearby_centerlines:
                    centerline_geom = centerlines_gdf.iloc[centerline_idx].geometry
                    distance = point_geom.distance(centerline_geom)
                    
                    if distance <= crosswalk_distance:
                        filtered_centerlines.add(centerline_idx)
            
            self.logger.info(f"Added {len(filtered_centerlines)} centerlines total (neighborhood + intersection connections)")
        
        # Return filtered GeoDataFrame
        if filtered_centerlines:
            return centerlines_gdf.iloc[list(filtered_centerlines)].reset_index(drop=True)
        else:
            # Return empty GeoDataFrame with same columns
            return centerlines_gdf.iloc[:0].copy()

    def _load_centerlines(self, centerlines_path: str) -> gpd.GeoDataFrame:
        """
        Load pre-generated centerlines from geoparquet file.
        
        Parameters:
        -----------
        centerlines_path : str
            Path to centerlines geoparquet file
            
        Returns:
        --------
        GeoDataFrame
            Loaded centerlines with required columns for processing
        """
        try:
            self.logger.info(f"Loading centerlines from: {centerlines_path}")
            
            # Load centerlines
            centerlines_gdf = gpd.read_parquet(centerlines_path)
            
            if centerlines_gdf is None or len(centerlines_gdf) == 0:
                self.logger.error("No centerlines found in file")
                return None
            
            # Add required columns for compatibility
            centerlines_gdf['width'] = 10.0  # Default width for block-level centerlines
            centerlines_gdf['segment_id'] = range(len(centerlines_gdf))
            centerlines_gdf['segment_length'] = centerlines_gdf.geometry.length
            centerlines_gdf['segment_type'] = centerlines_gdf.geometry.geom_type
            
            # Ensure parent_id exists
            if 'parent_id' not in centerlines_gdf.columns:
                centerlines_gdf['parent_id'] = centerlines_gdf['segment_id'].astype(str)
            
            self.logger.info(f"Successfully loaded {len(centerlines_gdf)} centerlines")
            self.logger.info(f"  - Average length: {centerlines_gdf['segment_length'].mean():.2f} feet")
            self.logger.info(f"  - Total length: {centerlines_gdf['segment_length'].sum():.2f} feet")
            
            return centerlines_gdf
            
        except Exception as e:
            self.logger.error(f"Error loading centerlines: {e}")
            return None
    
    def _generate_intersection_centerlines(self, intersection_points: List[Dict], crosswalk_distance: float = 100.0) -> gpd.GeoDataFrame:
        """
        Generate crosswalk centerlines BETWEEN pairs of intersection points.
        These represent zebra crossings at intersections.
        
        Parameters:
        -----------
        intersection_points : List[Dict]
            List of intersection point records with geometry
        crosswalk_distance : float
            Maximum distance for crosswalk connections
            
        Returns:
        --------
        GeoDataFrame
            New crosswalk centerline segments connecting intersection points
        """
        self.logger.info(f"Generating crosswalk centerlines between {len(intersection_points)} intersection points...")
        
        # Debug: Check if intersection_points is empty or None
        if not intersection_points:
            self.logger.warning("No intersection points provided - cannot generate crosswalk centerlines")
            return gpd.GeoDataFrame(columns=['geometry', 'width', 'segment_id', 'segment_length', 'segment_type', 'parent_id', 'is_crosswalk'], crs=PROJ_FT)
        
        # Debug: Show sample intersection points
        sample_points = intersection_points[:3] if len(intersection_points) >= 3 else intersection_points
        self.logger.info("Debug: Sample intersection points:")
        for i, point in enumerate(sample_points):
            self.logger.info(f"  Point {i}: ID={point.get('point_id', 'unknown')}, geom=({point['geometry'].x:.2f}, {point['geometry'].y:.2f})")
        
        new_centerlines = []
        processed_pairs = set()  # Track processed pairs to avoid duplicates
        
        # Create spatial index for intersection points
        point_geoms = [point['geometry'] for point in intersection_points]
        point_tree = STRtree(point_geoms)
        
        distances_checked = []  # Track distances for debugging
        
        # Find pairs of intersection points within crosswalk distance
        for i, point1 in enumerate(intersection_points):
            point1_geom = point1['geometry']
            point1_id = point1['point_id']
            
            # Find nearby intersection points
            nearby_points = point_tree.query(point1_geom.buffer(crosswalk_distance))
            
            for j in nearby_points:
                if i != j:  # Don't connect to self
                    point2 = intersection_points[j]
                    point2_geom = point2['geometry']
                    point2_id = point2['point_id']
                    
                    # Calculate distance between intersection points
                    distance = point1_geom.distance(point2_geom)
                    distances_checked.append(distance)
                    
                    if distance <= crosswalk_distance and distance > 0:  # Don't create zero-length lines
                        # Create a unique pair identifier to avoid duplicates (use sorted IDs)
                        pair_id = tuple(sorted([point1_id, point2_id]))
                        
                        if pair_id not in processed_pairs:
                            processed_pairs.add(pair_id)
                            
                            # Create a crosswalk centerline between the two intersection points
                            # Fix: Extract coordinates from Point geometries properly
                            point1_coords = (point1_geom.x, point1_geom.y)
                            point2_coords = (point2_geom.x, point2_geom.y)
                            crosswalk_centerline = LineString([point1_coords, point2_coords])
                            
                            # Create centerline record with all required columns
                            centerline_record = {
                                'geometry': crosswalk_centerline,
                                'width': 10.0,  # Default width for crosswalks
                                'segment_id': f"crosswalk_{point1_id}_to_{point2_id}",
                                'segment_length': crosswalk_centerline.length,
                                'segment_type': 'LineString',
                                'parent_id': f"crosswalk_{point1_id}_{point2_id}",
                                'source_intersection_id': point1_id,
                                'target_intersection_id': point2_id,
                                'crosswalk_distance': distance,
                                'is_crosswalk': True  # Add missing column
                            }
                            
                            new_centerlines.append(centerline_record)
                            
                            # Debug: Log first few crosswalks created
                            if len(new_centerlines) <= 5:
                                self.logger.info(f"  Created crosswalk {len(new_centerlines)}: {point1_id} -> {point2_id} (distance: {distance:.2f} feet)")
        
        # Debug: Show distance statistics
        if distances_checked:
            import numpy as np
            distances_array = np.array(distances_checked)
            self.logger.info(f"Distance statistics:")
            self.logger.info(f"  - Total distances checked: {len(distances_checked)}")
            self.logger.info(f"  - Min distance: {distances_array.min():.2f} feet")
            self.logger.info(f"  - Max distance: {distances_array.max():.2f} feet")
            self.logger.info(f"  - Mean distance: {distances_array.mean():.2f} feet")
            self.logger.info(f"  - Distances <= {crosswalk_distance} feet: {np.sum(distances_array <= crosswalk_distance)} ({np.sum(distances_array <= crosswalk_distance)/len(distances_array)*100:.1f}%)")
        
        if new_centerlines:
            # Convert to GeoDataFrame
            crosswalk_centerlines = gpd.GeoDataFrame(new_centerlines, crs=PROJ_FT)
            self.logger.info(f"Generated {len(crosswalk_centerlines)} crosswalk centerlines")
            self.logger.info(f"  - Average length: {crosswalk_centerlines['segment_length'].mean():.2f} feet")
            self.logger.info(f"  - Total length: {crosswalk_centerlines['segment_length'].sum():.2f} feet")
            
            # Debug: Show first few generated centerlines
            self.logger.info("Debug: First few crosswalk centerlines:")
            for i, (_, row) in enumerate(crosswalk_centerlines.head(3).iterrows()):
                self.logger.info(f"  {i+1}: {row['segment_id']} - length {row['segment_length']:.2f} feet")
            
            return crosswalk_centerlines
        else:
            self.logger.warning("No crosswalk centerlines generated!")
            self.logger.warning(f"  - Intersection points: {len(intersection_points)}")
            self.logger.warning(f"  - Crosswalk distance threshold: {crosswalk_distance} feet")
            if distances_checked:
                self.logger.warning(f"  - Minimum distance between points: {min(distances_checked):.2f} feet")
            return gpd.GeoDataFrame(columns=['geometry', 'width', 'segment_id', 'segment_length', 'segment_type', 'parent_id', 'is_crosswalk'], crs=PROJ_FT)



if __name__ == "__main__":
    # Use Google Fire for CLI
    fire.Fire(SampledPointNetwork) 
   