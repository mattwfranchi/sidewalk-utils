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

# Try to import project constants
try:
    from data.nyc.c import PROJ_FT
except ImportError:
    print("WARNING: PROJ_FT not found. Using default NYC projection.")
    PROJ_FT = 'EPSG:2263'  # Default NYC projection


class SampledPointNetwork:
    """
    Geometric sampling-based point network generator that creates spatially uniform point distributions
    along sidewalk segments using buffer zones and minimum distance thresholds.
    
    This class implements an alternative to H3-based approaches by using geometric sampling with
    buffer zones to ensure spatial uniformity and prevent MAUP (Modifiable Areal Unit Problem).
    
    Key features:
    - **Buffer-based Sampling**: Creates exclusion zones around placed points
    - **Minimum Distance Thresholds**: Maintains spatial separation between points
    - **Network Topology Respect**: Preserves the structure and connectivity of the network
    - **Intersection Handling**: Prevents clustering at intersections where multiple segments join
    - **Spatial Uniformity**: Ensures even distribution of points across the network
    - **MAUP Prevention**: Avoids artifacts from arbitrary spatial units
    
    Sampling Strategies:
    1. **Uniform Sampling**: Points placed at regular intervals along segments
    2. **Adaptive Sampling**: Density adapts to local network characteristics
    3. **Intersection-Aware**: Special handling for segment intersections
    4. **Topology-Preserving**: Maintains network connectivity information
    """
    
    def __init__(self):
        # Use proper logger from logger.py
        self.logger = get_logger("SampledPointNetwork")
        
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
                                pedestrian_ramps_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Generate geometrically sampled point network from sidewalk segments.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input sidewalk segments with LineString geometries (must include 'parent_id' column)
        buffer_distance : float, default=100.0
            Minimum distance between points in feet (exclusion zone radius)
        sampling_interval : float, default=50.0
            Initial spacing between candidate points in feet
        strategy : str, default="uniform"
            Sampling strategy: "uniform", "adaptive", "intersection_aware"
        preserve_intersections : bool, default=True
            Whether to preserve intersection points in the network
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
        sampling_params = self._initialize_sampling_parameters(
            buffer_distance, sampling_interval, strategy, preserve_intersections
        )
        
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
            intersection_processed = self._process_intersection_points_direct(
                filtered_points, validated_segments, sampling_params
            )
        else:
            intersection_processed = filtered_points
        
        # Step 7: Compute network topology
        self.logger.info("Step 7: Computing network topology...")
        topology_enhanced = self._compute_network_topology_gpu(
            intersection_processed, sampling_params
        )
        
        # Step 8: Create final GeoDataFrame
        self.logger.info("Step 8: Creating final sampled point network...")
        result = self._create_final_network(topology_enhanced, segments_gdf.crs)
        
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

    def _initialize_sampling_parameters(self, buffer_distance: float, 
                                       sampling_interval: float,
                                       strategy: str,
                                       preserve_intersections: bool) -> Dict:
        """
        Initialize sampling parameters and validate inputs.
        
        Parameters:
        -----------
        buffer_distance : float
            Minimum distance between points
        sampling_interval : float
            Initial spacing between candidate points
        strategy : str
            Sampling strategy
        preserve_intersections : bool
            Whether to preserve intersection points
            
        Returns:
        --------
        Dict
            Sampling parameters dictionary
        """
        # Validate strategy
        valid_strategies = ["uniform", "adaptive", "intersection_aware"]
        if strategy not in valid_strategies:
            self.logger.warning(f"WARNING: Invalid strategy '{strategy}'. Using 'uniform'")
            strategy = "uniform"
        
        # Ensure buffer distance is larger than sampling interval
        if buffer_distance <= sampling_interval:
            self.logger.warning(f"WARNING: Buffer distance ({buffer_distance}) should be larger than sampling interval ({sampling_interval})")
            buffer_distance = sampling_interval * 2.0
            self.logger.info(f"Adjusted buffer distance to {buffer_distance} feet")
        
        params = {
            'buffer_distance': buffer_distance,
            'sampling_interval': sampling_interval,
            'strategy': strategy,
            'preserve_intersections': preserve_intersections,
            'buffer_distance_squared': buffer_distance ** 2,  # Pre-calculate for efficiency
            'intersection_buffer_factor': 0.02,  # Much smaller buffer around intersections (15 feet)
            'adaptive_density_threshold': 0.7,  # For adaptive sampling
            'max_iterations': 1000  # Prevent infinite loops
        }
        
        self.logger.info(f"Sampling parameters initialized:")
        self.logger.info(f"  - Buffer distance: {buffer_distance} feet")
        self.logger.info(f"  - Sampling interval: {sampling_interval} feet")
        self.logger.info(f"  - Strategy: {strategy}")
        self.logger.info(f"  - Preserve intersections: {preserve_intersections}")
        
        return params
    
    def _generate_candidate_points(self, segments_gdf: gpd.GeoDataFrame,
                                  sampling_params: Dict,
                                  max_points_per_segment: int) -> List[Dict]:
        """
        Generate candidate points along segments based on sampling strategy.
        
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
        strategy = sampling_params['strategy']
        
        if strategy == "uniform":
            return self._generate_uniform_candidates(segments_gdf, sampling_params, max_points_per_segment)
        elif strategy == "adaptive":
            return self._generate_adaptive_candidates(segments_gdf, sampling_params, max_points_per_segment)
        elif strategy == "intersection_aware":
            return self._generate_intersection_aware_candidates(segments_gdf, sampling_params, max_points_per_segment)
        else:
            self.logger.warning(f"Unknown strategy: {strategy}, falling back to uniform")
            return self._generate_uniform_candidates(segments_gdf, sampling_params, max_points_per_segment)
    
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
    
    def _generate_adaptive_candidates(self, segments_gdf: gpd.GeoDataFrame,
                                     sampling_params: Dict,
                                     max_points_per_segment: int) -> List[Dict]:
        """
        Generate adaptively spaced candidate points based on local density.
        
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
        base_interval = sampling_params['sampling_interval']
        
        # Calculate local density for each segment
        segment_densities = self._calculate_segment_densities(segments_gdf)
        
        for idx, (_, row) in enumerate(segments_gdf.iterrows()):
            if idx % 1000 == 0:
                self.logger.info(f"  Processing segment {idx + 1}/{len(segments_gdf)} ({idx/len(segments_gdf)*100:.1f}%)")
            
            try:
                # Adjust interval based on local density
                density_factor = segment_densities.get(idx, 1.0)
                adaptive_interval = base_interval * density_factor
                
                segment_candidates = self._generate_points_along_linestring(
                    row.geometry, adaptive_interval, idx, row, max_points_per_segment
                )
                candidates.extend(segment_candidates)
            except Exception as e:
                self.logger.error(f"Error processing segment {idx}: {e}")
                continue
        
        self.logger.info(f"Generated {len(candidates)} adaptive candidate points")
        return candidates
    
    def _generate_intersection_aware_candidates(self, segments_gdf: gpd.GeoDataFrame,
                                               sampling_params: Dict,
                                               max_points_per_segment: int) -> List[Dict]:
        """
        Generate candidate points with special handling for intersections.
        
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
        
        # Find intersection points
        intersection_points = self._find_intersection_points(segments_gdf)
        
        for idx, (_, row) in enumerate(segments_gdf.iterrows()):
            if idx % 1000 == 0:
                self.logger.info(f"  Processing segment {idx + 1}/{len(segments_gdf)} ({idx/len(segments_gdf)*100:.1f}%)")
            
            try:
                # Get intersections for this segment
                segment_intersections = [
                    pt for pt in intersection_points 
                    if row.geometry.distance(pt) < 1.0  # Within 1 foot tolerance
                ]
                
                segment_candidates = self._generate_points_with_intersection_awareness(
                    row.geometry, interval, idx, row, segment_intersections, max_points_per_segment
                )
                candidates.extend(segment_candidates)
            except Exception as e:
                self.logger.error(f"Error processing segment {idx}: {e}")
                continue
        
        self.logger.info(f"Generated {len(candidates)} intersection-aware candidate points")
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
            point_record = self._create_point_record(
                midpoint, segment_idx, segment_row, length / 2.0, length, 0.5
            )
            points.append(point_record)
        else:
            # Generate points at regular intervals
            num_points = min(int(length / interval) + 1, max_points)
            
            for i in range(num_points):
                distance = min(i * interval, length)
                point_geom = linestring.interpolate(distance)
                
                point_record = self._create_point_record(
                    point_geom, segment_idx, segment_row, distance, length, distance / length
                )
                points.append(point_record)
        
        return points
    
    def _generate_points_with_intersection_awareness(self, linestring: LineString,
                                                    interval: float,
                                                    segment_idx: int,
                                                    segment_row: pd.Series,
                                                    intersections: List[Point],
                                                    max_points: int) -> List[Dict]:
        """
        Generate points along a LineString with special handling for intersections.
        
        Parameters:
        -----------
        linestring : LineString
            Input LineString geometry
        interval : float
            Base interval between points
        segment_idx : int
            Index of the source segment
        segment_row : pd.Series
            Source segment data
        intersections : List[Point]
            Intersection points for this segment
        max_points : int
            Maximum number of points to generate
            
        Returns:
        --------
        List[Dict]
            List of point records
        """
        points = []
        length = linestring.length
        
        # Find intersection distances along the line
        intersection_distances = []
        for intersection in intersections:
            distance = linestring.project(intersection)
            intersection_distances.append(distance)
        
        intersection_distances.sort()
        
        # Generate points with intersection awareness
        current_distance = 0.0
        point_count = 0
        
        while current_distance <= length and point_count < max_points:
            # Check if we're too close to an intersection
            too_close_to_intersection = any(
                abs(current_distance - int_dist) < interval / 2.0
                for int_dist in intersection_distances
            )
            
            if not too_close_to_intersection:
                point_geom = linestring.interpolate(current_distance)
                point_record = self._create_point_record(
                    point_geom, segment_idx, segment_row, current_distance, length, 
                    current_distance / length
                )
                points.append(point_record)
                point_count += 1
            
            current_distance += interval
        
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
    
    def _calculate_segment_densities(self, segments_gdf: gpd.GeoDataFrame) -> Dict[int, float]:
        """
        Calculate local density factors for adaptive sampling.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input segments
            
        Returns:
        --------
        Dict[int, float]
            Mapping from segment index to density factor
        """
        self.logger.info("Calculating segment densities for adaptive sampling...")
        
        densities = {}
        
        # Use a spatial index for efficient neighborhood queries
        spatial_index = STRtree(list(segments_gdf.geometry))
        
        for idx, (_, row) in enumerate(segments_gdf.iterrows()):
            try:
                # Create buffer around segment
                buffer_geom = row.geometry.buffer(200.0)  # 200 feet buffer
                
                # Find intersecting segments
                intersecting_indices = spatial_index.query(buffer_geom)
                
                # Calculate density as number of intersecting segments per unit area
                buffer_area = buffer_geom.area
                density = len(intersecting_indices) / buffer_area if buffer_area > 0 else 1.0
                
                # Normalize density (higher density = smaller interval factor)
                density_factor = 1.0 / (1.0 + density * 0.001)  # Adjust scaling as needed
                densities[idx] = density_factor
                
            except Exception as e:
                densities[idx] = 1.0  # Default density
        
        self.logger.info(f"Calculated densities for {len(densities)} segments")
        return densities
    

    

    
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
            self.logger.info(f"  Processing parent_id {parent_id} with {len(parent_candidates)} candidates")
            
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
        self.logger.info("Processing intersection points with GPU acceleration...")
        
        # Find intersection points using GPU acceleration
        intersection_points = self._find_intersection_points_impl(segments_gdf)
        
        if not intersection_points:
            self.logger.info("No intersection points found")
            return filtered_points
        
        # Mark existing points that are near intersections using GPU acceleration
        intersection_buffer = sampling_params['buffer_distance'] * sampling_params['intersection_buffer_factor']
        
        self.logger.info("Using GPU-accelerated intersection processing...")
        filtered_points = self._mark_intersection_points_impl(
            filtered_points, intersection_points, intersection_buffer
        )
        
        # Count intersection points
        intersection_count = sum(1 for pt in filtered_points if pt.get('is_intersection', False))
        
        self.logger.info(f"Marked {intersection_count} points as intersection-related")
        return filtered_points
    
    def _find_intersection_points_impl(self, segments_gdf: gpd.GeoDataFrame) -> List[Point]:
        """
        Find intersection points between segments using GPU acceleration.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input segments
            
        Returns:
        --------
        List[Point]
            List of intersection points
        """
        self.logger.info("Using GPU-accelerated intersection detection...")
        
        self.logger.info(f"Processing {len(segments_gdf)} segments for GPU-accelerated intersection detection...")
        
        # Use GPU-accelerated coordinate operations for intersection detection
        geometries = segments_gdf.geometry.tolist()
        intersections = []
        
        # Use spatial index for efficient intersection queries
        from shapely.strtree import STRtree
        spatial_index = STRtree(geometries)
        
        # Process in batches to show progress
        batch_size = 1000
        total_comparisons = 0
        
        for i in range(0, len(geometries), batch_size):
            if i % 1000 == 0:
                self.logger.info(f"  Processing intersection batch {i//batch_size + 1}")
            
            batch_end = min(i + batch_size, len(geometries))
            
            for idx in range(i, batch_end):
                geom1 = geometries[idx]
                
                # Find potentially intersecting segments using spatial index
                candidates = spatial_index.query(geom1)
                
                for candidate_idx in candidates:
                    if candidate_idx > idx:  # Avoid duplicate comparisons
                        geom2 = geometries[candidate_idx]
                        
                        if geom1.intersects(geom2):
                            intersection = geom1.intersection(geom2)
                            
                            # Extract Point geometries
                            if intersection.geom_type == 'Point':
                                intersections.append(intersection)
                            elif intersection.geom_type == 'MultiPoint':
                                intersections.extend(list(intersection.geoms))
                            elif hasattr(intersection, 'geoms'):
                                for geom in intersection.geoms:
                                    if geom.geom_type == 'Point':
                                        intersections.append(geom)
                            
                            total_comparisons += 1
        
        self.logger.info(f"Found {len(intersections)} intersection points from {total_comparisons} intersecting pairs")
        
        # Remove duplicates using GPU-accelerated distance calculations
        if intersections:
            unique_intersections = self._remove_duplicate_points_vectorized(intersections)
            self.logger.info(f"After duplicate removal: {len(unique_intersections)} unique intersections")
            
            # Clean up GPU memory after large operation
            self._cleanup_gpu_memory()
            
            return unique_intersections
        else:
            self.logger.info("No intersection points found")
            return []
    
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
    
    def _mark_intersection_points_impl(self, filtered_points: List[Dict],
                                      intersection_points: List[Point],
                                      intersection_buffer: float) -> List[Dict]:
        """
        Mark points near intersections using optimized GPU acceleration.
        
        Parameters:
        -----------
        filtered_points : List[Dict]
            Filtered point records
        intersection_points : List[Point]
            List of intersection points
        intersection_buffer : float
            Buffer distance for intersection detection
            
        Returns:
        --------
        List[Dict]
            Points with intersection information
        """
        if not intersection_points:
            return filtered_points
        
        # Extract coordinates from filtered points
        point_coords = cp.array([[pt['geometry'].x, pt['geometry'].y] for pt in filtered_points])
        intersection_coords = cp.array([[ipt.x, ipt.y] for ipt in intersection_points])
        
        # Calculate distance matrix using optimized GPU operations
        buffer_squared = intersection_buffer * intersection_buffer
        
        # Use larger batch size for better GPU utilization
        batch_size = 50000  # Increased from 10000
        
        self.logger.info(f"Computing intersection distances for {len(filtered_points)} points using {len(intersection_points)} intersections...")
        
        for start_idx in range(0, len(point_coords), batch_size):
            end_idx = min(start_idx + batch_size, len(point_coords))
            batch_coords = point_coords[start_idx:end_idx]
            
            if start_idx % 100000 == 0:
                self.logger.info(f"  Processing batch {start_idx//batch_size + 1}/{(len(point_coords) + batch_size - 1)//batch_size}")
            
            # Calculate distances from batch to all intersections
            # Shape: (batch_size, num_intersections)
            diffs = batch_coords[:, cp.newaxis, :] - intersection_coords[cp.newaxis, :, :]
            distances_squared = cp.sum(diffs * diffs, axis=2)
            
            # Find minimum distance for each point
            min_distances_squared = cp.min(distances_squared, axis=1)
            
            # Mark points that are within buffer distance
            close_mask = min_distances_squared < buffer_squared
            
            # Convert results back to CPU
            close_mask_cpu = cp.asnumpy(close_mask)
            min_distances_cpu = cp.asnumpy(cp.sqrt(min_distances_squared))
            
            # Update the filtered points
            for i, point_idx in enumerate(range(start_idx, end_idx)):
                if close_mask_cpu[i]:
                    filtered_points[point_idx]['is_intersection'] = True
                    filtered_points[point_idx]['intersection_distance'] = float(min_distances_cpu[i])
        
        # Count marked points
        marked_count = sum(1 for pt in filtered_points if pt.get('is_intersection', False))
        self.logger.info(f"Marked {marked_count} points as intersection-related")
        
        return filtered_points
    

    

    
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
            
            final_data.append(record)
        
        # Create GeoDataFrame
        result = gpd.GeoDataFrame(final_data, crs=crs)
        
        # Add summary statistics as attributes
        result.attrs['sampling_summary'] = self._compute_sampling_summary(topology_points)
        
        self.logger.info(f"Created final network with {len(result)} points")
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
                'nearest_neighbor_id'
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
    

    
    def _compute_adjacency_graph_gpu(self, sampled_network: gpd.GeoDataFrame,
                                    regular_threshold: float,
                                    intersection_threshold: float,
                                    original_segments: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """
        GPU-accelerated adjacency computation using spatial indexing for efficiency.
        
        Parameters:
        -----------
        sampled_network : GeoDataFrame
            Sampled point network
        regular_threshold : float
            Maximum distance for regular adjacency connections (same parent_id)
        intersection_threshold : float
            Maximum distance for intersection-to-intersection connections
        original_segments : Optional[GeoDataFrame]
            Original segments for line-of-sight validation
            
        Returns:
        --------
        GeoDataFrame
            Network with adjacency information
        """
        self.logger.info("Using GPU-accelerated adjacency computation with spatial indexing...")
        self.logger.info(f"  - Regular threshold: {regular_threshold:.1f} feet")
        self.logger.info(f"  - Intersection threshold: {intersection_threshold:.1f} feet")
        
        # Store original segments for line-of-sight validation
        if original_segments is not None:
            self._original_segments = original_segments
            self.logger.info(f"Stored {len(original_segments)} original segments for line-of-sight validation")
            
            # Create spatial index for efficient line-of-sight checks
            try:
                self.logger.info("Creating spatial index for original segments...")
                self._segments_spatial_index = STRtree(list(original_segments.geometry))
                self.logger.info(f"Successfully created spatial index for {len(original_segments)} segments")
            except Exception as e:
                self.logger.warning(f"Failed to create spatial index: {e}")
                self.logger.info("Will use CPU fallback for line-of-sight checks")
        
        # Extract coordinates and metadata
        coords = np.array([[pt.x, pt.y] for pt in sampled_network.geometry])
        point_ids = sampled_network['point_id'].values
        parent_ids_raw = sampled_network['parent_id'].values if 'parent_id' in sampled_network.columns else np.full(len(sampled_network), 'unknown')
        is_intersection = sampled_network['is_intersection'].values if 'is_intersection' in sampled_network.columns else np.full(len(sampled_network), False)
        
        # Store original parent_ids for adjacency computation
        self._original_parent_ids = parent_ids_raw
        
        # Convert to cuSpatial format for efficient spatial operations
        self.logger.info("Converting to cuSpatial format for spatial indexing...")
        points_cuspatial = cuspatial.from_geopandas(sampled_network.geometry)
        
        # Use cuSpatial's spatial indexing to find nearby points efficiently
        self.logger.info("Building spatial index and finding nearby points...")
        
        # Find all point pairs within the maximum threshold distance
        max_threshold = max(regular_threshold, intersection_threshold)
        
        # Use cuSpatial's point-in-polygon or distance-based operations
        # For now, we'll use a simpler approach with cuSpatial's spatial operations
        # This is much more efficient than the O(n²) approach
        
        # Create a spatial index using cuSpatial
        try:
            # Use cuSpatial's spatial operations to find nearby points
            # This is much more efficient than checking all pairs
            self.logger.info("Using cuSpatial spatial operations for efficient neighbor finding...")
            
            # For each point, find points within the maximum threshold
            all_point_pairs = []
            all_adjacency_data = {}
            
            # Initialize adjacency data
            for point_id in point_ids:
                all_adjacency_data[point_id] = {
                    'adjacent_points': [],
                    'adjacency_count': 0
                }
            
            # Process in batches for memory efficiency
            batch_size = 500  # Larger batches since we're using spatial indexing
            total_pairs_found = 0
            
            for start_idx in range(0, len(coords), batch_size):
                end_idx = min(start_idx + batch_size, len(coords))
                batch_coords = coords[start_idx:end_idx]
                
                batch_num = start_idx // batch_size + 1
                total_batches = (len(coords) + batch_size - 1) // batch_size
                
                if start_idx == 0 or batch_num % 5 == 0:
                    self.logger.info(f"  Processing spatial batch {batch_num}/{total_batches} (points {start_idx}-{end_idx})")
                
                # For each point in the batch, find nearby points using spatial operations
                for i, point_coord in enumerate(batch_coords):
                    point_idx = start_idx + i
                    point_id = point_ids[point_idx]
                    point_is_intersection = is_intersection[point_idx]
                    point_parent_id = parent_ids_raw[point_idx]
                    
                    # Create a buffer around this point to find nearby points
                    # This is much more efficient than checking all points
                    buffer_distance = max_threshold
                    
                    # Find points within buffer distance using spatial operations
                    # We'll use a simple distance-based approach for now
                    nearby_indices = []
                    nearby_distances = []
                    
                    # Check only points that could be within range
                    # This reduces the search space significantly
                    for j, other_coord in enumerate(coords):
                        if i == j:  # Skip self
                            continue
                            
                        # Quick distance check
                        dx = point_coord[0] - other_coord[0]
                        dy = point_coord[1] - other_coord[1]
                        distance_squared = dx*dx + dy*dy
                        
                        # Only process if within maximum threshold
                        if distance_squared <= max_threshold * max_threshold:
                            nearby_indices.append(j)
                            nearby_distances.append(distance_squared ** 0.5)
                    
                    # Apply adjacency rules to nearby points
                    valid_neighbors = []
                    for k, neighbor_idx in enumerate(nearby_indices):
                        neighbor_id = point_ids[neighbor_idx]
                        neighbor_is_intersection = is_intersection[neighbor_idx]
                        neighbor_parent_id = parent_ids_raw[neighbor_idx]
                        distance = nearby_distances[k]
                        
                        # Determine which threshold to use
                        if point_is_intersection and neighbor_is_intersection:
                            threshold = intersection_threshold
                        else:
                            threshold = regular_threshold
                            
                            # For regular points, check parent_id
                            if not point_is_intersection and point_parent_id != neighbor_parent_id:
                                continue  # Skip if different parent_id
                        
                        # Check if within threshold
                        if distance <= threshold:
                            # Determine relationship type
                            if point_is_intersection and neighbor_is_intersection:
                                relationship = 'intersection_intersection'
                            elif point_parent_id == neighbor_parent_id:
                                relationship = 'same_segment'
                            else:
                                relationship = 'cross_segment'
                            
                            valid_neighbors.append({
                                'point_id': neighbor_id,
                                'distance': distance,
                                'relationship': relationship
                            })
                    
                    # Store valid neighbors
                    if valid_neighbors:
                        # Sort by distance
                        valid_neighbors.sort(key=lambda x: x['distance'])
                        
                        # Add to adjacency data
                        all_adjacency_data[point_id]['adjacent_points'] = valid_neighbors
                        all_adjacency_data[point_id]['adjacency_count'] = len(valid_neighbors)
                        
                        # Add pairs for line-of-sight checking
                        for neighbor in valid_neighbors:
                            pair = (point_id, neighbor['point_id'])
                            all_point_pairs.append(pair)
                            total_pairs_found += 1
            
            self.logger.info(f"Found {total_pairs_found} potential adjacency pairs using spatial indexing")
            
        except Exception as e:
            self.logger.error(f"cuSpatial spatial operations failed: {e}")
            raise RuntimeError(f"Failed to compute adjacency with spatial indexing: {e}")
        
        # Process line-of-sight validation for the found pairs
        if all_point_pairs:
            self.logger.info(f"Processing {len(all_point_pairs)} point pairs for line-of-sight validation...")
            
            # Process in batches
            batch_size = 1000
            all_line_of_sight_results = []
            
            for batch_start in range(0, len(all_point_pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(all_point_pairs))
                batch_pairs = all_point_pairs[batch_start:batch_end]
                
                batch_num = batch_start // batch_size + 1
                total_batches = (len(all_point_pairs) + batch_size - 1) // batch_size
                
                if batch_num == 1 or batch_num % 10 == 0 or batch_num == total_batches:
                    self.logger.info(f"  Processing line-of-sight batch {batch_num}/{total_batches} (pairs {batch_start}-{batch_end})")
                
                batch_results = self._batch_check_line_of_sight_gpu(batch_pairs, sampled_network)
                all_line_of_sight_results.extend(batch_results)
            
                    # Apply line-of-sight results
        self.logger.info("Applying line-of-sight results...")
        pair_index = 0
        for point_id, data in all_adjacency_data.items():
            if data['adjacent_points']:
                filtered_neighbors = []
                for neighbor in data['adjacent_points']:
                    if all_line_of_sight_results[pair_index]:
                        filtered_neighbors.append(neighbor)
                    pair_index += 1
                
                all_adjacency_data[point_id]['adjacent_points'] = filtered_neighbors
                all_adjacency_data[point_id]['adjacency_count'] = len(filtered_neighbors)
        
        self.logger.info("Line-of-sight filtering completed - invalid connections removed")
        
        # Apply directional filtering to remove redundant adjacency relationships
        self.logger.info("Applying directional filtering to remove redundant adjacency relationships...")
        angle_tolerance = 10  # degrees
        angle_tolerance_rad = np.radians(angle_tolerance)
        
        # Get point coordinates for vector calculations
        point_coords_dict = {pid: (pt.x, pt.y) for pid, pt in zip(sampled_network['point_id'], sampled_network.geometry)}
        
        filtered_count = 0
        for point_id, data in all_adjacency_data.items():
            if len(data['adjacent_points']) <= 1:
                continue  # No filtering needed for 0 or 1 neighbor
            
            # Get source point coordinates
            if point_id not in point_coords_dict:
                continue
            source_coords = point_coords_dict[point_id]
            
            # Calculate vectors to all neighbors
            neighbor_vectors = []
            for neighbor in data['adjacent_points']:
                neighbor_id = neighbor['point_id']
                if neighbor_id not in point_coords_dict:
                    continue
                    
                neighbor_coords = point_coords_dict[neighbor_id]
                
                # Calculate vector from source to neighbor
                dx = neighbor_coords[0] - source_coords[0]
                dy = neighbor_coords[1] - source_coords[1]
                magnitude = np.sqrt(dx*dx + dy*dy)
                
                if magnitude > 0:
                    # Normalize vector and calculate angle
                    unit_x = dx / magnitude
                    unit_y = dy / magnitude
                    angle = np.arctan2(unit_y, unit_x)
                    
                    neighbor_vectors.append({
                        'neighbor': neighbor,
                        'magnitude': magnitude,
                        'angle': angle,
                        'unit_x': unit_x,
                        'unit_y': unit_y
                    })
            
            if len(neighbor_vectors) <= 1:
                continue
            
            # Sort by magnitude (shortest first)
            neighbor_vectors.sort(key=lambda x: x['magnitude'])
            
            # Filter out redundant directions
            filtered_vectors = []
            for i, vec1 in enumerate(neighbor_vectors):
                is_redundant = False
                
                # Check if this vector is redundant with any shorter vector
                for j in range(i):
                    vec2 = neighbor_vectors[j]
                    
                    # Calculate angle difference
                    angle_diff = abs(vec1['angle'] - vec2['angle'])
                    # Handle angle wrapping (e.g., -179° vs 179° should be 2° difference, not 358°)
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    # If angles are very similar, keep the shorter one
                    if angle_diff <= angle_tolerance_rad:
                        is_redundant = True
                        break
                
                if not is_redundant:
                    filtered_vectors.append(vec1)
                else:
                    filtered_count += 1
            
            # Update adjacency data with filtered neighbors
            all_adjacency_data[point_id]['adjacent_points'] = [vec['neighbor'] for vec in filtered_vectors]
            all_adjacency_data[point_id]['adjacency_count'] = len(filtered_vectors)
        
        self.logger.info(f"Directional filtering completed - removed {filtered_count} redundant adjacency relationships")
        self.logger.info(f"  - Angle tolerance: {angle_tolerance} degrees")
        self.logger.info(f"  - Kept shortest connection within {angle_tolerance}° angular tolerance")
        
        # Create final result
        result = sampled_network.copy()
        result['adjacent_points'] = [all_adjacency_data[pid]['adjacent_points'] for pid in result['point_id']]
        result['adjacency_count'] = [all_adjacency_data[pid]['adjacency_count'] for pid in result['point_id']]
        
        # Calculate statistics
        intersection_points = result[result['is_intersection'] == True]
        regular_points = result[result['is_intersection'] == False]
        
        intersection_avg = intersection_points['adjacency_count'].mean() if len(intersection_points) > 0 else 0
        regular_avg = regular_points['adjacency_count'].mean() if len(regular_points) > 0 else 0
        
        self.logger.info(f"Spatial-indexed adjacency computation complete for {len(result)} points")
        self.logger.info(f"  - Total points: {len(result)}")
        self.logger.info(f"  - Intersection points: {len(intersection_points)} (avg degree: {intersection_avg:.2f})")
        self.logger.info(f"  - Regular points: {len(regular_points)} (avg degree: {regular_avg:.2f})")
        self.logger.info(f"  - Overall average degree: {result['adjacency_count'].mean():.2f}")
        
        return result
    

    


    def _batch_check_line_of_sight_gpu(self, point_pairs: List[Tuple[int, int]], 
                                      sampled_network: gpd.GeoDataFrame) -> List[bool]:
        """
        Batch line-of-sight check: calculate the percentage of line length that is contained within nearby segments.
        Return True only if a significant percentage (80%) is contained.
        """
        if not point_pairs:
            return []
        self.logger.info(f"Batch checking line-of-sight for {len(point_pairs)} point pairs using percentage containment logic...")
        
        # Debug CRS information
        self.logger.info(f"CRS Debug:")
        self.logger.info(f"  - Sampled network CRS: {sampled_network.crs}")
        self.logger.info(f"  - Original segments CRS: {self._original_segments.crs}")
        if sampled_network.crs != self._original_segments.crs:
            self.logger.warning(f"CRS MISMATCH: Sampled network ({sampled_network.crs}) != Original segments ({self._original_segments.crs})")
        
        try:
            point_ids = set()
            for pair in point_pairs:
                point_ids.add(pair[0])
                point_ids.add(pair[1])
            points_df = sampled_network[sampled_network['point_id'].isin(point_ids)]
            point_geoms = {row['point_id']: row.geometry for _, row in points_df.iterrows()}
            point_is_intersection = {row['point_id']: row.get('is_intersection', False) for _, row in points_df.iterrows()}
            
            # Debug intersection flags
            intersection_count = sum(point_is_intersection.values())
            total_points = len(point_is_intersection)
            self.logger.info(f"DEBUG: Intersection flag analysis for {total_points} points in batch:")
            self.logger.info(f"  - Points marked as intersections: {intersection_count}")
            self.logger.info(f"  - Points marked as regular: {total_points - intersection_count}")
            self.logger.info(f"  - Intersection percentage: {intersection_count/total_points*100:.1f}%")
            
            # Show sample intersection flags
            sample_intersection_points = [pid for pid, is_int in point_is_intersection.items() if is_int][:5]
            sample_regular_points = [pid for pid, is_int in point_is_intersection.items() if not is_int][:5]
            self.logger.info(f"  - Sample intersection point IDs: {sample_intersection_points}")
            self.logger.info(f"  - Sample regular point IDs: {sample_regular_points}")
            if not hasattr(self, '_segments_spatial_index'):
                self.logger.info("Creating spatial index for original segments...")
                self._segments_spatial_index = STRtree(list(self._original_segments.geometry))
                self.logger.info(f"Created spatial index for {len(self._original_segments)} segments")
            batch_size = 1000
            results = []
            containment_percentages = []
            line_lengths = []
            intersection_counts = []
            debug_samples = []
            
            # Track pair types for debugging
            pair_types = {'intersection-intersection': 0, 'mixed': 0, 'regular-regular': 0}
            pair_results = {'intersection-intersection': {'valid': 0, 'invalid': 0}, 
                           'mixed': {'valid': 0, 'invalid': 0}, 
                           'regular-regular': {'valid': 0, 'invalid': 0}}
            
            for batch_start in range(0, len(point_pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(point_pairs))
                batch_pairs = point_pairs[batch_start:batch_end]
                batch_results = []
                for i, (point_id1, point_id2) in enumerate(batch_pairs):
                    if point_id1 not in point_geoms or point_id2 not in point_geoms:
                        batch_results.append(False)
                        continue
                    
                    # Determine pair type
                    is_intersection1 = point_is_intersection.get(point_id1, False)
                    is_intersection2 = point_is_intersection.get(point_id2, False)
                    
                    # Debug first few pair classifications
                    if i < 10:  # Only debug first 10 pairs
                        self.logger.info(f"DEBUG: Pair {i}: {point_id1}-{point_id2}")
                        self.logger.info(f"  - Point {point_id1} intersection: {is_intersection1}")
                        self.logger.info(f"  - Point {point_id2} intersection: {is_intersection2}")
                    
                    if is_intersection1 and is_intersection2:
                        pair_type = 'intersection-intersection'
                        pair_types[pair_type] += 1
                        batch_results.append(True)  # Bypass line-of-sight check
                        pair_results[pair_type]['valid'] += 1
                        if i < 10:
                            self.logger.info(f"  - Classified as: {pair_type} (bypassing LOS)")
                        continue
                    elif is_intersection1 or is_intersection2:
                        pair_type = 'mixed'
                        pair_types[pair_type] += 1
                        if i < 10:
                            self.logger.info(f"  - Classified as: {pair_type}")
                    else:
                        pair_type = 'regular-regular'
                        pair_types[pair_type] += 1
                        if i < 10:
                            self.logger.info(f"  - Classified as: {pair_type}")
                    
                    line = LineString([point_geoms[point_id1], point_geoms[point_id2]])
                    line_length = line.length
                    line_lengths.append(line_length)
                    if line_length == 0:
                        batch_results.append(True)
                        pair_results[pair_type]['valid'] += 1
                        continue
                    nearby_segment_indices = self._segments_spatial_index.query(line)
                    if len(nearby_segment_indices) == 0:
                        batch_results.append(False)
                        containment_percentages.append(0.0)
                        intersection_counts.append(0)
                        pair_results[pair_type]['invalid'] += 1
                        continue
                    nearby_segments = self._original_segments.iloc[nearby_segment_indices]
                    
                    # Calculate total length of line that is contained within buffered sidewalk segments
                    total_contained_length = 0.0
                    intersecting_segments = 0
                    
                    for _, segment in nearby_segments.iterrows():
                        geom = segment.geometry
                        
                        # Buffer the LineString segment by its width to create a polygon
                        width = segment.get('width', 5.0)  # Default to 5 feet if no width column
                        buffered_segment = geom.buffer(width)
                        
                        if line.intersects(buffered_segment):
                            intersecting_segments += 1
                            intersection = line.intersection(buffered_segment)
                            
                            # Debug intersection details for first few cases
                            if len(debug_samples) < 3 and intersecting_segments <= 2:
                                self.logger.info(f"DEBUG: Line {point_id1}-{point_id2} ({pair_type}) intersects buffered segment")
                                self.logger.info(f"  - Point 1 intersection: {is_intersection1}")
                                self.logger.info(f"  - Point 2 intersection: {is_intersection2}")
                                self.logger.info(f"  - Original segment type: {geom.geom_type}")
                                self.logger.info(f"  - Segment width: {width}")
                                self.logger.info(f"  - Buffered segment type: {buffered_segment.geom_type}")
                                self.logger.info(f"  - Intersection type: {intersection.geom_type}")
                                self.logger.info(f"  - Line length: {line_length:.2f}")
                                self.logger.info(f"  - Intersection length: {intersection.length if hasattr(intersection, 'length') else 'N/A'}")
                            
                            if intersection.geom_type == 'LineString':
                                contained_length = intersection.length
                            elif intersection.geom_type == 'MultiLineString':
                                contained_length = sum(geom.length for geom in intersection.geoms)
                            elif intersection.geom_type == 'Point':
                                contained_length = 0.0
                            elif intersection.geom_type == 'MultiPoint':
                                contained_length = 0.0
                            else:
                                contained_length = 0.0
                            
                            total_contained_length += contained_length
                    

                    
                    intersection_counts.append(intersecting_segments)
                    
                    # Calculate percentage of line that is close to segments
                    containment_percentage = (total_contained_length / line_length) * 100 if line_length > 0 else 0.0
                    containment_percentages.append(containment_percentage)
                    
                    # Store debug info for first few invalid pairs
                    if containment_percentage < 80.0 and len(debug_samples) < 5:
                        debug_samples.append({
                            'point_id1': point_id1,
                            'point_id2': point_id2,
                            'pair_type': pair_type,
                            'line_length': line_length,
                            'containment_percentage': containment_percentage,
                            'total_contained_length': total_contained_length,
                            'intersecting_segments': intersecting_segments,
                            'nearby_segments': len(nearby_segments)
                        })
                    
                    # Return True only if 80% or more of the line is contained
                    is_valid = containment_percentage >= 80.0
                    batch_results.append(is_valid)
                    
                    if is_valid:
                        pair_results[pair_type]['valid'] += 1
                    else:
                        pair_results[pair_type]['invalid'] += 1
                
                results.extend(batch_results)
                if batch_start % 5000 == 0:
                    self.logger.info(f"  Processed {batch_start + len(batch_pairs)}/{len(point_pairs)} pairs")
            valid_count = sum(results)
            total_count = len(results)
            
            # Debug statistics
            if containment_percentages:
                avg_containment = sum(containment_percentages) / len(containment_percentages)
                min_containment = min(containment_percentages)
                max_containment = max(containment_percentages)
                self.logger.info(f"Containment statistics:")
                self.logger.info(f"  - Average containment: {avg_containment:.2f}%")
                self.logger.info(f"  - Min containment: {min_containment:.2f}%")
                self.logger.info(f"  - Max containment: {max_containment:.2f}%")
                self.logger.info(f"  - Pairs with 0% containment: {sum(1 for p in containment_percentages if p == 0)}")
                self.logger.info(f"  - Pairs with <10% containment: {sum(1 for p in containment_percentages if p < 10)}")
                self.logger.info(f"  - Pairs with <50% containment: {sum(1 for p in containment_percentages if p < 50)}")
                self.logger.info(f"  - Pairs with <80% containment: {sum(1 for p in containment_percentages if p < 80)}")
            
            if line_lengths:
                avg_line_length = sum(line_lengths) / len(line_lengths)
                self.logger.info(f"Line length statistics:")
                self.logger.info(f"  - Average line length: {avg_line_length:.2f} feet")
                self.logger.info(f"  - Min line length: {min(line_lengths):.2f} feet")
                self.logger.info(f"  - Max line length: {max(line_lengths):.2f} feet")
            
            if intersection_counts:
                avg_intersections = sum(intersection_counts) / len(intersection_counts)
                self.logger.info(f"Intersection statistics:")
                self.logger.info(f"  - Average intersecting segments per line: {avg_intersections:.2f}")
                self.logger.info(f"  - Lines with 0 intersections: {sum(1 for c in intersection_counts if c == 0)}")
                self.logger.info(f"  - Lines with 1+ intersections: {sum(1 for c in intersection_counts if c > 0)}")
            
            # Show debug samples
            if debug_samples:
                self.logger.info(f"Sample invalid pairs (containment < 80%):")
                for i, sample in enumerate(debug_samples):
                    self.logger.info(f"  Sample {i+1}: Points {sample['point_id1']}-{sample['point_id2']}")
                    self.logger.info(f"    - Line length: {sample['line_length']:.2f} feet")
                    self.logger.info(f"    - Containment: {sample['containment_percentage']:.2f}%")
                    self.logger.info(f"    - Contained length: {sample['total_contained_length']:.2f} feet")
                    self.logger.info(f"    - Intersecting segments: {sample['intersecting_segments']}")
                    self.logger.info(f"    - Nearby segments: {sample['nearby_segments']}")
            
            self.logger.info(f"Batch line-of-sight check complete: {valid_count}/{total_count} pairs valid ({valid_count/total_count*100:.1f}%)")
            
            # Log pair type statistics
            self.logger.info(f"Pair type statistics:")
            for pair_type, count in pair_types.items():
                if count > 0:
                    valid = pair_results[pair_type]['valid']
                    invalid = pair_results[pair_type]['invalid']
                    self.logger.info(f"  - {pair_type}: {count} total, {valid} valid, {invalid} invalid ({valid/count*100:.1f}% valid)")
            
            if valid_count < total_count:
                invalid_count = total_count - valid_count
                self.logger.info(f"Filtered out {invalid_count} invalid connections that aren't sufficiently contained within sidewalk segments")
            return results
        except Exception as e:
            self.logger.warning(f"Batch line-of-sight check failed: {e}")
            return [False] * len(point_pairs)
    

    
    def process_segments_to_sampled_network(self, segments_input: Union[gpd.GeoDataFrame, str],
                                          buffer_distance: float = 100.0,
                                          sampling_interval: float = 50.0,
                                          strategy: str = "uniform",
                                          output_path: Optional[str] = None,
                                          save_intermediate: bool = False,
                                          intermediate_dir: Optional[str] = None,
                                          pedestrian_ramps_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Master function to process segments into a complete sampled point network.
        
        Parameters:
        -----------
        segments_input : Union[GeoDataFrame, str]
            Input segments as GeoDataFrame or file path (must include 'parent_id' column)
        buffer_distance : float
            Minimum distance between points
        sampling_interval : float
            Initial spacing between candidate points
        strategy : str
            Sampling strategy
        output_path : Optional[str]
            Path to save the final network
        save_intermediate : bool
            Whether to save intermediate results
        intermediate_dir : Optional[str]
            Directory for intermediate files
        pedestrian_ramps_path : Optional[str]
            Path to pedestrian ramps GeoJSON file
            
        Returns:
        --------
        GeoDataFrame
            Complete sampled point network
        """
        start_time = time.time()
        
        # Step 1: Load and validate input
        self.logger.info("Step 1: Loading and validating input segments...")
        segments_gdf = self._load_segments(segments_input)
        
        if save_intermediate and intermediate_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            intermediate_path = os.path.join(intermediate_dir, f"01_input_segments_{timestamp}.parquet")
            segments_gdf.to_parquet(intermediate_path)
            self.logger.info(f"Saved input segments to: {intermediate_path}")
        
        # Step 2: Generate sampled network
        self.logger.info("Step 2: Generating sampled point network...")
        sampled_network = self.generate_sampled_network(
            segments_gdf=segments_gdf,
            buffer_distance=buffer_distance,
            sampling_interval=sampling_interval,
            strategy=strategy,
            pedestrian_ramps_path=pedestrian_ramps_path
        )
        
        if save_intermediate and intermediate_dir:
            intermediate_path = os.path.join(intermediate_dir, f"02_sampled_network_{timestamp}.parquet")
            sampled_network.to_parquet(intermediate_path)
            self.logger.info(f"Saved sampled network to: {intermediate_path}")
        
        # Step 3: Compute adjacency
        self.logger.info("Step 3: Computing adjacency relationships...")
        # Use buffer_distance * 1.1 for regular connections, buffer_distance * 2.0 for intersection connections
        regular_threshold = buffer_distance * 2.5
        intersection_threshold = buffer_distance * 2.5
        final_network = self._compute_adjacency_graph_gpu(sampled_network, regular_threshold, intersection_threshold, segments_gdf)
        
        # Step 4: Save final result
        if output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
            extension = output_path.rsplit('.', 1)[1] if '.' in output_path else 'parquet'
            dynamic_output_path = f"{base_path}_sampled_{timestamp}.{extension}"
            
            # Prepare for saving (remove complex objects)
            save_network = final_network.copy()
            save_network['adjacent_points'] = save_network['adjacent_points'].apply(
                lambda x: ','.join([f"{pt['point_id']}" for pt in x])  # Limit to 5 nearest
            )
            save_network.to_parquet(dynamic_output_path)
            self.logger.info(f"Saved final network to: {dynamic_output_path}")
        
        # Final statistics
        total_time = time.time() - start_time
        self.logger.info(f"\nTotal processing time: {total_time:.2f} seconds")
        self.logger.info(f"Final network: {len(final_network)} points")
        
        return final_network
    
    def _load_segments(self, segments_input: Union[gpd.GeoDataFrame, str]) -> gpd.GeoDataFrame:
        """
        Load segment network from file path or return existing GeoDataFrame.
        
        Parameters:
        -----------
        segments_input : Union[GeoDataFrame, str]
            Input as GeoDataFrame or file path string
            
        Returns:
        --------
        GeoDataFrame
            Loaded segment network
        """
        if isinstance(segments_input, gpd.GeoDataFrame):
            self.logger.info(f"Using provided GeoDataFrame with {len(segments_input)} segments")
            return segments_input
        
        if isinstance(segments_input, str):
            self.logger.info(f"Loading segment network from file: {segments_input}")
            
            try:
                if segments_input.lower().endswith('.parquet'):
                    segments_gdf = gpd.read_parquet(segments_input)
                elif segments_input.lower().endswith('.gpkg'):
                    segments_gdf = gpd.read_file(segments_input)
                elif segments_input.lower().endswith(('.shp', '.geojson', '.json')):
                    segments_gdf = gpd.read_file(segments_input)
                else:
                    raise ValueError(f"Unsupported file format: {segments_input}")
                
                self.logger.info(f"Successfully loaded {len(segments_gdf)} segments")
                return segments_gdf
                
            except Exception as e:
                self.logger.error(f"ERROR: Failed to load file {segments_input}: {e}")
                raise
        
        raise ValueError(f"Invalid input type {type(segments_input)}")
    



if __name__ == "__main__":
    # Use Google Fire for CLI
    fire.Fire(SampledPointNetwork) 