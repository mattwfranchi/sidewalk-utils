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
        sampling_params = self._initialize_sampling_parameters(
            buffer_distance, sampling_interval, strategy, preserve_intersections
        )
        
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
        
        # Step 8: Compute adjacency relationships (optional)
        if compute_adjacency:
            self.logger.info("Step 8: Computing adjacency relationships...")
            # Use filtered centerlines for adjacency computation if provided, otherwise use segments
            adjacency_segments = adjacency_centerlines if adjacency_centerlines is not None else segments_gdf
            self.logger.info(f"Using {'filtered centerlines' if adjacency_centerlines is not None else 'neighborhood segments'} for adjacency computation")
            
            adjacency_enhanced = self.compute_adjacency_relationships(
                topology_enhanced, adjacency_segments, 
                walkshed_distance=walkshed_distance,
                crosswalk_distance=crosswalk_distance
            )
        else:
            adjacency_enhanced = topology_enhanced
        
        # Step 9: Create final GeoDataFrame
        self.logger.info("Step 9: Creating final sampled point network...")
        result = self._create_final_network(adjacency_enhanced, segments_gdf.crs)
        
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
            Sampling strategy (only uniform is supported)
        preserve_intersections : bool
            Whether to preserve intersection points
            
        Returns:
        --------
        Dict
            Sampling parameters dictionary
        """
        # Only uniform strategy is supported
        if strategy != "uniform":
            self.logger.warning(f"WARNING: Strategy '{strategy}' not supported. Using 'uniform'")
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
        Compute adjacency relationships using streamlined graph traversal.
        
        This method implements a simplified adjacency algorithm:
        1. Add crosswalk segments between nearby intersection points
        2. For each point, find its containing segment
        3. Traverse forward/backward along the segment to find adjacent points
        4. For intersection points, also traverse along connected segments
        
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
        self.logger.info("COMPUTING ADJACENCY RELATIONSHIPS (STREAMLINED)")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Initialize batch parameters
        if batch_params is None:
            batch_params = {
                'super_batch_size': 100,    # Process 100 streets at once (increased for simpler logic)
                'sub_batch_size': 2000,     # Process 2k points at once (increased for simpler logic)
                'micro_batch_size': 200,    # Network analysis batch size
                'max_adjacent_points': 50   # Maximum adjacent points per point
            }
        
        self.logger.info(f"Batch parameters: {batch_params}")
        self.logger.info(f"Crosswalk distance: {crosswalk_distance} feet")
        self.logger.info("Note: Adjacent distances are measured along segment paths (not straight-line)")
        
        # Step 1: Create enhanced network with crosswalks
        self.logger.info("Step 1: Creating enhanced network with crosswalks...")
        enhanced_segments = self._create_enhanced_network_with_crosswalks(
            points, segments_gdf, crosswalk_distance, batch_params
        )
        
        # NEW: Save enhanced network if save_intermediate is enabled
        if hasattr(self, '_save_intermediate') and self._save_intermediate and hasattr(self, '_intermediate_dir'):
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            enhanced_path = os.path.join(self._intermediate_dir, f"03_enhanced_network_with_crosswalks_{timestamp}.parquet")
            enhanced_segments.to_parquet(enhanced_path)
            self.logger.info(f"Saved enhanced network with crosswalks to: {enhanced_path}")
        
        # Step 2: Convert to GPU format for efficient processing
        self.logger.info("Step 2: Converting data to GPU format...")
        points_df, segments_df = self._convert_to_gpu_format(points, enhanced_segments)
        
        # Step 3: Compute adjacency relationships using graph traversal
        self.logger.info("Step 3: Computing adjacency relationships via graph traversal...")
        adjacency_start_time = time.time()
        
        adjacency_results = self._compute_adjacency_via_graph_traversal(
            points_df, segments_df, batch_params
        )
        
        adjacency_time = time.time() - adjacency_start_time
        self.logger.info(f"Adjacency computation completed in {adjacency_time:.2f} seconds")
        
        # Step 4: Update points with adjacency information
        self.logger.info("Step 4: Updating points with adjacency information...")
        updated_points = self._update_points_with_adjacency(points, adjacency_results)
        
        # Log final statistics
        total_time = time.time() - start_time
        self._log_adjacency_statistics(updated_points, total_time)
        
        return updated_points

    def _create_enhanced_network_with_crosswalks(self, points: List[Dict],
                                                segments_gdf: gpd.GeoDataFrame,
                                                crosswalk_distance: float,
                                                batch_params: Dict) -> gpd.GeoDataFrame:
        """
        Create enhanced network by adding crosswalk segments between nearby intersection points.
        
        Parameters:
        -----------
        points : List[Dict]
            Point records
        segments_gdf : GeoDataFrame
            Original sidewalk segments
        crosswalk_distance : float
            Maximum distance for crosswalk creation
        batch_params : Dict
            Batching parameters
            
        Returns:
        --------
        GeoDataFrame
            Enhanced segments including crosswalks
        """
        self.logger.info(f"Creating enhanced network with crosswalks...")
        
        # Identify intersection points
        intersection_points = []
        for point in points:
            if point.get('is_intersection', False) or point.get('is_pedestrian_ramp', False):
                intersection_points.append({
                    'geometry': point['geometry'],
                    'point_id': point['point_id'],
                    'parent_id': point.get('parent_id'),
                    'x': point['geometry'].x,
                    'y': point['geometry'].y
                })
        
        self.logger.info(f"Found {len(intersection_points)} intersection points")
        
        if len(intersection_points) < 2:
            self.logger.info("Not enough intersection points for crosswalks")
            return segments_gdf
        
        # Convert to CuPy arrays for efficient distance calculations
        intersection_coords = cp.array([[pt['x'], pt['y']] for pt in intersection_points])
        
        # Compute pairwise distances between intersection points
        self.logger.info("Computing pairwise distances between intersection points...")
        
        # Process in micro-batches to avoid memory issues
        crosswalks = []
        micro_batch_size = batch_params['micro_batch_size']
        
        for i in range(0, len(intersection_coords), micro_batch_size):
            end_i = min(i + micro_batch_size, len(intersection_coords))
            batch_coords = intersection_coords[i:end_i]
            
            # Calculate distances from batch to all intersection points
            diffs = batch_coords[:, cp.newaxis, :] - intersection_coords[cp.newaxis, :, :]
            distances = cp.sqrt(cp.sum(diffs * diffs, axis=2))
            
            # Find pairs within crosswalk distance (excluding self)
            for j in range(len(batch_coords)):
                point_idx = i + j
                if point_idx >= len(intersection_points):
                    break
                
                # Get distances for this point
                point_distances = distances[j]
                
                # Find nearby intersection points
                nearby_indices = cp.where((point_distances > 0) & (point_distances <= crosswalk_distance))[0]
                
                for nearby_idx in cp.asnumpy(nearby_indices):
                    if nearby_idx > point_idx:  # Avoid duplicate pairs
                        # Create crosswalk segment
                        point1 = intersection_points[point_idx]
                        point2 = intersection_points[nearby_idx]
                        
                        crosswalk_geom = LineString([point1['geometry'], point2['geometry']])
                        
                        crosswalks.append({
                            'geometry': crosswalk_geom,
                            'parent_id': f"crosswalk_{len(crosswalks):06d}",
                            'segment_type': 'crosswalk',
                            'is_crosswalk': True,
                            'length': crosswalk_geom.length,
                            'source_point_id': point1['point_id'],
                            'target_point_id': point2['point_id'],
                            'distance': float(point_distances[nearby_idx])
                        })
        
        self.logger.info(f"Created {len(crosswalks)} crosswalk segments")
        
        # Create GeoDataFrame for crosswalks
        if crosswalks:
            crosswalks_gdf = gpd.GeoDataFrame(crosswalks, crs=segments_gdf.crs)
            
            # Combine with original segments
            enhanced_segments = gpd.GeoDataFrame(
                pd.concat([segments_gdf, crosswalks_gdf], ignore_index=True),
                crs=segments_gdf.crs
            )
        else:
            enhanced_segments = segments_gdf
        
        self.logger.info(f"Enhanced network: {len(enhanced_segments)} total segments")
        return enhanced_segments

    def _convert_to_gpu_format(self, points: List[Dict], 
                              segments_gdf: gpd.GeoDataFrame) -> Tuple[cudf.DataFrame, cudf.DataFrame]:
        """
        Convert points and segments to GPU format for efficient processing.
        
        Parameters:
        -----------
        points : List[Dict]
            Point records
        segments_gdf : GeoDataFrame
            Enhanced segments including crosswalks
            
        Returns:
        --------
        Tuple[cudf.DataFrame, cudf.DataFrame]
            Points and segments in GPU format
        """
        # Convert points to CuDF
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

        # unzip x and y from points_data 
        x, y = zip(*[(point['x'], point['y']) for point in points_data])
        points_gdf = gpd.GeoDataFrame(points_data, geometry=gpd.points_from_xy(x, y))

        points_gdf = cuspatial.from_geopandas(points_gdf)
        
        # Convert segments to CuDF with explicit type handling
        segments_data = []
        for idx, (_, row) in enumerate(segments_gdf.iterrows()):
            try:
                # Extract start and end points of the segment
                coords = list(row.geometry.coords)
                if len(coords) >= 2:
                    start_point = coords[0]
                    end_point = coords[-1]
                    
                    # Ensure all values have consistent types
                    parent_id = str(row.get('parent_id', ''))
                    segment_type = str(row.get('segment_type', 'sidewalk'))
                    is_crosswalk = bool(row.get('is_crosswalk', False))
                    length = float(row.geometry.length)
                    
                    segments_data.append({
                        'segment_id': int(idx),
                        'parent_id': parent_id,
                        'segment_type': segment_type,
                        'is_crosswalk': is_crosswalk,
                        'length': length,
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
        
        # make linestrings from start and end points
        from shapely.geometry import LineString
        linestrings = segments_gdf.geometry
        segments_gdf = gpd.GeoDataFrame(segments_data, geometry=linestrings)
        segments_gdf = cuspatial.from_geopandas(segments_gdf)
        
        self.logger.info(f"Converted {len(points_gdf)} points and {len(segments_gdf)} segments to GPU format")
        return points_gdf, segments_gdf

    def _compute_adjacency_via_graph_traversal(self, points_df: cudf.DataFrame,
                                              segments_df: cudf.DataFrame,
                                              batch_params: Dict) -> List[Dict]:
        """
        Compute adjacency relationships using efficient graph traversal with cuSpatial quadtree.
        """
        self.logger.info("Computing adjacency via graph traversal with cuSpatial quadtree...")
        
        # Pre-build spatial index for efficient queries
        spatial_index = self._build_spatial_index(points_df, segments_df)
    
        points_df = points_df
        segments_df = segments_df
        
        # Process in batches
        super_batch_size = batch_params.get('super_batch_size', 100)
        sub_batch_size = batch_params.get('sub_batch_size', 2000)
        
        all_results = []
        
        for super_batch_start in range(0, len(points_df), super_batch_size):
            super_batch_end = min(super_batch_start + super_batch_size, len(points_df))
            super_batch = points_df.iloc[super_batch_start:super_batch_end]
            
            self.logger.info(f"Processing super-batch {super_batch_start//super_batch_size + 1}/{(len(points_df) + super_batch_size - 1)//super_batch_size}")
            
            batch_results = []
            
            for sub_batch_start in range(0, len(super_batch), sub_batch_size):
                sub_batch_end = min(sub_batch_start + sub_batch_size, len(super_batch))
                sub_batch = super_batch.iloc[sub_batch_start:sub_batch_end]
                
                # Use vectorized processing instead of iterrows for GPU GeoDataFrames
                batch_results.extend(self._process_sub_batch_vectorized(
                    sub_batch, points_df, segments_df, spatial_index
                ))
            
            all_results.extend(batch_results)
        
        self.logger.info(f"Completed adjacency computation for {len(all_results)} points")
        return all_results

    def _process_sub_batch_vectorized(self, sub_batch: cudf.DataFrame,
                                     points_df: cudf.DataFrame,
                                     segments_df: cudf.DataFrame,
                                     spatial_index) -> List[Dict]:
        """
        Process a sub-batch of points using TRUE vectorized operations with CuPy/cuDF broadcasting.
        
        Parameters:
        -----------
        sub_batch : cudf.DataFrame
            Sub-batch of points to process
        points_df : cudf.DataFrame
            All points in the dataset
        segments_df : cudf.DataFrame
            All segments in the dataset
        spatial_index : Dict
            Pre-built spatial index
            
        Returns:
        --------
        List[Dict]
            Adjacency results for the sub-batch
        """
        if len(sub_batch) == 0:
            return []
        
        # Check if we have enough points for vectorized processing
        if len(sub_batch) == 1:
            self.logger.warning("Single point in sub-batch - this may cause issues with vectorized processing")
        
        # Extract all coordinates for TRUE vectorized processing
        sub_batch_coords = cp.column_stack([
            sub_batch['x'].values,
            sub_batch['y'].values
        ])
        
        # Extract all point metadata at once
        sub_batch_point_ids = sub_batch['point_id'].values
        sub_batch_source_segments = sub_batch['source_segment_idx'].values
        sub_batch_distances = sub_batch['distance_along_segment'].values
        sub_batch_is_intersection = sub_batch['is_intersection'].values
        # Handle case where is_pedestrian_ramp column might not exist
        if 'is_pedestrian_ramp' in sub_batch.columns:
            sub_batch_is_pedestrian_ramp = sub_batch['is_pedestrian_ramp'].values
        else:
            sub_batch_is_pedestrian_ramp = cp.zeros(len(sub_batch), dtype=bool)
        
        # Get all points coordinates for distance calculations
        all_points_coords = cp.column_stack([
            points_df['x'].values,
            points_df['y'].values
        ])
        
        # Get all segments coordinates for segment-based adjacency (if available)
        if 'start_x' in segments_df.columns and 'start_y' in segments_df.columns:
            all_segments_start = cp.column_stack([
                segments_df['start_x'].values,
                segments_df['start_y'].values
            ])
            all_segments_end = cp.column_stack([
                segments_df['end_x'].values,
                segments_df['end_y'].values
            ])
        else:
            raise ValueError("Segments DataFrame does not contain start_x and start_y columns")
        
        # TRUE VECTORIZED PROCESSING: Process entire sub-batch at once
        batch_results = self._find_adjacent_points_batch_vectorized(
            sub_batch_coords, sub_batch_point_ids, sub_batch_source_segments, 
            sub_batch_distances, sub_batch_is_intersection, sub_batch_is_pedestrian_ramp,
            all_points_coords, points_df, segments_df, spatial_index
        )
        
        return batch_results

    def _find_adjacent_points_batch_vectorized(self, sub_batch_coords: cp.ndarray,
                                              sub_batch_point_ids: cp.ndarray,
                                              sub_batch_source_segments: cp.ndarray,
                                              sub_batch_distances: cp.ndarray,
                                              sub_batch_is_intersection: cp.ndarray,
                                              sub_batch_is_pedestrian_ramp: cp.ndarray,
                                              all_points_coords: cp.ndarray,
                                              points_df: cudf.DataFrame,
                                              segments_df: cudf.DataFrame,
                                              spatial_index) -> List[Dict]:
        """
        TRUE vectorized adjacency computation for entire batch of points.
        
        Parameters:
        -----------
        sub_batch_coords : cp.ndarray
            Coordinates of all points in sub-batch (shape: (n_points, 2))
        sub_batch_point_ids : cp.ndarray
            Point IDs for all points in sub-batch
        sub_batch_source_segments : cp.ndarray
            Source segment indices for all points in sub-batch
        sub_batch_distances : cp.ndarray
            Distances along segments for all points in sub-batch
        sub_batch_is_intersection : cp.ndarray
            Intersection flags for all points in sub-batch
        sub_batch_is_pedestrian_ramp : cp.ndarray
            Pedestrian ramp flags for all points in sub-batch
        all_points_coords : cp.ndarray
            Coordinates of all points in dataset
        points_df : cudf.DataFrame
            All points DataFrame
        segments_df : cudf.DataFrame
            All segments DataFrame
        spatial_index : Dict
            Spatial index
            
        Returns:
        --------
        List[Dict]
            Adjacency results for all points in batch
        """
        batch_results = []
        
        # Step 1: Find containing segments for all points in batch using vectorized operations
        containing_segments = self._find_containing_segments_batch_vectorized(
            sub_batch_coords, segments_df, spatial_index
        )
        
        # Step 2: Find segment-based adjacencies for all points
        segment_adjacencies = self._find_segment_adjacencies_batch_vectorized(
            sub_batch_coords, sub_batch_point_ids, sub_batch_distances,
            containing_segments, all_points_coords, points_df, spatial_index
        )
        
        # Step 3: Find intersection-based adjacencies for intersection points
        intersection_adjacencies = self._find_intersection_adjacencies_batch_vectorized(
            sub_batch_coords, sub_batch_point_ids, sub_batch_is_intersection, 
            sub_batch_is_pedestrian_ramp, all_points_coords, points_df, segments_df, spatial_index
        )
        
        # Step 4: Combine results for each point
        for i in range(len(sub_batch_coords)):
            point_id = int(sub_batch_point_ids[i])
            
            # Combine segment and intersection adjacencies
            all_adjacent_points = []
            all_adjacent_distances = []
            
            # Add segment-based adjacencies
            if i in segment_adjacencies:
                seg_points, seg_distances = segment_adjacencies[i]
                all_adjacent_points.extend(seg_points)
                all_adjacent_distances.extend(seg_distances)
            
            # Add intersection-based adjacencies
            if i in intersection_adjacencies:
                int_points, int_distances = intersection_adjacencies[i]
                all_adjacent_points.extend(int_points)
                all_adjacent_distances.extend(int_distances)
            
            # Create result record
            if all_adjacent_points:
                batch_results.append({
                    'point_id': point_id,
                    'adjacent_points': ','.join(map(str, all_adjacent_points)),
                    'adjacent_distances': ','.join(map(str, all_adjacent_distances)),
                    'adjacency_count': len(all_adjacent_points)
                })
            else:
                batch_results.append({
                    'point_id': point_id,
                    'adjacent_points': '',
                    'adjacent_distances': '',
                    'adjacency_count': 0
                })
        
        return batch_results

    def _find_containing_segments_batch_vectorized(self, sub_batch_coords: cp.ndarray,
                                                  segments_df: cudf.DataFrame,
                                                  spatial_index) -> List[Optional[cp.ndarray]]:
        """
        Find containing segments for all points in batch using vectorized operations.
        """
        if spatial_index is None:
            return [None] * len(sub_batch_coords)
        
        try:
            import cuspatial
            
            # Convert all points in batch to cuSpatial GeoSeries
            point_gs = cuspatial.GeoSeries.from_points_xy(sub_batch_coords.flatten())
            
            # Use quadtree to find nearest linestrings for all points at once
            # Pass the geometry as a GeoSeries (which is what the function expects)
            nearest_results = cuspatial.quadtree_point_to_nearest_linestring(
                spatial_index['linestring_quad_pairs'], 
                spatial_index['quadtree'], 
                spatial_index['key_to_point'],
                point_gs, 
                segments_df.geometry  # Pass the GeoSeries of geometries
            )
            
            # Process results for each point
            containing_segments = []
            tolerance = 3.0  # 3 foot tolerance
            
            for i in range(len(sub_batch_coords)):
                # Find results for this specific point
                point_results = nearest_results[nearest_results['point_index'] == i]
                
                if len(point_results) > 0:
                    min_distance = point_results['distance'].iloc[0]
                    if min_distance < tolerance:
                        nearest_idx = point_results['linestring_index'].iloc[0]
                        containing_segments.append(segments_df.iloc[nearest_idx])
                    else:
                        containing_segments.append(None)
                else:
                    containing_segments.append(None)
            
            return containing_segments
            
        except Exception as e:
            self.logger.info(f"Error in batch vectorized segment finding: {e}")
            # Fallback: return None for all points
            return [None] * len(sub_batch_coords)

    def _find_segment_adjacencies_batch_vectorized(self, sub_batch_coords: cp.ndarray,
                                                  sub_batch_point_ids: cp.ndarray,
                                                  sub_batch_distances: cp.ndarray,
                                                  containing_segments: List[Optional[cp.ndarray]],
                                                  all_points_coords: cp.ndarray,
                                                  points_df: cudf.DataFrame,
                                                  spatial_index) -> Dict[int, Tuple[List[int], List[float]]]:
        """
        Find segment-based adjacencies for all points in batch using vectorized operations.
        """
        segment_adjacencies = {}
        
        # Group points by their containing segments for efficient processing
        segment_groups = {}
        for i, segment in enumerate(containing_segments):
            if segment is not None:
                segment_id = int(segment['segment_id']) if 'segment_id' in segment.index else 0
                if segment_id not in segment_groups:
                    segment_groups[segment_id] = []
                segment_groups[segment_id].append(i)
        
        # Process each segment group
        for segment_id, point_indices in segment_groups.items():
            # Find all points on this segment
            segment_points = points_df[points_df['source_segment_idx'] == segment_id].copy()
            
            if len(segment_points) > 1:
                # Find adjacencies for all points in this segment group
                for point_idx in point_indices:
                    point_id = int(sub_batch_point_ids[point_idx])
                    distance_along = float(sub_batch_distances[point_idx])
                    
                    # Find forward and backward adjacencies
                    forward_result = self._find_next_point_along_segment_vectorized(
                        point_id, distance_along, segment_points, 'forward'
                    )
                    backward_result = self._find_next_point_along_segment_vectorized(
                        point_id, distance_along, segment_points, 'backward'
                    )
                    
                    adjacent_points = []
                    adjacent_distances = []
                    
                    if forward_result is not None:
                        forward_point_id, forward_distance = forward_result
                        adjacent_points.append(forward_point_id)
                        adjacent_distances.append(forward_distance)
                    
                    if backward_result is not None:
                        backward_point_id, backward_distance = backward_result
                        adjacent_points.append(backward_point_id)
                        adjacent_distances.append(backward_distance)
                    
                    if adjacent_points:
                        segment_adjacencies[point_idx] = (adjacent_points, adjacent_distances)
        
        return segment_adjacencies

    def _find_intersection_adjacencies_batch_vectorized(self, sub_batch_coords: cp.ndarray,
                                                       sub_batch_point_ids: cp.ndarray,
                                                       sub_batch_is_intersection: cp.ndarray,
                                                       sub_batch_is_pedestrian_ramp: cp.ndarray,
                                                       all_points_coords: cp.ndarray,
                                                       points_df: cudf.DataFrame,
                                                       segments_df: cudf.DataFrame,
                                                       spatial_index) -> Dict[int, Tuple[List[int], List[float]]]:
        """
        Find intersection-based adjacencies for all intersection points in batch using vectorized operations.
        """
        intersection_adjacencies = {}
        
        # Find intersection points in this batch
        intersection_indices = cp.where(sub_batch_is_intersection | sub_batch_is_pedestrian_ramp)[0]
        
        if len(intersection_indices) == 0:
            return intersection_adjacencies
        
        try:
            import cuspatial
            
            # Process intersection points in smaller batches to avoid memory issues
            batch_size = 100
            for batch_start in range(0, len(intersection_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(intersection_indices))
                batch_indices = intersection_indices[batch_start:batch_end]
                
                # Get coordinates for this batch of intersection points
                batch_coords = sub_batch_coords[batch_indices]
                batch_point_ids = sub_batch_point_ids[batch_indices]
                
                # Convert to cuSpatial GeoSeries
                batch_gs = cuspatial.GeoSeries.from_points_xy(batch_coords.flatten())
                
                # Find nearby segments for all intersection points in batch
                nearby_results = cuspatial.quadtree_point_to_nearest_linestring(
                    spatial_index['linestring_quad_pairs'], 
                    spatial_index['quadtree'], 
                    spatial_index['key_to_point'],
                    batch_gs, 
                    segments_df.geometry  # Pass the GeoSeries of geometries
                )
                
                # Process results for each intersection point
                tolerance = 5.0
                for i, point_idx in enumerate(batch_indices):
                    point_id = int(batch_point_ids[i])
                    
                    # Find results for this specific point
                    point_results = nearby_results[nearby_results['point_index'] == i]
                    nearby_segments = point_results[point_results['distance'] < tolerance]
                    
                    connected_points = []
                    connected_distances = []
                    
                    # For each connected segment, find the closest point
                    for segment_idx in nearby_segments['linestring_index'].values:
                        segment = segments_df.iloc[segment_idx]
                        
                        # Find points on this segment
                        segment_points = points_df[points_df['source_segment_idx'] == segment_idx].copy()
                        
                        if len(segment_points) > 0:
                            # Find the point closest to this intersection using vectorized distance calculation
                            segment_point_coords = cp.column_stack([
                                segment_points['x'].values,
                                segment_points['y'].values
                            ])
                            
                            # Compute distances from intersection to all segment points
                            point_coord = sub_batch_coords[point_idx:point_idx+1]
                            distances = cp.sqrt(cp.sum((segment_point_coords - point_coord) ** 2, axis=1))
                            
                            # Find closest point
                            closest_idx = cp.argmin(distances)
                            closest_point_id = int(segment_points.iloc[closest_idx]['point_id'])
                            closest_distance = float(distances[closest_idx])
                            
                            # Don't include the intersection point itself
                            if closest_point_id != point_id:
                                connected_points.append(closest_point_id)
                                connected_distances.append(closest_distance)
                    
                    if connected_points:
                        intersection_adjacencies[point_idx] = (connected_points, connected_distances)
            
            return intersection_adjacencies
            
        except Exception as e:
            self.logger.info(f"Error in batch vectorized intersection finding: {e}")
            return intersection_adjacencies



    def _find_next_point_along_segment_vectorized(self, point_id: int,
                                                 current_distance: float,
                                                 segment_points: cudf.DataFrame,
                                                 direction: str) -> Optional[Tuple[int, float]]:
        """
        Find the next point along a segment in the specified direction using vectorized operations.
        """
        # Remove the current point from consideration
        other_points = segment_points[segment_points['point_id'] != point_id].copy()
        
        if len(other_points) == 0:
            return None
        
        if direction == 'forward':
            # Find points with higher distance_along_segment
            candidates = other_points[other_points['distance_along_segment'] > current_distance]
            if len(candidates) > 0:
                # Return the closest one
                closest_idx = (candidates['distance_along_segment'] - current_distance).idxmin()
                closest_point_id = int(candidates.loc[closest_idx, 'point_id'])
                closest_distance = float(candidates.loc[closest_idx, 'distance_along_segment'] - current_distance)
                return (closest_point_id, closest_distance)
        else:  # backward
            # Find points with lower distance_along_segment
            candidates = other_points[other_points['distance_along_segment'] < current_distance]
            if len(candidates) > 0:
                # Return the closest one
                closest_idx = (current_distance - candidates['distance_along_segment']).idxmin()
                closest_point_id = int(candidates.loc[closest_idx, 'point_id'])
                closest_distance = float(current_distance - candidates.loc[closest_idx, 'distance_along_segment'])
                return (closest_point_id, closest_distance)
        
        return None

    def _build_spatial_index(self, points_df: cuspatial.GeoDataFrame, segments_df: cuspatial.GeoDataFrame):
        """
        Build spatial index using cuSpatial quadtree for efficient spatial queries.
        """
        try:
            import cuspatial
            
            points_df_cpu = points_df.to_geopandas()
            segments_df_cpu = segments_df.to_geopandas()
            points_bounds = points_df_cpu.total_bounds
            segments_bounds = segments_df_cpu.total_bounds
            # get global min and max of points_bounds and segments_bounds
            # each is a tuple of (minx, miny, maxx, maxy)
            min_x = min(points_bounds[0], segments_bounds[0])
            min_y = min(points_bounds[1], segments_bounds[1])
            max_x = max(points_bounds[2], segments_bounds[2])
            max_y = max(points_bounds[3], segments_bounds[3])

            # Build quadtree from points
            scale = max(max_x - min_x, max_y - min_y) / (1 << 8)
            max_depth = 8
            max_size = 10
            
            key_to_point, quadtree = cuspatial.quadtree_on_points(
                points_df.geometry,
                min_x, max_x, min_y, max_y,
                scale, max_depth, max_size
            )
            


            line_bboxes = cuspatial.linestring_bounding_boxes(segments_df.geometry, expansion_radius=1.0)



            
            # Join quadtree with segment bounding boxes
            linestring_quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
                quadtree, line_bboxes,
                min_x, max_x, min_y, max_y,
                scale, max_depth
            )
            
            result = cuspatial.quadtree_point_to_nearest_linestring(
                linestring_quad_pairs,
                quadtree,
                key_to_point,
                points_df.geometry,
                segments_df.geometry
            )

            print(result)
            
        except ImportError:
            self.logger.warning("cuSpatial not available, using CPU spatial index")
            return None





    def _update_points_with_adjacency(self, points: List[Dict], 
                                     adjacency_results: List[Dict]) -> List[Dict]:
        """
        Update points with adjacency information.
        
        Parameters:
        -----------
        points : List[Dict]
            Original point records
        adjacency_results : List[Dict]
            Adjacency computation results
            
        Returns:
        --------
        List[Dict]
            Updated points with adjacency information
        """
        # Create lookup dictionary for adjacency results
        adjacency_lookup = {result['point_id']: result for result in adjacency_results}
        
        # Log adjacency results summary
        self.logger.info(f"Updating {len(points)} points with {len(adjacency_results)} adjacency results")
        self.logger.info(f"Unique point_ids in adjacency results: {len(adjacency_lookup)}")
        
        # Check for duplicate point_ids in adjacency results
        point_id_counts = {}
        for result in adjacency_results:
            point_id = result['point_id']
            point_id_counts[point_id] = point_id_counts.get(point_id, 0) + 1
        
        duplicate_point_ids = {pid: count for pid, count in point_id_counts.items() if count > 1}
        if duplicate_point_ids:
            self.logger.warning(f"Found {len(duplicate_point_ids)} duplicate point_ids in adjacency results:")
            for pid, count in list(duplicate_point_ids.items())[:5]:  # Show first 5
                self.logger.warning(f"  Point_id {pid}: {count} results")
        
        # Update points with adjacency information
        updated_points = []
        points_with_adjacency = 0
        for point in points:
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
        return updated_points

    def _log_adjacency_statistics(self, points: List[Dict], total_time: float):
        """
        Log statistics about the adjacency computation.
        
        Parameters:
        -----------
        points : List[Dict]
            Points with adjacency information
        total_time : float
            Total processing time
        """
        if not points:
            return
        
        # Compute statistics
        adjacency_counts = [point.get('adjacency_count', 0) for point in points]
        total_adjacencies = sum(adjacency_counts)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ADJACENCY COMPUTATION COMPLETED")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Processing time: {total_time:.2f} seconds")
        self.logger.info(f"Total points: {len(points)}")
        self.logger.info(f"Total adjacencies: {total_adjacencies}")
        self.logger.info(f"Average adjacencies per point: {total_adjacencies/len(points):.2f}")
        self.logger.info(f"Max adjacencies per point: {max(adjacency_counts)}")
        self.logger.info(f"Min adjacencies per point: {min(adjacency_counts)}")
        
        # Distribution of adjacency counts
        unique_counts = set(adjacency_counts)
        for count in sorted(unique_counts):
            num_points = adjacency_counts.count(count)
            self.logger.info(f"  - {num_points} points with {count} adjacencies")
        
        self.logger.info("=" * 60)
    
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
        result.attrs['sampling_summary'] = self._compute_sampling_summary(topology_points)
        
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
        
        # Filter citywide centerlines to only include those relevant to neighborhood segments
        self.logger.info("Step 1c: Filtering citywide centerlines to neighborhood area...")
        filtered_centerlines = self._filter_centerlines_to_neighborhood(citywide_centerlines, segments_gdf, buffer_distance)
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
        
        # Step 4: Save final result
        if output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
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
        
        # Final statistics
        total_time = time.time() - start_time
        self.logger.info(f"\nTotal processing time: {total_time:.2f} seconds")
        self.logger.info(f"Final network: {len(final_network)} points")
        
        exit()


    
    # Removed _load_segments method - no longer needed with block-level processing

    def _filter_centerlines_to_neighborhood(self, centerlines_gdf: gpd.GeoDataFrame, neighborhood_segments: gpd.GeoDataFrame, buffer_distance: float = 100.0) -> gpd.GeoDataFrame:
        """
        Filter citywide centerlines to only include those relevant to neighborhood segments.
        Similar to how pedestrian ramps are filtered.
        
        Parameters:
        -----------
        centerlines_gdf : GeoDataFrame
            Citywide centerlines GeoDataFrame
        neighborhood_segments : GeoDataFrame
            Neighborhood segments GeoDataFrame
        buffer_distance : float
            Maximum distance to consider a centerline relevant to neighborhood
            
        Returns:
        --------
        GeoDataFrame
            Filtered centerlines relevant to neighborhood
        """
        # Create a spatial index for the neighborhood segments
        segment_tree = STRtree(list(neighborhood_segments.geometry))
        
        # Find centerlines that are within buffer_distance of any neighborhood segment
        filtered_centerlines = []
        
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
                    filtered_centerlines.append(idx)
        
        # Return filtered GeoDataFrame
        if filtered_centerlines:
            return centerlines_gdf.iloc[filtered_centerlines].reset_index(drop=True)
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
    



if __name__ == "__main__":
    # Use Google Fire for CLI
    fire.Fire(SampledPointNetwork) 
