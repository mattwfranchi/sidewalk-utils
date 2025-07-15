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
            intersection_processed = self._process_intersection_points(
                filtered_points, validated_segments, sampling_params
            )
        else:
            intersection_processed = filtered_points
        
        # Step 7: Compute network topology
        self.logger.info("Step 7: Computing network topology...")
        topology_enhanced = self._compute_network_topology(
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
            'intersection_buffer_factor': 1.5,  # Larger buffer around intersections
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
        
        # Add parent_id to candidate points based on source segment
        for candidate in candidate_points:
            # Get parent_id from source segment attributes
            if 'source_parent_id' not in candidate:
                candidate['source_parent_id'] = candidate.get('parent_id')
        
        # Group candidate points by parent_id
        candidates_by_parent_id = defaultdict(list)
        for candidate in candidate_points:
            parent_id = candidate.get('source_parent_id', candidate.get('parent_id'))
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
            filtered_parent_points = self._apply_buffer_filtering_within_parent_id(
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
    
    def _apply_buffer_filtering_within_parent_id(self, candidate_points: List[Dict],
                                                existing_points: List[Dict],
                                                sampling_params: Dict) -> List[Dict]:
        """
        Apply buffer filtering within a single parent_id group using GPU acceleration.
        
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
        if not candidate_points:
            return []
        
        return self._apply_buffer_filtering_parent_id(
            candidate_points, existing_points, sampling_params
        )
    

    
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
    
    def _apply_buffer_filtering(self, candidate_points: List[Dict],
                               sampling_params: Dict) -> List[Dict]:
        """
        Apply buffer-based filtering to maintain minimum distance between points using GPU acceleration.
        
        Parameters:
        -----------
        candidate_points : List[Dict]
            List of candidate point records
        sampling_params : Dict
            Sampling parameters
            
        Returns:
        --------
        List[Dict]
            Filtered point records
        """
        return self._apply_buffer_filtering_impl(candidate_points, sampling_params)
    

    
    def _apply_buffer_filtering_impl(self, candidate_points: List[Dict],
                                   sampling_params: Dict) -> List[Dict]:
        """
        GPU-accelerated buffer filtering implementation.
        
        Parameters:
        -----------
        candidate_points : List[Dict]
            List of candidate point records
        sampling_params : Dict
            Sampling parameters
            
        Returns:
        --------
        List[Dict]
            Filtered point records
        """
        self.logger.info(f"Applying GPU-accelerated buffer filtering to {len(candidate_points)} candidates...")
        
        buffer_distance = sampling_params['buffer_distance']
        buffer_distance_squared = sampling_params['buffer_distance_squared']
        
        # Extract coordinates for GPU processing
        coords = np.array([[pt['geometry'].x, pt['geometry'].y] for pt in candidate_points])
        
        # Transfer to GPU
        coords_gpu = cp.asarray(coords)
        
        # Sort by segment and position for consistent processing
        sort_keys = [(pt['source_segment_idx'], pt['distance_along_segment']) for pt in candidate_points]
        sort_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
        
        sorted_coords = coords_gpu[sort_indices]
        
        # GPU-accelerated filtering
        accepted_indices = []
        
        for i in range(len(sorted_coords)):
            if i % 10000 == 0:
                self.logger.info(f"  GPU processing {i + 1}/{len(sorted_coords)} ({i/len(sorted_coords)*100:.1f}%)")
            
            if not accepted_indices:
                accepted_indices.append(i)
                continue
            
            # Calculate distances to all accepted points
            current_point = sorted_coords[i:i+1]  # Keep 2D shape (1, 2)
            accepted_coords = sorted_coords[accepted_indices]
            
            # Vectorized distance calculation
            diffs = accepted_coords - current_point
            distances_squared = cp.sum(diffs * diffs, axis=1)
            
            # Check if any distance is too small
            if cp.min(distances_squared) >= buffer_distance_squared:
                accepted_indices.append(i)
        
        # Transfer results back to CPU
        accepted_indices_cpu = cp.asnumpy(cp.array(accepted_indices))
        
        # Create filtered point list
        accepted_points = []
        for i, orig_idx in enumerate(accepted_indices_cpu):
            sorted_idx = sort_indices[orig_idx]
            point = candidate_points[sorted_idx].copy()
            point['buffer_zone'] = point['geometry'].buffer(buffer_distance)
            point['point_id'] = i
            accepted_points.append(point)
        
        self.logger.info(f"GPU buffer filtering complete: {len(accepted_points)} points accepted ({len(accepted_points)/len(candidate_points)*100:.1f}%)")
        return accepted_points
    
    def _process_intersection_points(self, filtered_points: List[Dict],
                                    segments_gdf: gpd.GeoDataFrame,
                                    sampling_params: Dict) -> List[Dict]:
        """
        Process intersection points to ensure they are properly represented.
        Uses GPU acceleration when available for much faster processing.
        
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
        self.logger.info("Processing intersection points...")
        
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
        return self._find_intersections_cuspatial(segments_gdf)
    
    def _find_intersections_cuspatial(self, segments_gdf: gpd.GeoDataFrame) -> List[Point]:
        """
        Find intersection points using cuSpatial for GPU acceleration.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input segments
            
        Returns:
        --------
        List[Point]
            List of intersection points
        """
        # Use GeoPandas with GPU acceleration for coordinate operations
        geometries = segments_gdf.geometry.tolist()
        intersections = []
        
        self.logger.info(f"Processing {len(geometries)} segments for intersection detection...")
        
        # Use spatial index for efficient intersection queries
        from shapely.strtree import STRtree
        spatial_index = STRtree(geometries)
        
        # Process in batches to show progress
        batch_size = 100
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
            unique_intersections = self._remove_duplicate_points_impl(intersections)
            self.logger.info(f"After duplicate removal: {len(unique_intersections)} unique intersections")
            return unique_intersections
        else:
            self.logger.info("No intersections found")
            return []
    
    def _remove_duplicate_points_impl(self, points: List[Point]) -> List[Point]:
        """
        Remove duplicate points using GPU-accelerated distance calculations.
        
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
        
        for i in range(1, len(coords)):
            current_point = coords[i]
            
            # Calculate squared distances to all unique points
            unique_coords = coords[unique_indices]
            diffs = unique_coords - current_point
            distances_squared = cp.sum(diffs * diffs, axis=1)
            
            # Check if point is unique (not too close to existing points)
            if cp.min(distances_squared) >= tolerance_squared:
                unique_indices.append(i)
        
        # Convert back to CPU and return unique points
        unique_indices_cpu = cp.asnumpy(cp.array(unique_indices))
        return [points[i] for i in unique_indices_cpu]
    
    def _mark_intersection_points_impl(self, filtered_points: List[Dict],
                                      intersection_points: List[Point],
                                      intersection_buffer: float) -> List[Dict]:
        """
        Mark points near intersections using GPU acceleration.
        
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
        # Extract coordinates from filtered points
        point_coords = cp.array([[pt['geometry'].x, pt['geometry'].y] for pt in filtered_points])
        intersection_coords = cp.array([[ipt.x, ipt.y] for ipt in intersection_points])
        
        # Calculate distance matrix using GPU
        # For each point, find distance to all intersections
        buffer_squared = intersection_buffer * intersection_buffer
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        
        for start_idx in range(0, len(point_coords), batch_size):
            end_idx = min(start_idx + batch_size, len(point_coords))
            batch_coords = point_coords[start_idx:end_idx]
            
            # Calculate distances from batch to all intersections
            # Shape: (batch_size, num_intersections)
            diffs = batch_coords[:, cp.newaxis, :] - intersection_coords[cp.newaxis, :, :]
            distances_squared = cp.sum(diffs * diffs, axis=2)
            
            # Find minimum distance for each point
            min_distances_squared = cp.min(distances_squared, axis=1)
            min_distance_indices = cp.argmin(distances_squared, axis=1)
            
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
        
        return filtered_points
    

    
    def _compute_network_topology(self, points: List[Dict],
                                 sampling_params: Dict) -> List[Dict]:
        """
        Compute network topology relationships between points using GPU acceleration.
        
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
        self.logger.info("Computing network topology with GPU acceleration...")
        
        # Group points by source segment
        segment_groups = defaultdict(list)
        for point in points:
            segment_idx = point['source_segment_idx']
            segment_groups[segment_idx].append(point)
        
        # Sort points within each segment by position
        for segment_idx in segment_groups:
            # Ensure all points have distance_along_segment key
            for point in segment_groups[segment_idx]:
                if 'distance_along_segment' not in point:
                    self.logger.warning(f"Point {point.get('point_id', 'unknown')} missing distance_along_segment, using 0.0")
                    point['distance_along_segment'] = 0.0
            
            segment_groups[segment_idx].sort(key=lambda x: x['distance_along_segment'])
        
        # Use GPU-accelerated topology computation
        return self._compute_network_topology_gpu(points, sampling_params)
    
    def _compute_network_topology_gpu(self, points: List[Dict],
                                     sampling_params: Dict) -> List[Dict]:
        """
        GPU-accelerated network topology computation using cuSpatial and CuPy.
        
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
        
        # Use cuSpatial spatial indexing
        return self._compute_topology_with_cuspatial(points, coords_gpu, segment_indices_gpu, 
                                                   point_ids_gpu, is_intersection_gpu, topology_buffer)
    
    def _compute_topology_with_cuspatial(self, points: List[Dict],
                                        coords_gpu: cp.ndarray,
                                        segment_indices_gpu: cp.ndarray,
                                        point_ids_gpu: cp.ndarray,
                                        is_intersection_gpu: cp.ndarray,
                                        topology_buffer: float) -> List[Dict]:
        """
        Use cuSpatial for efficient spatial indexing in topology computation.
        
        Parameters:
        -----------
        points : List[Dict]
            Point records
        coords_gpu : cp.ndarray
            Point coordinates on GPU
        segment_indices_gpu : cp.ndarray
            Segment indices on GPU
        point_ids_gpu : cp.ndarray
            Point IDs on GPU
        is_intersection_gpu : cp.ndarray
            Intersection flags on GPU
        topology_buffer : float
            Topology buffer distance
            
        Returns:
        --------
        List[Dict]
            Points with topology information
        """
        self.logger.info("Using cuSpatial for topology spatial indexing...")
        
        # Create cuSpatial point dataset
        # Extract x and y coordinates separately for cuSpatial
        x_coords = coords_gpu[:, 0]
        y_coords = coords_gpu[:, 1]

        
        coords = cudf.DataFrame({"x": x_coords, "y": y_coords}).interleave_columns()

        points_gdf = cuspatial.GeoDataFrame({
            'geometry': cuspatial.GeoSeries.from_points_xy(coords),
            'point_id': point_ids_gpu,
            'segment_idx': segment_indices_gpu,
            'is_intersection': is_intersection_gpu
        })

        points_gdf = points_gdf.to_geopandas()
        
        # Create buffer around each point
        buffers = points_gdf.geometry.buffer(topology_buffer)

        #points_gdf = cuspatial.from_geopandas(points_gdf)
        
        # Process each point
        for idx in range(len(points_gdf)):
            if idx % 1000 == 0:
                self.logger.info(f"  Processing topology point {idx + 1}/{len(points_gdf)}")
            
            # Get buffer for this point
            point_buffer = buffers.iloc[idx]
            
            # Find points within buffer (excluding self)
            within_buffer = points_gdf[points_gdf.geometry.within(point_buffer)]
            
            if len(within_buffer) > 1:  # More than just self
                # Remove self from results
                within_buffer = within_buffer[within_buffer.index != idx]
                
                if len(within_buffer) > 0:
                    # Calculate distances
                    point_coord = coords_gpu[idx:idx+1]
                    neighbor_coords = coords_gpu[within_buffer.index.values]
                    
                    # Vectorized distance calculation
                    diffs = neighbor_coords - point_coord
                    distances = cp.sqrt(cp.sum(diffs * diffs, axis=1))
                    
                    # Sort by distance
                    sort_indices = cp.argsort(distances).get()
                    sorted_distances = distances[sort_indices]
                    sorted_indices = within_buffer.index.values[sort_indices]
                    
                    # Limit to 10 nearest neighbors
                    max_neighbors = min(10, len(sorted_indices))
                    sorted_distances = sorted_distances[:max_neighbors]
                    sorted_indices = sorted_indices[:max_neighbors]
                    
                    # Convert to CPU for relationship classification
                    distances_cpu = cp.asnumpy(sorted_distances)
                    indices_cpu = cp.asnumpy(sorted_indices)
                    
                    # Create neighbor records
                    neighbors = []
                    for j, (neighbor_idx, distance) in enumerate(zip(indices_cpu, distances_cpu)):
                        neighbor_point = points[neighbor_idx]
                        relationship = self._classify_relationship_gpu(
                            segment_indices_gpu[idx], is_intersection_gpu[idx],
                            segment_indices_gpu[neighbor_idx], is_intersection_gpu[neighbor_idx]
                        )
                        
                        neighbors.append({
                            'point_id': neighbor_point['point_id'],
                            'distance': float(distance),
                            'relationship': relationship
                        })
                    
                    points[idx]['network_neighbors'] = neighbors
                else:
                    points[idx]['network_neighbors'] = []
            else:
                points[idx]['network_neighbors'] = []
        
        self.logger.info(f"cuSpatial topology computation complete for {len(points)} points")
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
    
    def _classify_relationship(self, point1: Dict, point2: Dict) -> str:
        """
        Classify the relationship between two points.
        
        Parameters:
        -----------
        point1 : Dict
            First point
        point2 : Dict
            Second point
            
        Returns:
        --------
        str
            Relationship type
        """
        # Same segment
        if point1['source_segment_idx'] == point2['source_segment_idx']:
            return 'same_segment'
        
        # Both are intersections
        if point1.get('is_intersection') and point2.get('is_intersection'):
            return 'intersection_intersection'
        
        # One is intersection
        if point1.get('is_intersection') or point2.get('is_intersection'):
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
    
    def compute_adjacency_graph(self, sampled_network: gpd.GeoDataFrame,
                               connection_threshold: float = 200.0) -> gpd.GeoDataFrame:
        """
        Compute adjacency relationships between sampled points using GPU acceleration.
        
        Parameters:
        -----------
        sampled_network : GeoDataFrame
            Sampled point network
        connection_threshold : float
            Maximum distance for adjacency connections
            
        Returns:
        --------
        GeoDataFrame
            Network with adjacency information
        """
        self.logger.info("Computing adjacency relationships with GPU acceleration...")
        
        # Use GPU-accelerated adjacency computation
        return self._compute_adjacency_graph_gpu(sampled_network, connection_threshold)
    
    def _compute_adjacency_graph_gpu(self, sampled_network: gpd.GeoDataFrame,
                                    connection_threshold: float) -> gpd.GeoDataFrame:
        """
        GPU-accelerated adjacency computation using cuSpatial and CuPy.
        
        Parameters:
        -----------
        sampled_network : GeoDataFrame
            Sampled point network
        connection_threshold : float
            Maximum distance for adjacency connections
            
        Returns:
        --------
        GeoDataFrame
            Network with adjacency information
        """
        self.logger.info("Using GPU-accelerated adjacency computation...")
        
        # Extract coordinates and point IDs for GPU processing
        coords = np.array([[pt.x, pt.y] for pt in sampled_network.geometry])
        point_ids = sampled_network['point_id'].values
        
        # Transfer to GPU
        coords_gpu = cp.asarray(coords)
        point_ids_gpu = cp.asarray(point_ids)
        
        # Use cuSpatial for spatial indexing
        return self._compute_adjacency_with_cuspatial(sampled_network, coords_gpu, point_ids_gpu, connection_threshold)
    
    def _compute_adjacency_with_cuspatial(self, sampled_network: gpd.GeoDataFrame,
                                         coords_gpu: cp.ndarray,
                                         point_ids_gpu: cp.ndarray,
                                         connection_threshold: float) -> gpd.GeoDataFrame:
        """
        Use cuSpatial for efficient spatial indexing and nearest neighbor search.
        
        Parameters:
        -----------
        sampled_network : GeoDataFrame
            Sampled point network
        coords_gpu : cp.ndarray
            Point coordinates on GPU
        point_ids_gpu : cp.ndarray
            Point IDs on GPU
        connection_threshold : float
            Maximum distance for adjacency connections
            
        Returns:
        --------
        GeoDataFrame
            Network with adjacency information
        """
        self.logger.info("Using cuSpatial for spatial indexing...")
        
        # Create cuSpatial point dataset
        # Extract x and y coordinates separately for cuSpatial
        x_coords = coords_gpu[:, 0]
        y_coords = coords_gpu[:, 1]

        coords = cudf.DataFrame({"x": x_coords, "y": y_coords}).interleave_columns()
        
        points_gdf = cuspatial.GeoDataFrame({
            'geometry': cuspatial.GeoSeries.from_points_xy(coords),
            'point_id': point_ids_gpu
        })

        points_gdf = points_gdf.to_geopandas()
        
        # Create buffer around each point
        buffers = points_gdf.buffer(connection_threshold)

        
        
        # Spatial join to find intersecting points
        adjacency_info = []
        
        for idx in range(len(points_gdf)):
            if idx % 1000 == 0:
                self.logger.info(f"  Processing point {idx + 1}/{len(points_gdf)}")
            
            # Get buffer for this point
            point_buffer = buffers.iloc[idx]
            
            # Find points within buffer (excluding self)
            within_buffer = points_gdf[points_gdf.geometry.within(point_buffer)]
            
            if len(within_buffer) > 1:  # More than just self
                # Remove self from results
                within_buffer = within_buffer[within_buffer.index != idx]
                
                if len(within_buffer) > 0:
                    # Calculate distances
                    point_coord = coords_gpu[idx:idx+1]
                    neighbor_coords = coords_gpu[within_buffer.index.values]
                    
                    # Vectorized distance calculation
                    diffs = neighbor_coords - point_coord
                    distances = cp.sqrt(cp.sum(diffs * diffs, axis=1))
                    
                    # Sort by distance
                    sort_indices = cp.argsort(distances)
                    sorted_distances = distances[sort_indices]
                    sorted_point_ids = point_ids_gpu[within_buffer.index.values][sort_indices]
                    
                    # Convert to CPU
                    distances_cpu = cp.asnumpy(sorted_distances)
                    point_ids_cpu = cp.asnumpy(sorted_point_ids)
                    
                    # Create adjacency records
                    adjacent_points = []
                    for distance, point_id in zip(distances_cpu, point_ids_cpu):
                        adjacent_points.append({
                            'point_id': int(point_id),
                            'distance': float(distance)
                        })
                    
                    adjacency_info.append({
                        'adjacent_points': adjacent_points,
                        'adjacency_count': len(adjacent_points)
                    })
                else:
                    adjacency_info.append({
                        'adjacent_points': [],
                        'adjacency_count': 0
                    })
            else:
                adjacency_info.append({
                    'adjacent_points': [],
                    'adjacency_count': 0
                })
        
        # Add adjacency information to the network
        result = sampled_network.copy()
        result['adjacent_points'] = [info['adjacent_points'] for info in adjacency_info]
        result['adjacency_count'] = [info['adjacency_count'] for info in adjacency_info]
        
        self.logger.info(f"cuSpatial adjacency computation complete for {len(result)} points")
        self.logger.info(f"Average adjacency: {result['adjacency_count'].mean():.2f}")
        
        return result
    

    

    
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
        final_network = self.compute_adjacency_graph(sampled_network)
        
        # Step 4: Save final result
        if output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
            extension = output_path.rsplit('.', 1)[1] if '.' in output_path else 'parquet'
            dynamic_output_path = f"{base_path}_sampled_{timestamp}.{extension}"
            
            # Prepare for saving (remove complex objects)
            save_network = final_network.copy()
            save_network['adjacent_points'] = save_network['adjacent_points'].apply(
                lambda x: ','.join([f"{pt['point_id']}:{pt['distance']:.2f}" for pt in x[:5]])  # Limit to 5 nearest
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