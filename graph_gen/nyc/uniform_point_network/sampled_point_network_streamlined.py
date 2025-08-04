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

# Import helper modules
from centerline_helper import CenterlineHelper
from adjacency_helper import AdjacencyHelper
from sampling_helper import SamplingHelper
from gpu_helper import GPUHelper

# Try to import project constants
try:
    from data.nyc.c import PROJ_FT
except ImportError:
    print("WARNING: PROJ_FT not found. Using default NYC projection.")
    PROJ_FT = 'EPSG:2263'


class SampledPointNetwork:
    """
    Geometric sampling-based point network generator that creates spatially uniform point distributions
    along block-level sidewalk centerlines using buffer zones and minimum distance thresholds.
    
    Key features:
    - Block-Level Processing: Converts sidewalk polygons to centerlines
    - Buffer-based Sampling: Creates exclusion zones around placed points
    - Minimum Distance Thresholds: Maintains spatial separation
    - Network Topology Respect: Preserves structure and connectivity
    - Spatial Uniformity: Ensures even distribution across the network
    - GPU Acceleration: Uses RAPIDS for high-performance computation
    """
    
    def __init__(self):
        self.logger = get_logger("SampledPointNetwork")
        self.logger.setLevel("INFO")
        
        # Initialize helper modules
        self.adjacency_helper = AdjacencyHelper()
        
        # Check GPU backend status for NetworkX operations
        gpu_status = self.adjacency_helper.check_gpu_backend_status()
        
        self.sampling_helper = SamplingHelper()
        self.gpu_helper = GPUHelper()
        
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
                                pedestrian_ramps_path: Optional[str] = None,
                                compute_adjacency: bool = False,
                                walkshed_distance: float = 328.0,
                                crosswalk_distance: float = 100.0,
                                adjacency_centerlines: Optional[gpd.GeoDataFrame] = None,
                                save_intermediate: bool = False,
                                intermediate_dir: Optional[str] = None,
                                output_path: Optional[str] = None) -> gpd.GeoDataFrame:
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
            Whether to preserve intersection points in the network
        min_segment_length : float, default=10.0
            Minimum segment length to consider for sampling
        max_points_per_segment : int, default=100
            Maximum number of points per segment
        pedestrian_ramps_path : Optional[str], default=None
            Path to pedestrian ramps GeoJSON file
        compute_adjacency : bool, default=False
            Whether to compute adjacency relationships
        walkshed_distance : float, default=328.0
            Walkshed radius in feet (100 meters)
        crosswalk_distance : float, default=100.0
            Maximum distance for crosswalk creation in feet
        adjacency_centerlines : Optional[gpd.GeoDataFrame], default=None
            Centerlines for adjacency computation
        save_intermediate : bool, default=False
            Whether to save intermediate topology results for inspection
        intermediate_dir : Optional[str], default=None
            Directory for saving intermediate files
        output_path : Optional[str], default=None
            Output file path (used for determining intermediate directory if intermediate_dir not provided)
            
        Returns:
        --------
        GeoDataFrame
            Sampled point network with spatial buffers and topology information
        """
        self.logger.info("Starting geometric sampling network generation")
        start_time = time.time()
        
        # Step 1: Load and process pedestrian ramps
        pedestrian_ramps_points = self.sampling_helper.load_and_process_pedestrian_ramps(
            pedestrian_ramps_path, segments_gdf
        )
        
        # Step 2: Validate and prepare input segments
        validated_segments = self._validate_and_prepare_segments(segments_gdf, min_segment_length)
        
        if len(validated_segments) == 0:
            self.logger.error("No valid segments found after validation")
            return self._create_empty_result(segments_gdf.crs)
        
        # Step 3: Initialize sampling parameters
        if strategy != "uniform":
            self.logger.warning(f"Strategy '{strategy}' not supported. Using 'uniform'")
            strategy = "uniform"
        
        if buffer_distance <= sampling_interval:
            buffer_distance = sampling_interval * 2.0
            self.logger.info(f"Adjusted buffer distance to {buffer_distance} feet")
        
        sampling_params = {
            'buffer_distance': buffer_distance,
            'sampling_interval': sampling_interval,
            'strategy': strategy,
            'preserve_intersections': preserve_intersections,
            'buffer_distance_squared': buffer_distance ** 2,
            'max_iterations': 1000
        }
        
        self.logger.info(f"Sampling parameters: buffer={buffer_distance}ft, interval={sampling_interval}ft")
        
        # Store adjacency parameters
        self._compute_adjacency = compute_adjacency
        self._walkshed_distance = walkshed_distance
        self._crosswalk_distance = crosswalk_distance
        
        # Step 4: Generate crosswalk centerlines and compute segment network connectivity BEFORE candidate points
        self.logger.info("Step 4: Generating crosswalk centerlines and computing segment network connectivity...")
        if pedestrian_ramps_points:
            crosswalk_centerlines = self.sampling_helper.generate_intersection_centerlines(
                pedestrian_ramps_points, crosswalk_distance
            )
            
            validated_segments_with_flag = validated_segments.copy()
            if 'is_crosswalk' not in validated_segments_with_flag.columns:
                validated_segments_with_flag['is_crosswalk'] = False
            
            # Combine original segments with crosswalk centerlines for graph creation
            if len(crosswalk_centerlines) > 0:
                combined_segments_for_graph = pd.concat([validated_segments_with_flag, crosswalk_centerlines], ignore_index=True)
                self.logger.info(f"Combined {len(validated_segments_with_flag)} segments + {len(crosswalk_centerlines)} crosswalks = {len(combined_segments_for_graph)} total")
            else:
                combined_segments_for_graph = validated_segments_with_flag
                self.logger.info("No crosswalks generated - using original segments only")
        else:

            combined_segments_for_graph = validated_segments.copy()
            if 'is_crosswalk' not in combined_segments_for_graph.columns:
                combined_segments_for_graph['is_crosswalk'] = False
            self.logger.info("No pedestrian ramps - using original segments only")
        
        # Step 4b: Create NetworkX graph from segments + crosswalks (early graph creation)
        self.logger.info("Step 4b: Creating NetworkX graph from segments + crosswalks...")
        # Note: We'll create the graph without points first, then update it later with points
        network_graph = self.adjacency_helper.create_segment_network_graph(
            combined_segments_for_graph, []  # Empty points list for now
        )
        # Store the graph in the adjacency helper for later use
        self.adjacency_helper.segment_graph = network_graph
        
        # Save segment network connectivity for visualization
        if save_intermediate:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            
            # Determine output directory
            if intermediate_dir:
                topology_output_dir = intermediate_dir
            elif output_path:
                topology_output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
            else:
                topology_output_dir = '.'
                
            self._save_segment_network_connectivity(combined_segments_for_graph, 
                                                   topology_output_dir, timestamp)
        
        # Step 5: Generate candidate points (moved from step 4)
        self.logger.info("Step 5: Generating candidate points...")
        candidate_points = self.sampling_helper.generate_candidate_points(
            validated_segments, sampling_params, max_points_per_segment
        )
        
        if not candidate_points:
            self.logger.error("No candidate points generated")
            return self._create_empty_result(segments_gdf.crs)
        
        # Step 6: Apply parent_id-aware buffer filtering (moved from step 5)
        self.logger.info("Step 6: Applying parent_id-aware buffer filtering...")
        filtered_points = self.sampling_helper.apply_buffer_filtering(
            candidate_points, pedestrian_ramps_points, sampling_params
        )
        
        if not filtered_points:
            self.logger.error("No points survived buffer filtering")
            return self._create_empty_result(segments_gdf.crs)
        
        # Step 7: Handle intersections (for uniform sampling, skip special processing) - moved from step 6
        self.logger.info("Step 7: Handling intersections...")
        intersection_processed = filtered_points
        
        # Step 7b: Update NetworkX graph with actual sampled points
        self.logger.info("Step 7b: Updating NetworkX graph with sampled points...")
        network_graph = self.adjacency_helper.create_segment_network_graph(
            combined_segments_for_graph, intersection_processed
        )
        
        # Store the updated graph in the adjacency helper for later use
        self.adjacency_helper.segment_graph = network_graph
        
        # Store the points for adjacency computation (keeping same variable name for compatibility)
        topology_enhanced = intersection_processed
        
        # Save network topology results for inspection (now with correct SSSP distances)
        if save_intermediate:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            
            # Determine output directory with same logic as _save_inspection_files
            if intermediate_dir:
                topology_output_dir = intermediate_dir
            elif output_path:
                topology_output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
            else:
                topology_output_dir = '.'
                
            self._save_network_topology_results(topology_enhanced, topology_output_dir, timestamp)
        
        # Step 8: Extract adjacency data from NetworkX graph (if requested)
        if compute_adjacency:
            self.logger.info("Step 8: Extracting adjacency data from NetworkX graph...")
            adjacency_enhanced = self._extract_adjacency_from_graph(topology_enhanced)
        else:
            adjacency_enhanced = topology_enhanced
        
        # Save segment adjacency analysis now that adjacency computation is complete
        if save_intermediate:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            
            # Determine output directory with same logic as before
            if intermediate_dir:
                topology_output_dir = intermediate_dir
            elif output_path:
                topology_output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
            else:
                topology_output_dir = '.'
                
            self._save_segment_adjacency_analysis(adjacency_enhanced, topology_output_dir, timestamp)
        
        # Step 9: Create final GeoDataFrame
        self.logger.info("Step 9: Creating final GeoDataFrame...")
        result = self._create_final_network(adjacency_enhanced, segments_gdf.crs)
        
        # Store combined centerlines for inspection
        if compute_adjacency and 'combined_centerlines' in locals():
            result.attrs['combined_centerlines_with_crosswalks'] = combined_centerlines
        else:
            result.attrs['combined_centerlines_with_crosswalks'] = None
        
        # Log final statistics
        total_time = time.time() - start_time
        self._log_final_statistics(result, candidate_points, total_time)
        
        return result
    
    def _extract_adjacency_from_graph(self, points: List[Dict]) -> List[Dict]:
        """
        Extract adjacency data directly from the NetworkX graph created in Step 7b.
        
        This replaces the old compute_adjacency_relationships method by simply
        querying the NetworkX graph that already contains all the adjacency information.
        """
        self.logger.info("Extracting adjacency data from NetworkX graph...")
        
        # Check if we have a NetworkX graph
        if not hasattr(self.adjacency_helper, 'segment_graph') or self.adjacency_helper.segment_graph is None:
            self.logger.error("No NetworkX graph found! Cannot extract adjacency data.")
            return points
        
        graph = self.adjacency_helper.segment_graph
        updated_points = []
        adjacency_stats = {'total_connections': 0, 'points_with_neighbors': 0}
        
        for point in points:
            point_id = point.get('point_id')
            
            # Skip points without valid point_id
            if point_id is None:
                point['adjacent_points'] = ''
                point['adjacent_distances'] = ''
                point['adjacency_count'] = 0
                updated_points.append(point)
                continue
            
            # Get direct neighbors from the NetworkX graph
            if graph.has_node(point_id):
                immediate_neighbors = list(graph.neighbors(point_id))
                
                if immediate_neighbors:
                    # Extract distances from edge data
                    neighbor_distances = []
                    valid_neighbors = []
                    
                    for neighbor_id in immediate_neighbors:
                        edge_data = graph.get_edge_data(point_id, neighbor_id)
                        if edge_data and 'distance' in edge_data:
                            distance = edge_data['distance']
                            neighbor_distances.append(distance)
                            valid_neighbors.append(neighbor_id)
                    
                    # Format adjacency data
                    point['adjacent_points'] = ','.join(map(str, valid_neighbors))
                    point['adjacent_distances'] = ','.join(f"{d:.2f}" for d in neighbor_distances)
                    point['adjacency_count'] = len(valid_neighbors)
                    
                    # Update statistics
                    adjacency_stats['total_connections'] += len(valid_neighbors)
                    if len(valid_neighbors) > 0:
                        adjacency_stats['points_with_neighbors'] += 1
                else:
                    point['adjacent_points'] = ''
                    point['adjacent_distances'] = ''
                    point['adjacency_count'] = 0
            else:
                point['adjacent_points'] = ''
                point['adjacent_distances'] = ''
                point['adjacency_count'] = 0
            
            updated_points.append(point)
        
        # Log final statistics
        self.logger.info(f"Extracted adjacency data for {adjacency_stats['points_with_neighbors']}/{len(updated_points)} points")
        self.logger.info(f"Average connections per point: {adjacency_stats['total_connections']/len(updated_points):.2f}")
        
        return updated_points

    def _validate_and_prepare_segments(self, segments_gdf: gpd.GeoDataFrame, 
                                      min_segment_length: float) -> gpd.GeoDataFrame:
        """Validate and prepare input segments for sampling."""
        self.logger.info(f"Validating {len(segments_gdf)} input segments...")
        
        # Remove invalid geometries
        valid_geoms = segments_gdf[segments_gdf.geometry.is_valid & ~segments_gdf.geometry.is_empty]
        
        # Filter by minimum length
        #long_enough = valid_geoms[valid_geoms.geometry.length >= min_segment_length]
        
        # Add segment metadata
        result = valid_geoms.copy()
        result['segment_id'] = range(len(result))
        result['segment_length'] = result.geometry.length
        result['segment_type'] = result.geometry.geom_type
        
        # Handle MultiLineString by exploding to individual LineStrings
        if (result.geometry.geom_type == 'MultiLineString').any():
            result = result.explode(index_parts=True).reset_index(drop=True)
            result['segment_id'] = range(len(result))
        
        # Validate parent_id field exists
        if 'parent_id' not in result.columns:
            self.logger.error("'parent_id' column not found in segments data")
            raise ValueError("Missing required 'parent_id' column in segments data")
        
        self.logger.info(f"After validation: {len(result)} segments")
        self.logger.info(f"  - Average length: {result['segment_length'].mean():.2f} feet")
        self.logger.info(f"  - Unique parent_ids: {result['parent_id'].nunique()}")
        
        return result
    
    def _create_final_network(self, topology_points: List[Dict], crs) -> gpd.GeoDataFrame:
        """Create the final GeoDataFrame with all point network data."""
        self.logger.info("Creating final point network GeoDataFrame...")
        
        # Prepare data for GeoDataFrame
        final_data = []
        
        for point in topology_points:
            # Use actual segment_id if available, fall back to source_segment_idx
            actual_segment_id = point.get('source_segment_id', point['source_segment_idx'])
            
            # Create simplified record for GeoDataFrame
            record = {
                'geometry': point['geometry'],
                'point_id': point['point_id'],
                'source_segment_idx': actual_segment_id,  # Use consistent segment identifier
                'distance_along_segment': point['distance_along_segment'],
                'segment_total_length': point['segment_total_length'],
                'position_ratio': point['position_ratio'],
                'is_intersection': point.get('is_intersection', False),
                'neighbor_count': len(point.get('network_neighbors', [])),
                'buffer_distance': point['buffer_zone'].area / np.pi ** 0.5 if point.get('buffer_zone') else None,
            }
            
            # Add parent_id directly
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
        summary = self._compute_sampling_summary(topology_points)
        result.attrs['sampling_summary'] = summary
        
        return result
    
    def _compute_sampling_summary(self, points: List[Dict]) -> Dict:
        """Compute summary statistics for the sampling process."""
        if not points:
            return {}
        
        # Extract distances to nearest neighbors
        neighbor_distances = []
        for point in points:
            neighbors = point.get('network_neighbors', [])
            if neighbors:
                neighbor_distances.append(neighbors[0]['distance'])
        
        # Extract segment identifiers using consistent logic
        segment_ids = []
        for pt in points:
            actual_segment_id = pt.get('source_segment_id', pt['source_segment_idx'])
            segment_ids.append(actual_segment_id)
        
        summary = {
            'total_points': len(points),
            'intersection_points': sum(1 for pt in points if pt.get('is_intersection', False)),
            'avg_neighbor_distance': np.mean(neighbor_distances) if neighbor_distances else None,
            'min_neighbor_distance': np.min(neighbor_distances) if neighbor_distances else None,
            'max_neighbor_distance': np.max(neighbor_distances) if neighbor_distances else None,
            'unique_segments': len(set(segment_ids)),
            'avg_points_per_segment': len(points) / len(set(segment_ids)) if segment_ids else 0,
        }
        
        return summary
    
    def _create_empty_result(self, crs) -> gpd.GeoDataFrame:
        """Create an empty result GeoDataFrame with the correct schema."""
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
        """Log final statistics about the sampling process."""
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
            
            if summary.get('avg_neighbor_distance'):
                self.logger.info(f"  - Avg neighbor distance: {summary['avg_neighbor_distance']:.2f} feet")

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
        Master function to process segments into a complete sampled point network.
        
        Parameters:
        -----------
        segments_input : str
            Path to neighborhood segments parquet file
        centerlines_path : str
            Path to citywide centerlines geoparquet file
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
        compute_adjacency : bool
            Whether to compute adjacency relationships
        walkshed_distance : float
            Walkshed radius in feet
        crosswalk_distance : float
            Maximum distance for crosswalk creation
            
        Returns:
        --------
        GeoDataFrame
            Complete sampled point network
        """
        start_time = time.time()
        
        # Step 1: Load and validate input
        self.logger.info("Loading neighborhood segments for point sampling...")
        segments_gdf = gpd.read_parquet(segments_input)
        self.logger.info(f"Loaded {len(segments_gdf)} neighborhood segments")
        
        # Load citywide centerlines for adjacency computation
        citywide_centerlines = self._load_centerlines(centerlines_path)
        if citywide_centerlines is None:
            self.logger.error("Failed to load citywide centerlines")
            return self._create_empty_result(PROJ_FT)
        
        # Load intersection points for centerline filtering
        intersection_points = None
        pedestrian_ramps_points = None
        if pedestrian_ramps_path:
            ramps_gdf = gpd.read_file(pedestrian_ramps_path)
            ramps_gdf = ramps_gdf.to_crs(PROJ_FT)
            
            # Filter ramps to neighborhood area
            filtered_ramps = self.sampling_helper._filter_pedestrian_ramps(ramps_gdf, segments_gdf, buffer_distance)
            
            # Convert to list of dicts for centerline filtering
            intersection_points = []
            for idx, row in filtered_ramps.iterrows():
                intersection_points.append({
                    'geometry': row.geometry,
                    'point_id': f"ramp_{idx}",
                    'is_intersection': True,
                    'is_pedestrian_ramp': True
                })
            
            pedestrian_ramps_points = self.sampling_helper.load_and_process_pedestrian_ramps(
                pedestrian_ramps_path, segments_gdf
            )
        
        # Filter centerlines to neighborhood area
        filtered_centerlines = self._filter_centerlines_to_neighborhood(
            citywide_centerlines, segments_gdf, buffer_distance, 
            intersection_points=intersection_points, crosswalk_distance=crosswalk_distance
        )
        self.logger.info(f"Filtered centerlines: {len(citywide_centerlines)} -> {len(filtered_centerlines)}")
        
        # Save intermediate files if requested
        if save_intermediate and intermediate_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            segments_path = os.path.join(intermediate_dir, f"01_neighborhood_segments_{timestamp}.parquet")
            segments_gdf.to_parquet(segments_path)
            
            centerlines_path = os.path.join(intermediate_dir, f"01_filtered_centerlines_{timestamp}.parquet")
            filtered_centerlines.to_parquet(centerlines_path)
        
        # Step 2: Generate sampled network
        sampled_network = self.generate_sampled_network(
            segments_gdf=segments_gdf,
            buffer_distance=buffer_distance,
            sampling_interval=sampling_interval,
            strategy=strategy,
            pedestrian_ramps_path=pedestrian_ramps_path,
            compute_adjacency=compute_adjacency,
            walkshed_distance=walkshed_distance,
            crosswalk_distance=crosswalk_distance,
            adjacency_centerlines=filtered_centerlines,
            save_intermediate=save_intermediate,
            intermediate_dir=intermediate_dir,
            output_path=output_path
        )
        
        # Step 3: Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        if output_path:
            base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
            extension = output_path.rsplit('.', 1)[1] if '.' in output_path else 'parquet'
            dynamic_output_path = f"{base_path}_sampled_{timestamp}.{extension}"
            
            # Remove complex objects that can't be serialized
            save_network = sampled_network.copy()
            if 'network_neighbors' in save_network.columns:
                save_network = save_network.drop(columns=['network_neighbors'])
            save_network.to_parquet(dynamic_output_path)
            self.logger.info(f"Saved final network to: {dynamic_output_path}")
        
        # Save inspection files if requested
        self._save_inspection_files(sampled_network, filtered_centerlines, 
                                   pedestrian_ramps_points, crosswalk_distance,
                                   compute_adjacency, save_intermediate, 
                                   intermediate_dir, output_path, timestamp)
        
        # Final statistics
        total_time = time.time() - start_time
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Final network: {len(sampled_network)} points")
        
        exit()


    def _load_centerlines(self, centerlines_path: str) -> gpd.GeoDataFrame:
        """Load pre-generated centerlines from geoparquet file."""
        try:
            self.logger.info(f"Loading centerlines from: {centerlines_path}")
            centerlines_gdf = gpd.read_parquet(centerlines_path)
            
            if centerlines_gdf is None or len(centerlines_gdf) == 0:
                self.logger.error("No centerlines found in file")
                return None
            
            # Add required columns for compatibility
            centerlines_gdf['width'] = 10.0
            centerlines_gdf['segment_id'] = range(len(centerlines_gdf))
            centerlines_gdf['segment_length'] = centerlines_gdf.geometry.length
            centerlines_gdf['segment_type'] = centerlines_gdf.geometry.geom_type
            
            # Ensure parent_id exists
            if 'parent_id' not in centerlines_gdf.columns:
                centerlines_gdf['parent_id'] = centerlines_gdf['segment_id'].astype(str)
            
            self.logger.info(f"Successfully loaded {len(centerlines_gdf)} centerlines")
            return centerlines_gdf
            
        except Exception as e:
            self.logger.error(f"Error loading centerlines: {e}")
            return None
    
    def _filter_centerlines_to_neighborhood(self, centerlines_gdf: gpd.GeoDataFrame, 
                                          neighborhood_segments: gpd.GeoDataFrame, 
                                          buffer_distance: float = 100.0, 
                                          intersection_points: Optional[List[Dict]] = None, 
                                          crosswalk_distance: float = 100.0) -> gpd.GeoDataFrame:
        """Filter citywide centerlines to neighborhood area and intersection points."""
        # Create spatial index for neighborhood segments
        segment_tree = STRtree(list(neighborhood_segments.geometry))
        
        # Find centerlines near neighborhood segments
        filtered_centerlines = set()
        
        for idx, centerline_geom in enumerate(centerlines_gdf.geometry):
            nearby_segments = segment_tree.query(centerline_geom.buffer(buffer_distance))
            
            if len(nearby_segments) > 0:
                min_distance = float('inf')
                for segment_idx in nearby_segments:
                    segment_geom = neighborhood_segments.iloc[segment_idx].geometry
                    distance = centerline_geom.distance(segment_geom)
                    min_distance = min(min_distance, distance)
                
                if min_distance <= buffer_distance:
                    filtered_centerlines.add(idx)
        
        # Add centerlines near intersection points
        if intersection_points:
            centerline_tree = STRtree(list(centerlines_gdf.geometry))
            
            for point in intersection_points:
                point_geom = point['geometry']
                nearby_centerlines = centerline_tree.query(point_geom.buffer(crosswalk_distance))
                
                for centerline_idx in nearby_centerlines:
                    centerline_geom = centerlines_gdf.iloc[centerline_idx].geometry
                    distance = point_geom.distance(centerline_geom)
                    
                    if distance <= crosswalk_distance:
                        filtered_centerlines.add(centerline_idx)
        
        # Return filtered GeoDataFrame
        if filtered_centerlines:
            return centerlines_gdf.iloc[list(filtered_centerlines)].reset_index(drop=True)
        else:
            return centerlines_gdf.iloc[:0].copy()

    def _save_network_topology_results(self, topology_enhanced: List[Dict], intermediate_dir: str, timestamp: str):
        """Save the network topology results (points with adjacency info) to a parquet file."""
        if not topology_enhanced:
            self.logger.warning("No topology points to save.")
            return

        # Prepare data for GeoDataFrame
        topology_data = []
        for point in topology_enhanced:
            # Use actual segment_id if available, fall back to source_segment_idx
            actual_segment_id = point.get('source_segment_id', point['source_segment_idx'])
            
            record = {
                'geometry': point['geometry'],
                'point_id': point['point_id'],
                'source_segment_idx': actual_segment_id,  # Use consistent segment identifier
                'distance_along_segment': point['distance_along_segment'],
                'segment_total_length': point['segment_total_length'],
                'position_ratio': point['position_ratio'],
                'is_intersection': point.get('is_intersection', False),
                'neighbor_count': len(point.get('network_neighbors', [])),
                'buffer_distance': point['buffer_zone'].area / np.pi ** 0.5 if point.get('buffer_zone') else None,
            }
            
            # Add parent_id directly
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
            
            topology_data.append(record)

        # Create GeoDataFrame
        topology_gdf = gpd.GeoDataFrame(topology_data, crs=PROJ_FT)
        
        # Save to parquet
        topology_path = os.path.join(intermediate_dir, f"02_network_topology_{timestamp}.parquet")
        topology_gdf.to_parquet(topology_path)
        self.logger.info(f"Saved network topology results to: {topology_path}")

    def _save_segment_network_connectivity(self, combined_segments_for_graph: gpd.GeoDataFrame, 
                                         intermediate_dir: str, timestamp: str):
        """Save segment network connectivity structure for visualization."""
        self.logger.info("Saving segment network connectivity for visualization...")
        
        # Extract segment-to-segment connectivity from the pre-computed NetworkX graph
        segment_connectivity = self.adjacency_helper._extract_segment_connectivity_from_graph(combined_segments_for_graph)
        
        # Extract segment network edges and nodes
        edges_data = []
        nodes_data = []
        
        # Process segment connections (edges)
        for segment_idx, connected_segments in segment_connectivity.items():
            for connected_idx in connected_segments:
                # Avoid duplicate edges (A->B and B->A)
                if segment_idx < connected_idx:
                    # Get the actual segment_ids for consistent indexing
                    source_seg_id = combined_segments_for_graph.iloc[segment_idx].get('segment_id', segment_idx)
                    target_seg_id = combined_segments_for_graph.iloc[connected_idx].get('segment_id', connected_idx)
                    
                    edges_data.append({
                        'source_segment_idx': segment_idx,
                        'target_segment_idx': connected_idx,
                        'source_segment_id': str(source_seg_id),
                        'target_segment_id': str(target_seg_id),
                        'connection_type': 'segment_to_segment'
                    })
        
        # Process segments (nodes)
        for idx, row in combined_segments_for_graph.iterrows():
            # Get segment properties
            segment_id = row.get('segment_id', idx)
            is_crosswalk = row.get('is_crosswalk', False)
            segment_type = 'crosswalk' if is_crosswalk else 'sidewalk'
            
            # Count connections for this segment
            connected_count = len(segment_connectivity.get(idx, []))
            
            nodes_data.append({
                'segment_idx': idx,
                'segment_id': str(segment_id),
                'parent_id': str(row.get('parent_id', '')),
                'segment_type': segment_type,
                'is_crosswalk': is_crosswalk,
                'segment_length': row.geometry.length,
                'geometry': row.geometry,
                'connected_segments': connected_count,
                'connected_segment_list': ','.join(map(str, segment_connectivity.get(idx, [])))
            })
        
        # Save segment network edges to CSV
        if edges_data:
            edges_df = pd.DataFrame(edges_data)
            edges_path = os.path.join(intermediate_dir, f"01_segment_network_edges_{timestamp}.csv")
            edges_df.to_csv(edges_path, index=False)
            self.logger.info(f"Saved segment network edges to: {edges_path}")
        
        # Save segment network nodes to GeoJSON for visualization
        if nodes_data:
            # Create GeoDataFrame for segments
            segments_gdf = gpd.GeoDataFrame(nodes_data, crs=combined_segments_for_graph.crs)
            
            # Save as GeoJSON for easy visualization
            segments_path = os.path.join(intermediate_dir, f"01_segment_network_segments_{timestamp}.geojson")
            segments_gdf.to_file(segments_path, driver='GeoJSON')
            self.logger.info(f"Saved segment network segments to: {segments_path}")
            
            # Also save as parquet for analysis
            segments_parquet_path = os.path.join(intermediate_dir, f"01_segment_network_segments_{timestamp}.parquet")
            segments_gdf.to_parquet(segments_parquet_path)
            self.logger.info(f"Saved segment network segments to: {segments_parquet_path}")
        
        # Save segment network summary statistics
        network_summary = {
            'total_segments': len(nodes_data),
            'total_connections': len(edges_data),
            'crosswalk_segments': sum(1 for node in nodes_data if node.get('is_crosswalk', False)),
            'sidewalk_segments': sum(1 for node in nodes_data if not node.get('is_crosswalk', False)),
            'avg_connections_per_segment': sum(node['connected_segments'] for node in nodes_data) / len(nodes_data) if nodes_data else 0,
            'max_connections_per_segment': max(node['connected_segments'] for node in nodes_data) if nodes_data else 0,
            'network_density': len(edges_data) / (len(nodes_data) * (len(nodes_data) - 1)) if len(nodes_data) > 1 else 0
        }
        
        # Save network summary
        summary_path = os.path.join(intermediate_dir, f"01_segment_network_summary_{timestamp}.json")
        import json
        with open(summary_path, 'w') as f:
            json.dump(network_summary, f, indent=2)
        self.logger.info(f"Saved segment network summary to: {summary_path}")
        
        # Log segment network statistics
        self.logger.info(f"Segment network connectivity summary:")
        self.logger.info(f"  - Total segments: {network_summary['total_segments']}")
        self.logger.info(f"  - Crosswalk segments: {network_summary['crosswalk_segments']}")
        self.logger.info(f"  - Sidewalk segments: {network_summary['sidewalk_segments']}")
        self.logger.info(f"  - Total connections: {network_summary['total_connections']}")
        self.logger.info(f"  - Average connections per segment: {network_summary['avg_connections_per_segment']:.2f}")
        self.logger.info(f"  - Network density: {network_summary['network_density']:.4f}")
        
        # Save a NetworkX graph representation for easier visualization
        self._save_segment_networkx_graph(segment_connectivity, combined_segments_for_graph, 
                                        intermediate_dir, timestamp)
        
        # Save complete segment network as GeoJSON for visualization
        self._save_complete_segment_network(combined_segments_for_graph, intermediate_dir, timestamp)

    def _save_segment_networkx_graph(self, segment_connectivity: Dict, combined_segments_for_graph: gpd.GeoDataFrame,
                                   intermediate_dir: str, timestamp: str):
        """Save the segment network as a NetworkX graph with geometric properties."""
        import networkx as nx
        import pickle
        
        self.logger.info("Creating NetworkX graph representation of segment network...")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (segments) with attributes
        for idx, row in combined_segments_for_graph.iterrows():
            segment_id = str(row.get('segment_id', idx))
            
            G.add_node(segment_id, **{
                'segment_idx': idx,
                'segment_id': segment_id,
                'parent_id': str(row.get('parent_id', '')),
                'segment_type': 'crosswalk' if row.get('is_crosswalk', False) else 'sidewalk',
                'is_crosswalk': row.get('is_crosswalk', False),
                'segment_length': row.geometry.length,
                'geometry': row.geometry,
                'centroid_x': row.geometry.centroid.x,
                'centroid_y': row.geometry.centroid.y
            })
        
        # Add edges (connections) with attributes
        for segment_idx, connected_segments in segment_connectivity.items():
            source_seg_id = str(combined_segments_for_graph.iloc[segment_idx].get('segment_id', segment_idx))
            
            for connected_idx in connected_segments:
                target_seg_id = str(combined_segments_for_graph.iloc[connected_idx].get('segment_id', connected_idx))
                
                # Calculate distance between segment centroids
                source_centroid = combined_segments_for_graph.iloc[segment_idx].geometry.centroid
                target_centroid = combined_segments_for_graph.iloc[connected_idx].geometry.centroid
                centroid_distance = source_centroid.distance(target_centroid)
                
                G.add_edge(source_seg_id, target_seg_id, **{
                    'source_segment_idx': segment_idx,
                    'target_segment_idx': connected_idx,
                    'connection_type': 'segment_to_segment',
                    'centroid_distance': centroid_distance
                })
        
        # Save as pickle file for NetworkX operations
        graph_path = os.path.join(intermediate_dir, f"01_segment_networkx_graph_{timestamp}.pkl")
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)
        self.logger.info(f"Saved NetworkX segment graph to: {graph_path}")
        
        # Save graph info as JSON
        graph_info = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'is_connected': nx.is_connected(G),
            'number_of_components': nx.number_connected_components(G),
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }
        
        if G.number_of_nodes() > 0:
            graph_info['degree_distribution'] = dict(G.degree())
        
        graph_info_path = os.path.join(intermediate_dir, f"01_segment_networkx_info_{timestamp}.json")
        import json
        with open(graph_info_path, 'w') as f:
            json.dump(graph_info, f, indent=2, default=str)
        self.logger.info(f"Saved NetworkX graph info to: {graph_info_path}")
        
        self.logger.info(f"NetworkX segment graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Save a lookup table for easier visualization
        self._save_segment_lookup_table(combined_segments_for_graph, intermediate_dir, timestamp)

    def _save_segment_lookup_table(self, combined_segments_for_graph: gpd.GeoDataFrame, 
                                 intermediate_dir: str, timestamp: str):
        """Save a lookup table mapping segment_idx to segment_id for easier visualization."""
        lookup_data = []
        
        for idx, row in combined_segments_for_graph.iterrows():
            segment_id = row.get('segment_id', idx)
            lookup_data.append({
                'segment_idx': idx,
                'segment_id': str(segment_id),
                'parent_id': str(row.get('parent_id', '')),
                'segment_type': 'crosswalk' if row.get('is_crosswalk', False) else 'sidewalk',
                'is_crosswalk': row.get('is_crosswalk', False),
                'segment_length': row.geometry.length,
                'centroid_x': row.geometry.centroid.x,
                'centroid_y': row.geometry.centroid.y
            })
        
        lookup_df = pd.DataFrame(lookup_data)
        lookup_path = os.path.join(intermediate_dir, f"01_segment_lookup_table_{timestamp}.csv")
        lookup_df.to_csv(lookup_path, index=False)
        self.logger.info(f"Saved segment lookup table to: {lookup_path}")

    def _save_complete_segment_network(self, combined_segments_for_graph: gpd.GeoDataFrame, intermediate_dir: str, timestamp: str):
        """Save the complete segment network (including crosswalks) as GeoJSON for visualization."""
        self.logger.info("Saving complete segment network for visualization...")
        
        # Prepare the segment network data
        network_data = []
        
        for idx, row in combined_segments_for_graph.iterrows():
            # Determine segment type
            segment_type = 'crosswalk' if row.get('is_crosswalk', False) else 'sidewalk'
            
            # Get segment_id and ensure it's a string
            segment_id = row.get('segment_id', idx)
            if segment_id is not None:
                segment_id = str(segment_id)
            else:
                segment_id = str(idx)
            
            # Create network record
            network_record = {
                'geometry': row.geometry,
                'segment_id': segment_id,
                'parent_id': str(row.get('parent_id', '')),
                'segment_type': segment_type,
                'is_crosswalk': row.get('is_crosswalk', False),
                'segment_length': row.get('segment_length', row.geometry.length),
                'segment_index': idx
            }
            
            # Add any additional attributes from the original data
            for col in row.index:
                if col not in ['geometry', 'segment_id', 'parent_id', 'segment_type', 'is_crosswalk', 'segment_length', 'segment_index']:
                    network_record[col] = row[col]
            
            network_data.append(network_record)
        
        # Create GeoDataFrame
        network_gdf = gpd.GeoDataFrame(network_data, crs=combined_segments_for_graph.crs)
        
        # Save as GeoJSON for visualization
        network_path = os.path.join(intermediate_dir, f"01_complete_segment_network_{timestamp}.geojson")
        network_gdf.to_file(network_path, driver='GeoJSON')
        self.logger.info(f"Saved complete segment network to: {network_path}")
        
        # Also save as parquet for analysis
        network_parquet_path = os.path.join(intermediate_dir, f"01_complete_segment_network_{timestamp}.parquet")
        network_gdf.to_parquet(network_parquet_path)
        self.logger.info(f"Saved complete segment network to: {network_parquet_path}")
        
        # Log segment network statistics
        crosswalk_count = sum(1 for record in network_data if record.get('is_crosswalk', False))
        sidewalk_count = len(network_data) - crosswalk_count
        
        self.logger.info(f"Complete segment network statistics:")
        self.logger.info(f"  - Total segments: {len(network_data)}")
        self.logger.info(f"  - Sidewalk segments: {sidewalk_count}")
        self.logger.info(f"  - Crosswalk segments: {crosswalk_count}")
        self.logger.info(f"  - Average segment length: {sum(r['segment_length'] for r in network_data) / len(network_data):.2f} feet")

    def _save_segment_adjacency_analysis(self, topology_enhanced: List[Dict], intermediate_dir: str, timestamp: str):
        """Save detailed segment adjacency relationships from the network topology."""
        if not topology_enhanced:
            self.logger.warning("No topology points to analyze segment adjacencies.")
            return

        # Create a mapping from source_segment_idx to actual segment_id if available
        # This handles cases where source_segment_idx might not match segment_id due to filtering
        segment_id_mapping = {}
        for point in topology_enhanced:
            source_idx = point['source_segment_idx']
            # Check if point has source_segment_id which would be the actual segment_id
            actual_segment_id = point.get('source_segment_id', source_idx)
            segment_id_mapping[source_idx] = actual_segment_id

        # Analyze segment adjacencies from network topology
        segment_adjacencies = defaultdict(set)
        relationship_counts = defaultdict(int)
        
        for point in topology_enhanced:
            point_segment_idx = point['source_segment_idx']
            # Use the mapped segment ID for consistency with saved parquet files
            point_segment = segment_id_mapping.get(point_segment_idx, point_segment_idx)
            
            # Parse adjacency information from NetworkX-based adjacency computation
            adjacent_points_str = point.get('adjacent_points', '')
            adjacent_distances_str = point.get('adjacent_distances', '')
            
            if adjacent_points_str:
                # Parse comma-separated neighbor IDs and distances
                neighbor_ids = [int(x.strip()) for x in adjacent_points_str.split(',') if x.strip()]
                neighbor_distances = [float(x.strip()) for x in adjacent_distances_str.split(',') if x.strip()]
                
                for i, neighbor_point_id in enumerate(neighbor_ids):
                    neighbor_distance = neighbor_distances[i] if i < len(neighbor_distances) else 0.0
                    
                    # Find the neighbor point to get its segment
                    neighbor_point = next((p for p in topology_enhanced if p['point_id'] == neighbor_point_id), None)
                    if neighbor_point:
                        neighbor_segment_idx = neighbor_point['source_segment_idx']
                        # Use the mapped segment ID for the neighbor as well
                        neighbor_segment = segment_id_mapping.get(neighbor_segment_idx, neighbor_segment_idx)
                        
                        # Record segment adjacency using consistent segment IDs
                        segment_adjacencies[point_segment].add(neighbor_segment)
                        
                        # Classify relationship type based on segments
                        if point_segment_idx == neighbor_segment_idx:  # Use original indices for relationship classification
                            relationship_counts['same_segment'] += 1
                        elif point.get('is_intersection', False) or neighbor_point.get('is_intersection', False):
                            relationship_counts['intersection'] += 1
                        else:
                            relationship_counts['cross_segment'] += 1
        
        # Create segment adjacency records using consistent segment IDs
        segment_adjacency_data = []
        for segment_id, adjacent_segments in segment_adjacencies.items():
            segment_adjacency_data.append({
                'segment_id': segment_id,
                'adjacent_segments': ','.join(map(str, sorted(adjacent_segments))),
                'adjacent_segment_count': len(adjacent_segments)
            })
        
        # Save segment adjacency matrix
        if segment_adjacency_data:
            segment_df = pd.DataFrame(segment_adjacency_data)
            segment_adjacency_path = os.path.join(intermediate_dir, f"02_segment_to_segment_adjacencies_{timestamp}.csv")
            segment_df.to_csv(segment_adjacency_path, index=False)
            self.logger.info(f"Saved segment-to-segment adjacencies to: {segment_adjacency_path}")
        
        # Create detailed point-level adjacency data with network neighbors
        point_adjacency_data = []
        for point in topology_enhanced:
            neighbors = point.get('network_neighbors', [])
            point_segment_idx = point['source_segment_idx']
            point_segment = segment_id_mapping.get(point_segment_idx, point_segment_idx)
            
            if neighbors:
                # Create records for each neighbor relationship
                for neighbor in neighbors:
                    # Find the neighbor point
                    neighbor_point = next((p for p in topology_enhanced if p['point_id'] == neighbor['point_id']), None)
                    if neighbor_point:
                        neighbor_segment_idx = neighbor_point['source_segment_idx']
                        neighbor_segment = segment_id_mapping.get(neighbor_segment_idx, neighbor_segment_idx)
                        
                        point_adjacency_data.append({
                            'point_id': point['point_id'],
                            'point_segment': point_segment,  # Use consistent segment ID
                            'point_parent_id': point.get('parent_id', ''),
                            'neighbor_id': neighbor['point_id'],
                            'neighbor_segment': neighbor_segment,  # Use consistent segment ID
                            'neighbor_parent_id': neighbor_point.get('parent_id', ''),
                            'distance': neighbor['distance'],
                            'relationship': neighbor.get('relationship', 'unknown')
                        })
        
        # Save detailed point adjacency relationships
        if point_adjacency_data:
            point_df = pd.DataFrame(point_adjacency_data)
            point_adjacency_path = os.path.join(intermediate_dir, f"02_point_adjacency_relationships_{timestamp}.csv")
            point_df.to_csv(point_adjacency_path, index=False)
            self.logger.info(f"Saved point adjacency relationships to: {point_adjacency_path}")
        
        # Log relationship type summary
        self.logger.info("Network topology relationship summary:")
        for relationship, count in relationship_counts.items():
            self.logger.info(f"  - {relationship}: {count} connections")
        
        self.logger.info(f"Segment adjacency summary:")
        self.logger.info(f"  - {len(segment_adjacencies)} segments have adjacent segments")
        total_adjacencies = sum(len(adj) for adj in segment_adjacencies.values())
        self.logger.info(f"  - {total_adjacencies} total segment adjacency relationships")


    def _save_inspection_files(self, sampled_network: gpd.GeoDataFrame,
                              filtered_centerlines: gpd.GeoDataFrame,
                              pedestrian_ramps_points: List[Dict],
                              crosswalk_distance: float,
                              compute_adjacency: bool,
                              save_intermediate: bool,
                              intermediate_dir: Optional[str],
                              output_path: Optional[str],
                              timestamp: str):
        """Save intermediate files for inspection."""
        if not save_intermediate:
            return
            
        # Determine output directory
        if intermediate_dir:
            output_dir = intermediate_dir
        elif output_path:
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        else:
            output_dir = '.'
        
        # Get combined centerlines from main flow
        main_flow_centerlines = sampled_network.attrs.get('combined_centerlines_with_crosswalks')
        
        if main_flow_centerlines is not None and len(main_flow_centerlines) > 0:
            combined_centerlines = main_flow_centerlines
            if 'is_crosswalk' in combined_centerlines.columns:
                crosswalk_centerlines = combined_centerlines[combined_centerlines['is_crosswalk'] == True].copy()
            else:
                crosswalk_centerlines = gpd.GeoDataFrame(columns=['geometry'])
        elif compute_adjacency and pedestrian_ramps_points:
            # Generate crosswalks for inspection
            crosswalk_centerlines = self.sampling_helper.generate_intersection_centerlines(
                pedestrian_ramps_points, crosswalk_distance
            )
            
            if len(crosswalk_centerlines) > 0:
                combined_centerlines = pd.concat([filtered_centerlines, crosswalk_centerlines], ignore_index=True)
            else:
                combined_centerlines = filtered_centerlines
        else:
            combined_centerlines = filtered_centerlines
            crosswalk_centerlines = gpd.GeoDataFrame(columns=['geometry'])
        
        # Save combined centerlines
        centerlines_inspection_path = os.path.join(output_dir, f"03_combined_centerlines_{timestamp}.parquet")
        combined_centerlines['segment_id'] = combined_centerlines['segment_id'].astype(str)
        combined_centerlines.to_parquet(centerlines_inspection_path)
        
        # Save crosswalk centerlines if they exist
        if len(crosswalk_centerlines) > 0:
            crosswalk_inspection_path = os.path.join(output_dir, f"03_crosswalk_centerlines_{timestamp}.parquet")
            crosswalk_centerlines.to_parquet(crosswalk_inspection_path)
        
        # Save network with centerlines for inspection
        network_with_centerlines_path = os.path.join(output_dir, f"03_network_with_centerlines_{timestamp}.parquet")
        
        # Create combined inspection file
        points_for_inspection = sampled_network.copy()
        points_for_inspection['feature_type'] = 'point'
        points_for_inspection['feature_id'] = points_for_inspection['point_id'].astype(str)
        
        centerlines_for_inspection = combined_centerlines.copy()
        centerlines_for_inspection['feature_type'] = 'centerline'
        centerlines_for_inspection['feature_id'] = centerlines_for_inspection['segment_id'].astype(str)
        
        combined_inspection = pd.concat([points_for_inspection, centerlines_for_inspection], ignore_index=True)
        combined_inspection['parent_id'] = combined_inspection['parent_id'].astype(str)
        combined_inspection.to_parquet(network_with_centerlines_path)


if __name__ == "__main__":
    # Use Google Fire for CLI
    fire.Fire(SampledPointNetwork) 
