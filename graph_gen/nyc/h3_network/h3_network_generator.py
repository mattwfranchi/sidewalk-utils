import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
import h3
from typing import Dict, List, Set, Tuple, Optional, Union
import argparse
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Try to import RAPIDS libraries (optional)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import cudf
    CUDDF_AVAILABLE = True
except ImportError:
    CUDDF_AVAILABLE = False
    cudf = None

try:
    import cugraph
    CUGGRAPH_AVAILABLE = True
except ImportError:
    CUGGRAPH_AVAILABLE = False
    cugraph = None

# Check if H3 is available
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    h3 = None

# Set RAPIDS availability
RAPIDS_AVAILABLE = CUPY_AVAILABLE and CUDDF_AVAILABLE

# Try to import project constants
try:
    from data.nyc.c import PROJ_FT
except ImportError:
    PROJ_FT = 'EPSG:3627'  # Default NYC projection


class H3NetworkGenerator:
    """
    H3-based sidewalk network generator that partitions segment-level networks into point-based networks.
    
    This class creates a spatially uniform, point-based sidewalk network where:
    - Each point represents an H3 hexagonal cell that intersects with sidewalk segments
    - The point is located at the H3 cell centroid
    - The geometry associated with each point is the sidewalk segment(s) clipped to the hexagonal boundary
    - Every H3 cell that overlaps with any part of the sidewalk network gets a point
    
    Key benefits:
    - Spatially uniform resolution (all points represent the same geographic area)
    - Excellent spatial indexing capabilities via H3 indices
    - Complete coverage of sidewalk network
    - Precise hexagonal boundaries for clean clipping
    """
    
    def __init__(self, logger=None, verbose=False):
        self.logger = logger
        self.verbose = verbose
        self._log = self._get_logger()
        
        # Verify H3 availability
        if not H3_AVAILABLE:
            self._log("WARNING: H3 library not available. Please install h3-py: pip install h3")
        else:
            # Test H3 functionality
            self._test_h3_functionality()
        
        # Set up GPU memory pool for better memory management
        try:
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            self._log("GPU memory pool initialized for optimized memory management")
        except Exception as e:
            self._log(f"Could not initialize GPU memory pool: {e}")
    
    def _get_logger(self):
        """Get logger function that handles None logger gracefully"""
        if self.logger:
            return self.logger.info
        
        def log_function(msg):
            if self.verbose or "ERROR" in msg or "WARNING" in msg:
                print(f"[H3NetworkGenerator] {msg}", flush=True)
        
        return log_function
    
    def generate_h3_network(self, segments_gdf: gpd.GeoDataFrame, 
                           h3_resolution: int = 13,
                           memory_optimized: bool = False) -> gpd.GeoDataFrame:
        """
        Generate H3-based point network from sidewalk segments.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input sidewalk segments with LineString geometries
        h3_resolution : int
            H3 resolution (default: 13 for ~36.1m² cells)
        memory_optimized : bool
            Use memory-optimized approaches for large datasets
            
        Returns:
        --------
        GeoDataFrame
            Points representing H3 cells with clipped segment geometries
        """
        if not H3_AVAILABLE:
            self._log("ERROR: H3 library not available. Cannot generate H3 network.")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 
                         'avg_sidewalk_width', 'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        self._log(f"Generating H3-{h3_resolution} network from {len(segments_gdf)} segments")
        
        # Step 1: Convert to WGS84 for H3 operations
        self._log("Step 1: Converting to WGS84 coordinate system for H3 operations")
        segments_wgs84 = segments_gdf.to_crs('EPSG:4326')
        
        # Step 2: Generate H3 cells covering the network extent
        self._log("Step 2: Generating H3 grid covering network extent")
        intersecting_cells = self._find_intersecting_h3_cells(segments_wgs84, h3_resolution)
        
        if not intersecting_cells:
            self._log("No H3 cells intersect with sidewalk segments")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 
                         'avg_sidewalk_width', 'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        # Step 3: Create H3 cell points and clip segments
        self._log("Step 3: Creating H3 cell points and clipping segments")
        h3_points = self._create_h3_points_with_clipped_segments(
            segments_gdf, intersecting_cells, h3_resolution, memory_optimized
        )
        
        # DEBUG: Add comprehensive debugging here
        self._log(f"DEBUG: After _create_h3_points_with_clipped_segments:")
        self._log(f"  - Type of h3_points: {type(h3_points)}")
        self._log(f"  - Length of h3_points: {len(h3_points) if h3_points else 'None/Empty'}")
        
        if h3_points:
            # Check the structure of the first few points
            sample_size = min(3, len(h3_points))
            for i in range(sample_size):
                self._log(f"  - Sample point {i}: keys = {list(h3_points[i].keys()) if isinstance(h3_points[i], dict) else 'Not a dict'}")
                if isinstance(h3_points[i], dict):
                    geometry_val = h3_points[i].get('geometry')
                    self._log(f"    - geometry type: {type(geometry_val)}")
                    self._log(f"    - geometry value: {geometry_val}")
        
        if not h3_points:
            self._log("No H3 points generated")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 
                         'avg_sidewalk_width', 'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        # Step 4: Create final GeoDataFrame with additional safety checks
        self._log("Step 4: Creating final H3-based point network")
        
        # DEBUG: Validate h3_points structure before creating GeoDataFrame
        self._log("DEBUG: Validating h3_points structure before GeoDataFrame creation...")
        
        # Check if all points have geometry
        points_with_geometry = 0
        points_without_geometry = 0
        geometry_types = set()
        
        for i, point in enumerate(h3_points):
            if isinstance(point, dict) and 'geometry' in point:
                geom = point['geometry']
                if geom is not None:
                    points_with_geometry += 1
                    geometry_types.add(str(type(geom)))
                else:
                    points_without_geometry += 1
                    self._log(f"  - Point {i} has None geometry")
            else:
                points_without_geometry += 1
                self._log(f"  - Point {i} missing geometry key or not a dict")
        
        self._log(f"DEBUG: Geometry validation results:")
        self._log(f"  - Points with valid geometry: {points_with_geometry}")
        self._log(f"  - Points without geometry: {points_without_geometry}")
        self._log(f"  - Geometry types found: {geometry_types}")
        
        if points_without_geometry > 0:
            self._log(f"ERROR: Found {points_without_geometry} points without valid geometry. Cannot create GeoDataFrame.")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 
                         'avg_sidewalk_width', 'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        try:
            result = gpd.GeoDataFrame(h3_points, crs=segments_gdf.crs)
            self._log(f"DEBUG: Successfully created GeoDataFrame with {len(result)} rows")
            self._log(f"DEBUG: GeoDataFrame columns: {list(result.columns)}")
            self._log(f"DEBUG: GeoDataFrame CRS: {result.crs}")
        except Exception as e:
            self._log(f"ERROR: Failed to create GeoDataFrame: {e}")
            self._log(f"DEBUG: h3_points sample for troubleshooting:")
            if h3_points:
                import pandas as pd
                try:
                    # Try to create a DataFrame first to see what happens
                    df = pd.DataFrame(h3_points)
                    self._log(f"  - DataFrame columns: {list(df.columns)}")
                    self._log(f"  - DataFrame shape: {df.shape}")
                    self._log(f"  - DataFrame dtypes: {df.dtypes.to_dict()}")
                except Exception as df_e:
                    self._log(f"  - Could not create DataFrame: {df_e}")
            raise e
        
        # Log final statistics
        self._log_final_statistics(result, intersecting_cells)
        
        return result
    
    def _find_intersecting_h3_cells(self, segments_wgs84: gpd.GeoDataFrame, h3_resolution: int) -> Set[str]:
        """
        Find H3 cells that intersect with sidewalk segments.
        
        Parameters:
        -----------
        segments_wgs84 : GeoDataFrame
            Sidewalk segments in WGS84 CRS
        h3_resolution : int
            H3 resolution
            
        Returns:
        --------
        Set[str]
            Set of H3 cell IDs that intersect with segments
        """
        self._log("Finding H3 cells that intersect with sidewalk segments...")
        
        # Use GPU acceleration for very large datasets (per-segment approach is more CPU-friendly)
        if len(segments_wgs84) > 50000 and RAPIDS_AVAILABLE:
            return self._find_intersecting_h3_cells_gpu(segments_wgs84, h3_resolution)
        else:
            return self._find_intersecting_h3_cells_cpu(segments_wgs84, h3_resolution)
    
    def _find_intersecting_h3_cells_gpu(self, segments_wgs84: gpd.GeoDataFrame, 
                                       h3_resolution: int) -> Set[str]:
        """
        GPU-accelerated version using buffered polygon approach with dynamic H3-based buffer sizing.
        """
        self._log("Using GPU-accelerated buffered polygon approach for complete H3 intersection coverage...")
        
        try:
            # Calculate buffer size based on H3 hexagon size (same as CPU version)
            h3_edge_lengths = {
                0: 1107712.591, 1: 418676.005, 2: 158244.655, 3: 59810.857,
                4: 22606.379, 5: 8544.408, 6: 3229.482, 7: 1220.629,
                8: 461.354, 9: 174.375, 10: 65.907, 11: 24.910,
                12: 9.415, 13: 3.559, 14: 1.348, 15: 0.509
            }
            
            edge_length_m = h3_edge_lengths.get(h3_resolution, 3.559)
            max_intersection_distance_m = 2.0 * edge_length_m
            buffer_degrees = max_intersection_distance_m / 111320.0
            
            self._log(f"H3 resolution {h3_resolution}: edge={edge_length_m:.3f}m")
            self._log(f"Using conservative buffer of {max_intersection_distance_m:.3f}m ({buffer_degrees:.8f} degrees)")
            self._log(f"This guarantees ALL intersecting hexagons are captured")
            
            intersecting_cells = set()
            total_segments = len(segments_wgs84)
            
            self._log(f"Processing {total_segments} segments with GPU-accelerated buffered polygon approach...")
            
            # Process segments in batches for GPU efficiency
            batch_size = 250  # Smaller batches for buffered processing
            
            for batch_start in range(0, total_segments, batch_size):
                batch_end = min(batch_start + batch_size, total_segments)
                batch_segments = segments_wgs84.iloc[batch_start:batch_end]
                
                # Progress logging
                batch_num = batch_start // batch_size + 1
                total_batches = (total_segments + batch_size - 1) // batch_size
                if batch_num % 5 == 0 or batch_num == 1 or batch_num == total_batches:
                    self._log(f"Processing GPU batch {batch_num}/{total_batches} ({batch_start + 1}-{batch_end} of {total_segments} segments)")
                
                # Process batch of segments using buffered approach
                batch_cells = set()
                for idx, (_, row) in enumerate(batch_segments.iterrows()):
                    try:
                        geom = row.geometry
                        segment_cells = set()
                        
                        if geom.geom_type == 'LineString':
                            segment_cells = self._get_buffered_h3_cells_for_linestring(
                                geom, h3_resolution, buffer_degrees
                            )
                        elif geom.geom_type == 'MultiLineString':
                            # Handle MultiLineString by processing each component
                            for line_geom in geom.geoms:
                                line_cells = self._get_buffered_h3_cells_for_linestring(
                                    line_geom, h3_resolution, buffer_degrees
                                )
                                segment_cells.update(line_cells)
                                        
                        batch_cells.update(segment_cells)
                                        
                    except Exception as e:
                        continue
                
                # Add batch results to main set
                intersecting_cells.update(batch_cells)
                
                # Memory cleanup for GPU
                if batch_num % 20 == 0:
                    try:
                        import gc
                        gc.collect()
                        if CUPY_AVAILABLE:
                            cp.get_default_memory_pool().free_all_blocks()
                    except:
                        pass
            
            # Now filter out cells that don't actually intersect (post-processing verification)
            self._log(f"GPU processing found {len(intersecting_cells)} candidate H3 cells using conservative buffer")
            self._log("Performing geometric verification to remove false positives...")
            
            verified_cells = self._verify_h3_intersections(segments_wgs84, intersecting_cells, h3_resolution)
            
            self._log(f"After verification: {len(verified_cells)} cells actually intersect with segments")
            self._log(f"Filtered out {len(intersecting_cells) - len(verified_cells)} false positives")
            
            return verified_cells
            
        except Exception as e:
            self._log(f"GPU buffered polygon processing failed, falling back to CPU: {e}")
            return self._find_intersecting_h3_cells_cpu(segments_wgs84, h3_resolution)
    
    def _find_intersecting_h3_cells_cpu(self, segments_wgs84: gpd.GeoDataFrame, h3_resolution: int) -> Set[str]:
        """
        CPU version using buffered polygon approach with dynamic H3-based buffer sizing.
        """
        self._log("Using CPU buffered polygon approach for complete H3 intersection coverage...")
        
        intersecting_cells = set()
        total_segments = len(segments_wgs84)
        
        # Calculate buffer size based on H3 hexagon size at the given resolution
        # H3 edge lengths in meters (approximate)
        h3_edge_lengths = {
            0: 1107712.591, 1: 418676.005, 2: 158244.655, 3: 59810.857,
            4: 22606.379, 5: 8544.408, 6: 3229.482, 7: 1220.629,
            8: 461.354, 9: 174.375, 10: 65.907, 11: 24.910,
            12: 9.415, 13: 3.559, 14: 1.348, 15: 0.509
        }
        
        edge_length_m = h3_edge_lengths.get(h3_resolution, 3.559)  # Default to res 13
        
        # Calculate the MAXIMUM distance from a LineString to hexagon center for ANY intersecting hexagon
        # This occurs when the LineString grazes the hexagon at a vertex
        # Maximum distance = circumradius + half the diagonal across the hexagon
        # For regular hexagon: circumradius = edge_length, diagonal = 2 * edge_length
        # So max_distance = edge_length + edge_length = 2 * edge_length
        max_intersection_distance_m = 2.0 * edge_length_m
        
        # Convert to degrees (approximate for mid-latitudes)
        buffer_degrees = max_intersection_distance_m / 111320.0
        
        self._log(f"H3 resolution {h3_resolution}: edge={edge_length_m:.3f}m")
        self._log(f"Using conservative buffer of {max_intersection_distance_m:.3f}m ({buffer_degrees:.8f} degrees)")
        self._log(f"This guarantees ALL intersecting hexagons are captured")
        self._log(f"Processing {total_segments} segments with buffered polygon approach...")
        
        for idx, (_, row) in enumerate(segments_wgs84.iterrows()):
            # Progress logging every 500 segments
            if idx % 500 == 0:
                self._log(f"Processing segment {idx+1}/{total_segments} ({idx/total_segments*100:.1f}%)")
            
            try:
                geom = row.geometry
                segment_cells = set()
                
                if geom.geom_type == 'LineString':
                    segment_cells = self._get_buffered_h3_cells_for_linestring(
                        geom, h3_resolution, buffer_degrees
                    )
                elif geom.geom_type == 'MultiLineString':
                    # Handle MultiLineString by processing each component
                    for line_geom in geom.geoms:
                        line_cells = self._get_buffered_h3_cells_for_linestring(
                            line_geom, h3_resolution, buffer_degrees
                        )
                        segment_cells.update(line_cells)
                else:
                    self._log(f"Warning: Segment {idx} has unsupported geometry type: {geom.geom_type}")
                    continue
                
                intersecting_cells.update(segment_cells)
            
        except Exception as e:
                self._log(f"Error processing segment {idx}: {e}")
                continue
        
        # Now filter out cells that don't actually intersect (post-processing verification)
        self._log(f"Found {len(intersecting_cells)} candidate H3 cells using conservative buffer")
        self._log("Performing geometric verification to remove false positives...")
        
        verified_cells = self._verify_h3_intersections(segments_wgs84, intersecting_cells, h3_resolution)
        
        self._log(f"After verification: {len(verified_cells)} cells actually intersect with segments")
        self._log(f"Filtered out {len(intersecting_cells) - len(verified_cells)} false positives")
        
        return verified_cells
    
    def _get_buffered_h3_cells_for_linestring(self, linestring, h3_resolution: int, buffer_degrees: float) -> Set[str]:
        """
        Get H3 cells that intersect with a LineString using buffered polygon approach.
        
        Parameters:
        -----------
        linestring : LineString
            Input LineString geometry in WGS84
        h3_resolution : int
            H3 resolution
        buffer_degrees : float
            Buffer size in degrees
            
        Returns:
        --------
        Set[str]
            Set of H3 cell IDs that intersect with the buffered LineString
        """
        try:
            # Create buffer around the LineString
            buffered_geom = linestring.buffer(buffer_degrees)
            
            if buffered_geom.is_empty or not buffered_geom.is_valid:
                return set()
            
            # Convert to GeoJSON format for H3
            if buffered_geom.geom_type == 'Polygon':
                buffered_geojson = {
                    "type": "Polygon",
                    "coordinates": [list(buffered_geom.exterior.coords)]
                }
            elif buffered_geom.geom_type == 'MultiPolygon':
                # Handle case where buffer creates multiple polygons
                all_cells = set()
                for poly in buffered_geom.geoms:
                    poly_geojson = {
                        "type": "Polygon", 
                        "coordinates": [list(poly.exterior.coords)]
                    }
                    try:
                        poly_cells = h3.geo_to_cells(poly_geojson, h3_resolution)
                        all_cells.update(poly_cells)
                    except Exception as e:
                        continue
                return all_cells
            else:
                return set()
            
            # Get H3 cells for the buffered polygon
            cells = h3.geo_to_cells(buffered_geojson, h3_resolution)
            return set(cells)
            
        except Exception as e:
            # Fallback: get cells for LineString endpoints
            try:
                coords = list(linestring.coords)
                fallback_cells = set()
                
                for coord in [coords[0], coords[-1]]:
                    try:
                        cell_id = h3.latlng_to_cell(coord[1], coord[0], h3_resolution)
                        if cell_id:
                            fallback_cells.add(cell_id)
                except Exception as e:
                    continue
            
                return fallback_cells
                
            except Exception as e:
                return set()
    
    def _verify_h3_intersections(self, segments_wgs84: gpd.GeoDataFrame, candidate_cells: Set[str], h3_resolution: int) -> Set[str]:
        """
        Verify that H3 cells actually intersect with segments using exact geometric intersection.
        This removes false positives from the conservative buffering approach.
        
        Parameters:
        -----------
        segments_wgs84 : GeoDataFrame
            Segments in WGS84 CRS
        candidate_cells : Set[str]
            Candidate H3 cell IDs from buffering
        h3_resolution : int
            H3 resolution
            
        Returns:
        --------
        Set[str]
            Verified H3 cell IDs that actually intersect with segments
        """
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree
        
        # Create spatial index for segments
        segment_geometries = list(segments_wgs84.geometry)
        spatial_index = STRtree(segment_geometries)
        
        verified_cells = set()
        total_candidates = len(candidate_cells)
        
        for i, cell_id in enumerate(candidate_cells):
            # Progress logging every 1000 cells
            if i % 1000 == 0:
                self._log(f"Verifying cell {i+1}/{total_candidates} ({i/total_candidates*100:.1f}%)")
            
            try:
                # Get H3 cell boundary as polygon in WGS84
                cell_boundary = h3.cell_to_boundary(cell_id)
                cell_boundary_geojson = [[coord[1], coord[0]] for coord in cell_boundary]
                h3_polygon = Polygon(cell_boundary_geojson)
                
                if not h3_polygon.is_valid:
                    continue
                
                # Use spatial index to find candidate segments
                candidate_segments = spatial_index.query(h3_polygon)
                
                # Check if any segment actually intersects this hexagon
                cell_intersects = False
                for segment_idx in candidate_segments:
                    if segment_idx < len(segments_wgs84):
                        segment_geom = segments_wgs84.geometry.iloc[segment_idx]
                        if h3_polygon.intersects(segment_geom):
                            cell_intersects = True
                            break
                
                if cell_intersects:
                    verified_cells.add(cell_id)
                            
            except Exception as e:
                # Skip cells that cause errors
                continue
        
        return verified_cells
    
    def _create_h3_points_with_clipped_segments(self, segments_gdf: gpd.GeoDataFrame,
                                               intersecting_cells: Set[str],
                                               h3_resolution: int,
                                               memory_optimized: bool) -> List[Dict]:
        """
        Create H3 cell points with clipped segment geometries using optimized approach.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Original sidewalk segments
        intersecting_cells : Set[str]
            Set of H3 cell IDs that intersect with segments
        h3_resolution : int
            H3 resolution
        memory_optimized : bool
            Use memory-optimized approach
            
        Returns:
        --------
        List[Dict]
            List of point records with H3 cell information and clipped segments
        """
        self._log("Creating H3 cell points with clipped segment geometries (optimized)")
        
        # Create spatial index for efficient segment lookup
        self._log("Creating spatial index for segment lookup...")
        from shapely.strtree import STRtree
        segment_geometries = list(segments_gdf.geometry)
        spatial_index = STRtree(segment_geometries)
        
        h3_points = []
        
        # Use larger batch sizes for better performance
        if memory_optimized and len(intersecting_cells) > 5000:
            # Larger batches for memory-optimized mode
            cell_batches = self._batch_cells(intersecting_cells, batch_size=2000)
            self._log(f"Processing {len(intersecting_cells)} cells in {len(cell_batches)} batches of 2000")
        else:
            # Even larger batches for non-memory-optimized mode
            cell_batches = self._batch_cells(intersecting_cells, batch_size=5000)
            self._log(f"Processing {len(intersecting_cells)} cells in {len(cell_batches)} batches of 5000")
        
        # Process batches with progress reporting
        for batch_idx, cell_batch in enumerate(cell_batches):
            if batch_idx % 10 == 0 or batch_idx == len(cell_batches) - 1:
                self._log(f"Processing H3 cell batch {batch_idx + 1}/{len(cell_batches)} ({len(cell_batch)} cells)")
            
            batch_points = self._process_h3_cell_batch_optimized(
                segments_gdf, cell_batch, h3_resolution, spatial_index
            )
            h3_points.extend(batch_points)
            
            # Memory cleanup for large datasets
            if memory_optimized and batch_idx % 50 == 0:
                import gc
                gc.collect()
        
        return h3_points
    
    def _process_h3_cell_batch_optimized(self, segments_gdf: gpd.GeoDataFrame,
                                        cell_batch: Set[str],
                                        h3_resolution: int,
                                        spatial_index: 'STRtree') -> List[Dict]:
        """
        Process a batch of H3 cells to create points with clipped segments (optimized).
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Original sidewalk segments
        cell_batch : Set[str]
            Batch of H3 cell IDs to process
        h3_resolution : int
            H3 resolution
        spatial_index : STRtree
            Spatial index of segment geometries
            
        Returns:
        --------
        List[Dict]
            List of point records for this batch
        """
        # Import required libraries
        from shapely.geometry import Polygon, Point
        
        batch_points = []
        
        for cell_id in cell_batch:
            try:
                # Get H3 cell boundary as polygon
                cell_boundary = h3.cell_to_boundary(cell_id)
                
                # Validate H3 boundary data
                if not cell_boundary or len(cell_boundary) < 3:
                    self._log(f"Invalid H3 boundary data for cell {cell_id}, skipping")
                    continue
                
                cell_boundary_geojson = [[coord[1], coord[0]] for coord in cell_boundary]
                h3_polygon = Polygon(cell_boundary_geojson)
                
                # Validate created polygon
                if not h3_polygon.is_valid:
                    self._log(f"Invalid H3 polygon created for cell {cell_id}, skipping")
                    continue
                
                # Convert to the same CRS as segments for clipping
                h3_polygon_gdf = gpd.GeoDataFrame(
                    geometry=[h3_polygon], 
                    crs='EPSG:4326'
                ).to_crs(segments_gdf.crs)
                h3_polygon = h3_polygon_gdf.geometry.iloc[0]
                
                # Validate H3 polygon geometry
                if h3_polygon is None or not hasattr(h3_polygon, 'is_valid') or not h3_polygon.is_valid:
                    self._log(f"Invalid H3 polygon geometry for cell {cell_id}, skipping")
                    continue
                
                # Get H3 cell centroid
                centroid = h3.cell_to_latlng(cell_id)
                
                # Validate centroid data
                if not centroid or len(centroid) != 2:
                    self._log(f"Invalid H3 centroid data for cell {cell_id}, skipping")
                    continue
                
                # h3 returns (lat, lng), convert to Point(lng, lat) for WGS84
                centroid_point = Point(centroid[1], centroid[0])
                
                # Validate created point
                if not centroid_point.is_valid:
                    self._log(f"Invalid centroid point created for cell {cell_id}, skipping")
                    continue  
                centroid_gdf = gpd.GeoDataFrame(
                    geometry=[centroid_point], 
                    crs='EPSG:4326'
                ).to_crs(segments_gdf.crs)
                centroid_point = centroid_gdf.geometry.iloc[0]
                
                # Validate centroid geometry
                if centroid_point is None or not hasattr(centroid_point, 'is_valid') or not centroid_point.is_valid:
                    self._log(f"Invalid centroid geometry for H3 cell {cell_id}, skipping")
                    continue
                
                # Use spatial index to find intersecting segments efficiently
                intersecting_segments = self._clip_segments_to_h3_cell_optimized(
                    segments_gdf, h3_polygon, cell_id, h3_resolution, spatial_index
                )
                
                if intersecting_segments:
                    # Extract clipped geometries for parquet storage
                    clipped_geometries = []
                    segment_widths = []
                    
                    for seg in intersecting_segments:
                        clipped_geom = seg.get('clipped_geometry')
                        if clipped_geom and not clipped_geom.is_empty:
                            clipped_geometries.append(clipped_geom)
                        
                        # Extract width from original segment attributes
                        if 'original_width' in seg:
                            segment_widths.append(seg['original_width'])
                    
                    # Create MultiLineString from clipped segments for storage
                    if clipped_geometries:
                        if len(clipped_geometries) == 1:
                            clipped_multigeom = clipped_geometries[0]
                        else:
                            from shapely.geometry import MultiLineString
                            clipped_multigeom = MultiLineString(clipped_geometries)
                    else:
                        clipped_multigeom = None
                    
                    # Calculate average width
                    avg_width = sum(segment_widths) / len(segment_widths) if segment_widths else None
                    
                    # Create point record for this H3 cell
                    point_record = {
                        'geometry': centroid_point,  # Keep centroid as main geometry
                        'h3_cell_id': cell_id,
                        'h3_resolution': h3_resolution,
                        'clipped_segments_geometry': clipped_multigeom,  # Store clipped segments for parquet
                        'avg_sidewalk_width': avg_width,  # Store average width
                        'clipped_segments': intersecting_segments,  # Keep detailed info (removed in parquet prep)
                        'segment_count': len(intersecting_segments),
                        'total_clipped_length': sum(
                            seg.get('clipped_length', 0) for seg in intersecting_segments
                        )
                    }
                    batch_points.append(point_record)
                    
            except Exception as e:
                self._log(f"Error processing H3 cell {cell_id}: {e}")
                continue
        
        return batch_points
    
    def _clip_segments_to_h3_cell_optimized(self, segments_gdf: gpd.GeoDataFrame,
                                           h3_polygon: Polygon,
                                           cell_id: str,
                                           h3_resolution: int,
                                           spatial_index: 'STRtree') -> List[Dict]:
        """
        Optimized version of clipping segments to H3 cell using spatial indexing.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Sidewalk segments
        h3_polygon : Polygon
            H3 cell polygon
        cell_id : str
            H3 cell ID
        h3_resolution : int
            H3 resolution
        spatial_index : STRtree
            Spatial index for efficient segment lookup
            
        Returns:
        --------
        List[Dict]
            List of clipped segment dictionaries
        """
        clipped_segments = []
        
        try:
            # Use spatial index to find candidate segments efficiently
            candidate_segments = spatial_index.query(h3_polygon)
            
            for segment_idx in candidate_segments:
                if segment_idx >= len(segments_gdf):
                    continue
                    
                segment_row = segments_gdf.iloc[segment_idx]
                segment_geom = segment_row.geometry
                
                # Check if segment intersects with H3 cell
                if h3_polygon.intersects(segment_geom):
                    try:
                        # Clip segment to H3 cell boundary
                        clipped_geom = h3_polygon.intersection(segment_geom)
                        
                        if not clipped_geom.is_empty:
                            # Calculate clipped length
                            clipped_length = clipped_geom.length
                            
                            # Create clipped segment record
                            clipped_segment = {
                                'original_segment_id': segment_idx,
                                'h3_cell_id': cell_id,
                                'h3_resolution': h3_resolution,
                                'clipped_geometry': clipped_geom,
                                'clipped_length': clipped_length,
                                'original_length': segment_geom.length,
                                'clipping_ratio': clipped_length / segment_geom.length if segment_geom.length > 0 else 0
                            }
                            
                            # Add original segment attributes
                            for col in segments_gdf.columns:
                                if col != 'geometry':
                                    clipped_segment[f'original_{col}'] = segment_row[col]
                            
                            clipped_segments.append(clipped_segment)
                            
                    except Exception as e:
                        self._log(f"Error clipping segment {segment_idx} to cell {cell_id}: {e}")
                        continue
            
            return clipped_segments
            
        except Exception as e:
            self._log(f"Error in optimized segment clipping for cell {cell_id}: {e}")
            return []
    
    def _batch_cells(self, cells: Set[str], batch_size: int = 1000) -> List[Set[str]]:
        """
        Split cells into batches for memory-optimized processing.
        
        Parameters:
        -----------
        cells : Set[str]
            Set of H3 cell IDs
        batch_size : int
            Size of each batch
            
        Returns:
        --------
        List[Set[str]]
            List of cell batches
        """
        cell_list = list(cells)
        batches = []
        
        for i in range(0, len(cell_list), batch_size):
            batch = set(cell_list[i:i + batch_size])
            batches.append(batch)
        
        return batches
    
    def _log_final_statistics(self, result: gpd.GeoDataFrame, intersecting_cells: Set[str]):
        """
        Log final statistics about the generated H3 network.
        
        Parameters:
        -----------
        result : GeoDataFrame
            Final H3-based point network
        intersecting_cells : Set[str]
            Set of intersecting H3 cell IDs
        """
        self._log(f"H3 network generation complete: {len(result)} points representing H3 cells")
        
        # Calculate statistics
        if len(result) > 0:
            # Segment statistics
            segment_counts = result['segment_count'].values
            total_clipped_segments = sum(segment_counts)
            avg_segments_per_cell = total_clipped_segments / len(result)
            
            # Length statistics
            total_clipped_length = result['total_clipped_length'].sum()
            
            # Width statistics  
            width_values = result['avg_sidewalk_width'].dropna()
            if len(width_values) > 0:
                avg_sidewalk_width = width_values.mean()
                min_sidewalk_width = width_values.min()
                max_sidewalk_width = width_values.max()
            else:
                avg_sidewalk_width = None
                min_sidewalk_width = None
                max_sidewalk_width = None
            
            # Geometry statistics
            has_clipped_geom = result['clipped_segments_geometry'].notna().sum()
            
            self._log(f"Final Statistics:")
            self._log(f"  - H3 points generated: {len(result)}")
            self._log(f"  - Total clipped segments: {total_clipped_segments}")
            self._log(f"  - Average segments per cell: {avg_segments_per_cell:.2f}")
            self._log(f"  - Total clipped length: {total_clipped_length:.2f} feet")
            self._log(f"  - H3 cell coverage: {len(intersecting_cells)} cells")
            self._log(f"  - Points with clipped geometries: {has_clipped_geom}")
            
            # Sidewalk width statistics
            if avg_sidewalk_width is not None:
                self._log(f"  - Average sidewalk width: {avg_sidewalk_width:.2f} feet")
                self._log(f"  - Min sidewalk width: {min_sidewalk_width:.2f} feet") 
                self._log(f"  - Max sidewalk width: {max_sidewalk_width:.2f} feet")
                self._log(f"  - Points with width data: {len(width_values)}/{len(result)}")
            else:
                self._log(f"  - No sidewalk width data available")
            
            # Distribution statistics
            self._log(f"  - Min segments per cell: {min(segment_counts)}")
            self._log(f"  - Max segments per cell: {max(segment_counts)}")
            self._log(f"  - Median segments per cell: {np.median(segment_counts):.1f}")
        else:
            self._log(f"Final Statistics: No H3 points generated")
    
    def compute_h3_adjacency(self, h3_points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute adjacency relationships between H3 cells using H3's built-in adjacency (optimized).
        
        Parameters:
        -----------
        h3_points_gdf : GeoDataFrame
            H3-based point network
            
        Returns:
        --------
        GeoDataFrame
            H3 points with adjacency information
        """
        if not H3_AVAILABLE:
            self._log("ERROR: H3 library not available. Cannot compute H3 adjacency.")
            return h3_points_gdf
        
        self._log("Computing H3 cell adjacency relationships (optimized)")
        
        # Get H3 resolution from the first point
        if len(h3_points_gdf) == 0:
            return h3_points_gdf
        
        h3_resolution = h3_points_gdf.iloc[0]['h3_resolution']
        
        # Use GPU acceleration if available and dataset is large
        if RAPIDS_AVAILABLE and len(h3_points_gdf) > 10000:
            self._log("Using GPU acceleration for adjacency computation")
            return self._compute_h3_adjacency_gpu(h3_points_gdf, h3_resolution)
        else:
            return self._compute_h3_adjacency_cpu(h3_points_gdf, h3_resolution)
    
    def _process_adjacency_batch_optimized(self, h3_points_gdf: gpd.GeoDataFrame,
                                    batch_indices: pd.Index,
                                    h3_cell_to_index: Dict[str, int],
                                    h3_resolution: int) -> Dict:
        """
        Process adjacency batch using optimized Python operations.
        
        Parameters:
        -----------
        h3_points_gdf : GeoDataFrame
            H3 points dataframe
        batch_indices : pd.Index
            Batch of indices to process
        h3_cell_to_index : Dict[str, int]
            Mapping from H3 cell IDs to indices for efficient lookup
        h3_resolution : int
            H3 resolution
            
        Returns:
        --------
        Dict
            Adjacency dictionary for this batch (maps index to list of H3 cell IDs)
        """
        batch_adjacency = {}
        
        try:
            # Process each cell in the batch
            for idx in batch_indices:
                cell_id = h3_points_gdf.loc[idx, 'h3_cell_id']
                
                # Get neighboring H3 cells using H3 library (without any extra parameters)
                try:
                    neighbors = h3.grid_disk(cell_id, 1)
                    neighbors.remove(cell_id)  # Remove self
                    
                    if not neighbors:
                        batch_adjacency[idx] = []
                        continue
                except Exception as e:
                    self._log(f"Error getting neighbors for H3 cell {cell_id}: {e}")
                    batch_adjacency[idx] = []
                    continue
                
                # Store H3 cell IDs of neighbors that exist in our dataset
                adjacent_cell_ids = []
                for neighbor_id in neighbors:
                    if neighbor_id in h3_cell_to_index:
                        adjacent_cell_ids.append(neighbor_id)  # Store H3 cell ID, not index
                
                batch_adjacency[idx] = adjacent_cell_ids
            
            return batch_adjacency
            
        except Exception as e:
            self._log(f"Optimized adjacency batch processing failed: {e}")
            # Fallback to CPU processing
            for idx in batch_indices:
                try:
                    row = h3_points_gdf.loc[idx]
                    cell_id = row['h3_cell_id']
                    
                    # Get neighboring H3 cells (without any extra parameters)
                    try:
                        neighbors = h3.grid_disk(cell_id, 1)
                        neighbors.remove(cell_id)  # Remove self
                        
                        # Find which neighbors are in our dataset and store their H3 cell IDs
                        adjacent_cell_ids = []
                        for neighbor_id in neighbors:
                            neighbor_mask = h3_points_gdf['h3_cell_id'] == neighbor_id
                            if neighbor_mask.any():
                                adjacent_cell_ids.append(neighbor_id)  # Store H3 cell ID, not index
                        
                        batch_adjacency[idx] = adjacent_cell_ids
                    except Exception as e:
                        self._log(f"Error getting neighbors for H3 cell {cell_id}: {e}")
                        batch_adjacency[idx] = []
                    
                except Exception as e:
                    batch_adjacency[idx] = []
            
            return batch_adjacency
    
    def _compute_h3_adjacency_cpu(self, h3_points_gdf: gpd.GeoDataFrame, h3_resolution: int) -> gpd.GeoDataFrame:
        """
        CPU-optimized adjacency computation with larger batch sizes.
        """
        # Create adjacency dictionary
        adjacency_dict = {}
        
        # Process in larger batches for better performance
        batch_size = 1000
        total_points = len(h3_points_gdf)
        
        self._log(f"Computing adjacency for {total_points} points in batches of {batch_size}")
        
        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            batch_indices = h3_points_gdf.index[batch_start:batch_end]
            
            if batch_start % 5000 == 0:
                self._log(f"Processing adjacency batch {batch_start//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}")
            
            # Process batch
            for idx in batch_indices:
                row = h3_points_gdf.loc[idx]
                cell_id = row['h3_cell_id']
                
                # Get neighboring H3 cells (without any extra parameters)
                try:
                    neighbors = h3.grid_disk(cell_id, 1)
                    neighbors.remove(cell_id)  # Remove self
                    
                    # Find which neighbors are in our dataset and store their H3 cell IDs
                    adjacent_cell_ids = []
                    for neighbor_id in neighbors:
                        # Find if this neighbor exists in our dataset
                        neighbor_mask = h3_points_gdf['h3_cell_id'] == neighbor_id
                        if neighbor_mask.any():
                            adjacent_cell_ids.append(neighbor_id)  # Store H3 cell ID, not index
                    
                    adjacency_dict[idx] = adjacent_cell_ids
                except Exception as e:
                    self._log(f"Error getting neighbors for H3 cell {cell_id}: {e}")
                    adjacency_dict[idx] = []
        
        # Add adjacency information to the dataframe
        result = h3_points_gdf.copy()
        result['h3_adjacent_cells'] = result.index.map(adjacency_dict)
        result['h3_adjacency_count'] = result['h3_adjacent_cells'].apply(len)
        
        # Log adjacency statistics
        self._log_adjacency_statistics(result)
        
        return result
    
    def _compute_h3_adjacency_gpu(self, h3_points_gdf: gpd.GeoDataFrame, h3_resolution: int) -> gpd.GeoDataFrame:
        """
        GPU-accelerated adjacency computation using optimized Python operations.
        """
        try:
            self._log("Using optimized batch processing for adjacency computation...")
            
            # Create adjacency dictionary
            adjacency_dict = {}
            
            # Process in larger batches for better performance
            batch_size = 2000
            total_points = len(h3_points_gdf)
            
            self._log(f"Computing adjacency for {total_points} points in batches of {batch_size}")
            
            # Create a mapping from H3 cell IDs to indices for efficient lookup
            h3_cell_to_index = {cell_id: idx for idx, cell_id in enumerate(h3_points_gdf['h3_cell_id'])}
            
            for batch_start in range(0, total_points, batch_size):
                batch_end = min(batch_start + batch_size, total_points)
                batch_indices = h3_points_gdf.index[batch_start:batch_end]
                
                if batch_start % 10000 == 0:
                    self._log(f"Processing adjacency batch {batch_start//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}")
                
                # Process batch efficiently
                batch_adjacency = self._process_adjacency_batch_optimized(
                    h3_points_gdf, batch_indices, h3_cell_to_index, h3_resolution
                )
                adjacency_dict.update(batch_adjacency)
            
            # Add adjacency information to the dataframe
            result = h3_points_gdf.copy()
            result['h3_adjacent_cells'] = result.index.map(adjacency_dict)
            result['h3_adjacency_count'] = result['h3_adjacent_cells'].apply(len)
            
            # Log adjacency statistics
            self._log_adjacency_statistics(result)
            
            return result
            
        except Exception as e:
            self._log(f"Optimized adjacency computation failed, falling back to CPU: {e}")
            return self._compute_h3_adjacency_cpu(h3_points_gdf, h3_resolution)
    

    
    def create_h3_graph(self, h3_points_gdf: gpd.GeoDataFrame) -> 'cugraph.Graph':
        """
        Create a cuGraph graph from H3 points for advanced graph analytics.
        
        Parameters:
        -----------
        h3_points_gdf : GeoDataFrame
            H3 points with adjacency information
            
        Returns:
        --------
        cugraph.Graph
            cuGraph graph object for advanced analytics
        """
        if not CUGGRAPH_AVAILABLE:
            self._log("ERROR: cuGraph not available. Cannot create H3 graph.")
            return None
        
        try:
            self._log("Creating cuGraph graph from H3 points...")
            
            # Create mapping from H3 cell IDs to numerical indices for the graph
            h3_cell_ids = h3_points_gdf['h3_cell_id'].tolist()
            cell_id_to_graph_idx = {cell_id: i for i, cell_id in enumerate(h3_cell_ids)}
            
            # Extract edges from adjacency information
            edges = []
            for idx, row in h3_points_gdf.iterrows():
                source_cell_id = row['h3_cell_id']
                source_graph_idx = cell_id_to_graph_idx[source_cell_id]
                
                adjacent_cell_ids = row.get('h3_adjacent_cells', [])
                
                for target_cell_id in adjacent_cell_ids:
                    if target_cell_id in cell_id_to_graph_idx:
                        target_graph_idx = cell_id_to_graph_idx[target_cell_id]
                        edges.append((source_graph_idx, target_graph_idx))
            
            if not edges:
                self._log("No edges found in H3 network")
                return None
            
            # Convert to cuDF for cuGraph
            edge_df = cudf.DataFrame(edges, columns=['src', 'dst'])
            
            # Create cuGraph graph with optimal settings (remove store_transposed parameter)
            G = cugraph.Graph(directed=False)
            G.from_cudf_edgelist(edge_df, source='src', destination='dst')
            
            self._log(f"Created cuGraph graph with {G.number_of_vertices()} vertices and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            self._log(f"Failed to create cuGraph graph: {e}")
            return None
    
    def analyze_h3_graph(self, G: 'cugraph.Graph') -> Dict:
        """
        Perform advanced graph analytics on H3 network using cuGraph.
        
        Parameters:
        -----------
        G : cugraph.Graph
            cuGraph graph object
            
        Returns:
        --------
        Dict
            Graph analytics results
        """
        if G is None:
            return {}
        
        try:
            self._log("Performing advanced graph analytics with cuGraph...")
            
            analytics = {}
            
            # Basic graph statistics
            analytics['num_vertices'] = G.number_of_vertices()
            analytics['num_edges'] = G.number_of_edges()
            
            # Avoid division by zero for density calculation
            if analytics['num_vertices'] > 1:
                max_possible_edges = analytics['num_vertices'] * (analytics['num_vertices'] - 1) / 2
                analytics['density'] = analytics['num_edges'] / max_possible_edges if max_possible_edges > 0 else 0
            else:
                analytics['density'] = 0
            
            # Connected components analysis
            try:
                components = cugraph.connected_components(G)
                analytics['num_components'] = components['labels'].nunique()
                analytics['largest_component_size'] = components['labels'].value_counts().max()
            except Exception as e:
                self._log(f"Connected components analysis failed: {e}")
                analytics['num_components'] = 1
                analytics['largest_component_size'] = analytics['num_vertices']
            
            # PageRank for centrality analysis
            try:
                pagerank = cugraph.pagerank(G)
                analytics['avg_pagerank'] = pagerank['pagerank'].mean()
                analytics['max_pagerank'] = pagerank['pagerank'].max()
            except Exception as e:
                self._log(f"PageRank analysis failed: {e}")
                analytics['avg_pagerank'] = 1.0 / analytics['num_vertices'] if analytics['num_vertices'] > 0 else 0
                analytics['max_pagerank'] = analytics['avg_pagerank']
            
            # Degree distribution using in_degree + out_degree for undirected graphs
            try:
                # For undirected graphs, we can use the edge list to calculate degrees
                edge_df = G.view_edge_list()
                
                # Count degrees by combining source and destination
                src_counts = edge_df['src'].value_counts()
                dst_counts = edge_df['dst'].value_counts()
                
                # Combine and sum (since it's undirected, each edge contributes to both vertices)
                all_vertices = cudf.concat([src_counts, dst_counts]).groupby(level=0).sum()
                
                analytics['avg_degree'] = all_vertices.mean()
                analytics['max_degree'] = all_vertices.max()
                analytics['min_degree'] = all_vertices.min()
            except Exception as e:
                self._log(f"Degree analysis failed: {e}")
                # Fallback estimation
                if analytics['num_vertices'] > 0:
                    avg_degree = (2 * analytics['num_edges']) / analytics['num_vertices']
                    analytics['avg_degree'] = avg_degree
                    analytics['max_degree'] = avg_degree * 2  # rough estimate
                    analytics['min_degree'] = max(0, avg_degree // 2)  # rough estimate
                else:
                    analytics['avg_degree'] = 0
                    analytics['max_degree'] = 0
                    analytics['min_degree'] = 0
            
            self._log(f"Graph analytics complete:")
            self._log(f"  - Vertices: {analytics['num_vertices']}")
            self._log(f"  - Edges: {analytics['num_edges']}")
            self._log(f"  - Components: {analytics['num_components']}")
            self._log(f"  - Average degree: {analytics['avg_degree']:.2f}")
            self._log(f"  - Density: {analytics['density']:.4f}")
            
            return analytics
            
        except Exception as e:
            self._log(f"Graph analytics failed: {e}")
            return {}
    
    def _log_adjacency_statistics(self, result: gpd.GeoDataFrame):
        """
        Log adjacency statistics for the result.
        """
        avg_adjacency = result['h3_adjacency_count'].mean()
        max_adjacency = result['h3_adjacency_count'].max()
        isolated_count = (result['h3_adjacency_count'] == 0).sum()
        
        self._log(f"H3 adjacency statistics:")
        self._log(f"  - Average adjacent cells: {avg_adjacency:.2f}")
        self._log(f"  - Maximum adjacent cells: {max_adjacency}")
        self._log(f"  - Isolated cells: {isolated_count} ({isolated_count/len(result)*100:.1f}%)")
    
    def process_segments_to_h3_network(self, segments_input: Union[gpd.GeoDataFrame, str],
                                      h3_resolution: int = 9,
                                      output_path: Optional[str] = None,
                                      save_intermediate: bool = False,
                                      intermediate_dir: Optional[str] = None,
                                      use_gpu: bool = True,
                                      enable_graph_analytics: bool = True) -> gpd.GeoDataFrame:
        """
        Master function to process input segments all the way to a fully processed H3 point network with adjacency.
        
        Parameters:
        -----------
        segments_input : Union[gpd.GeoDataFrame, str]
            Input segments as GeoDataFrame or path to GeoParquet file
        h3_resolution : int, default=9
            H3 resolution for network generation
        output_path : Optional[str], default=None
            Path to save the final H3 network as GeoParquet
        save_intermediate : bool, default=False
            Whether to save intermediate results
        intermediate_dir : Optional[str], default=None
            Directory to save intermediate results
        use_gpu : bool, default=True
            Whether to use GPU acceleration
        enable_graph_analytics : bool, default=True
            Whether to perform advanced graph analytics with cuGraph
            
        Returns:
        --------
        gpd.GeoDataFrame
            Fully processed H3 network with adjacency information
        """
        start_time = time.time()
        self._log("=" * 60)
        self._log("STARTING H3 NETWORK GENERATION PIPELINE")
        self._log("=" * 60)
        
        # Step 1: Load and validate input
        self._log("\nSTEP 1: Loading and validating input segments...")
        segments_gdf = self._load_segments(segments_input)
        self._validate_segments(segments_gdf)
        
        if save_intermediate and intermediate_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            intermediate_path = os.path.join(intermediate_dir, f"01_input_segments_res{h3_resolution}_{timestamp}.parquet")
            segments_gdf.to_parquet(intermediate_path)
            self._log(f"Saved input segments to: {intermediate_path}")
        
        # Step 2: Generate H3 cells
        self._log("\nSTEP 2: Generating H3 cells from segments...")
        h3_cells = self._generate_h3_cells(segments_gdf, h3_resolution, use_gpu=use_gpu)
        
        if save_intermediate and intermediate_dir:
            intermediate_path = os.path.join(intermediate_dir, f"02_h3_cells_res{h3_resolution}_{timestamp}.parquet")
            # Create DataFrame with H3 cell IDs only (no geometry needed for intermediate storage)
            h3_cells_df = pd.DataFrame({'h3_cell_id': list(h3_cells)})
            h3_cells_df.to_parquet(intermediate_path)
            self._log(f"Saved H3 cells to: {intermediate_path}")
        
        # Step 3: Convert to H3 points
        self._log("\nSTEP 3: Converting H3 cells to points...")
        try:
            self._log(f"DEBUG: About to convert {len(h3_cells)} H3 cells to points")
            h3_points_gdf = self._convert_h3_cells_to_points(h3_cells, h3_resolution, segments_gdf)
            
            # Validate the result
            if h3_points_gdf is None:
                self._log("ERROR: _convert_h3_cells_to_points returned None")
                raise ValueError("H3 points conversion failed: returned None")
            
            self._log(f"DEBUG: Successfully converted H3 cells to points")
            self._log(f"DEBUG: h3_points_gdf type: {type(h3_points_gdf)}")
            self._log(f"DEBUG: h3_points_gdf shape: {h3_points_gdf.shape if hasattr(h3_points_gdf, 'shape') else 'N/A'}")
            
            if hasattr(h3_points_gdf, 'columns'):
                self._log(f"DEBUG: h3_points_gdf columns: {list(h3_points_gdf.columns)}")
                
                # Check if geometry column exists
                if 'geometry' not in h3_points_gdf.columns:
                    self._log("ERROR: No geometry column in h3_points_gdf")
                    raise ValueError("H3 points conversion failed: no geometry column")
                
                # Check if geometry column has valid data
                if h3_points_gdf['geometry'].isna().all():
                    self._log("ERROR: All geometry values are NaN in h3_points_gdf")
                    raise ValueError("H3 points conversion failed: all geometries are NaN")
                
                non_null_geoms = h3_points_gdf['geometry'].notna().sum()
                self._log(f"DEBUG: Non-null geometries: {non_null_geoms}/{len(h3_points_gdf)}")
            
            if hasattr(h3_points_gdf, 'crs'):
                self._log(f"DEBUG: h3_points_gdf CRS: {h3_points_gdf.crs}")
        
        except Exception as e:
            self._log(f"ERROR: Step 3 failed: {e}")
            import traceback
            self._log(f"DEBUG: Step 3 traceback: {traceback.format_exc()}")
            raise e
        
        if save_intermediate and intermediate_dir:
            intermediate_path = os.path.join(intermediate_dir, f"03_h3_points_res{h3_resolution}_{timestamp}.parquet")
            # Prepare for parquet saving
            save_points_gdf = self._prepare_for_parquet_saving(h3_points_gdf)
            save_points_gdf.to_parquet(intermediate_path)
            self._log(f"Saved H3 points to: {intermediate_path}")
        
        # Step 4: Compute adjacency
        self._log("\nSTEP 4: Computing H3 adjacency...")
        h3_network_gdf = self.compute_h3_adjacency(h3_points_gdf)
        
        if save_intermediate and intermediate_dir:
            intermediate_path = os.path.join(intermediate_dir, f"04_h3_network_with_adjacency_res{h3_resolution}_{timestamp}.parquet")
            # Prepare for parquet saving
            save_network_gdf = self._prepare_for_parquet_saving(h3_network_gdf)
            save_network_gdf.to_parquet(intermediate_path)
            self._log(f"Saved H3 network with adjacency to: {intermediate_path}")
        
        # Step 5: Advanced graph analytics (optional)
        if enable_graph_analytics and CUGGRAPH_AVAILABLE:
            self._log("\nSTEP 5: Performing advanced graph analytics with cuGraph...")
            try:
                # Create cuGraph graph
                G = self.create_h3_graph(h3_network_gdf)
                
                if G is not None:
                    # Perform graph analytics
                    analytics = self.analyze_h3_graph(G)
                    
                    # Add analytics summary to the result
                    if analytics:
                        h3_network_gdf.attrs['graph_analytics'] = analytics
                        self._log("Graph analytics completed and attached to result")
                else:
                    self._log("Could not create cuGraph graph for analytics")
                    
            except Exception as e:
                self._log(f"Graph analytics failed: {e}")
        
        # Step 6: Generate dynamic filename and save final result
        final_time = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(final_time))
        
        if output_path:
            # If output path provided, enhance it with resolution and timestamp
            base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
            extension = output_path.rsplit('.', 1)[1] if '.' in output_path else 'parquet'
            dynamic_output_path = f"{base_path}_res{h3_resolution}_{timestamp}.{extension}"
        else:
            # Generate completely dynamic filename
            dynamic_output_path = f"h3_network_res{h3_resolution}_{timestamp}.parquet"
        
        self._log(f"\nSTEP 6: Saving final H3 network to: {dynamic_output_path}")
        save_gdf = self._prepare_for_parquet_saving(h3_network_gdf)
        save_gdf.to_parquet(dynamic_output_path)
        self._log("Final H3 network saved successfully")
        
        # Update the final result to include the actual output path used
        h3_network_gdf.attrs['output_path'] = dynamic_output_path
        
        # Final statistics
        total_time = time.time() - start_time
        self._log("\n" + "=" * 60)
        self._log("H3 NETWORK GENERATION COMPLETED")
        self._log("=" * 60)
        self._log(f"Total processing time: {total_time:.2f} seconds")
        self._log(f"Input segments: {len(segments_gdf)}")
        self._log(f"Generated H3 cells: {len(h3_cells)}")
        self._log(f"Final H3 points: {len(h3_network_gdf)}")
        self._log(f"H3 resolution: {h3_resolution}")
        self._log(f"Average adjacency: {h3_network_gdf['h3_adjacency_count'].mean():.2f}")
        if output_path or 'output_path' in h3_network_gdf.attrs:
            actual_output = h3_network_gdf.attrs.get('output_path', dynamic_output_path)
            self._log(f"Output saved to: {actual_output}")
        
        return h3_network_gdf
    
    def _load_segments(self, segments_input: Union[gpd.GeoDataFrame, str]) -> Optional[gpd.GeoDataFrame]:
        """
        Load segment network from file path or return existing GeoDataFrame.
        
        Parameters:
        -----------
        segments_input : Union[GeoDataFrame, str]
            Input as GeoDataFrame or file path string
            
        Returns:
        --------
        Optional[GeoDataFrame]
            Loaded segment network or None if loading failed
        """
        # If already a GeoDataFrame, return as is
        if isinstance(segments_input, gpd.GeoDataFrame):
            self._log(f"Using provided GeoDataFrame with {len(segments_input)} segments")
            return segments_input
        
        # If string, treat as file path
        if isinstance(segments_input, str):
            file_path = segments_input
            self._log(f"Loading segment network from file: {file_path}")
            
            try:
                # Determine file format and load accordingly
                if file_path.lower().endswith('.parquet'):
                    # Load as GeoParquet
                    self._log("Loading as GeoParquet file")
                    segments_gdf = gpd.read_parquet(file_path)
                elif file_path.lower().endswith('.gpkg'):
                    # Load as GeoPackage
                    self._log("Loading as GeoPackage file")
                    segments_gdf = gpd.read_file(file_path)
                elif file_path.lower().endswith(('.shp', '.geojson', '.json')):
                    # Load as shapefile or GeoJSON
                    self._log(f"Loading as {file_path.split('.')[-1].upper()} file")
                    segments_gdf = gpd.read_file(file_path)
                else:
                    self._log(f"ERROR: Unsupported file format: {file_path}")
                    self._log("Supported formats: .parquet (GeoParquet), .gpkg, .shp, .geojson, .json")
                    return None
                
                self._log(f"✓ Successfully loaded {len(segments_gdf)} segments from {file_path}")
                return segments_gdf
                
            except Exception as e:
                self._log(f"ERROR: Failed to load file {file_path}: {e}")
                return None
        
        # Invalid input type
        self._log(f"ERROR: Invalid input type {type(segments_input)}. Expected GeoDataFrame or file path string.")
        return None
    
    def _validate_segments(self, segments_gdf: gpd.GeoDataFrame):
        """
        Validate input segment network for H3 processing.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input sidewalk segments
        """
        try:
            # Check if GeoDataFrame is empty
            if len(segments_gdf) == 0:
                self._log("ERROR: Input GeoDataFrame is empty")
                return
            
            # Check if geometry column exists
            if 'geometry' not in segments_gdf.columns:
                self._log("ERROR: No geometry column found in GeoDataFrame")
                return
            
            # Check if CRS is set
            if segments_gdf.crs is None:
                self._log("ERROR: GeoDataFrame has no CRS defined")
                return
            
            # Check geometry types
            geom_types = segments_gdf.geometry.geom_type.unique()
            if not all(geom_type in ['LineString', 'MultiLineString'] for geom_type in geom_types):
                invalid_types = [t for t in geom_types if t not in ['LineString', 'MultiLineString']]
                self._log(f"ERROR: Invalid geometry types found: {invalid_types}. Only LineString and MultiLineString are supported")
                return
            
            # Check for valid geometries
            invalid_geoms = segments_gdf[~segments_gdf.geometry.is_valid]
            if len(invalid_geoms) > 0:
                self._log(f"WARNING: Found {len(invalid_geoms)} invalid geometries")
            
            # Check for empty geometries
            empty_geoms = segments_gdf[segments_gdf.geometry.is_empty]
            if len(empty_geoms) > 0:
                self._log(f"WARNING: Found {len(empty_geoms)} empty geometries")
            
        except Exception as e:
            self._log(f"Validation error: {str(e)}")
    
    def _generate_h3_cells(self, segments_gdf: gpd.GeoDataFrame, h3_resolution: int, use_gpu: bool = True) -> Set[str]:
        """
        Generate H3 cells from sidewalk segments.
        
        Parameters:
        -----------
        segments_gdf : GeoDataFrame
            Input sidewalk segments
        h3_resolution : int
            H3 resolution
        use_gpu : bool
            Whether to use GPU acceleration
            
        Returns:
        --------
        Set[str]
            Set of H3 cell IDs
        """
        self._log("Generating H3 cells from sidewalk segments...")
        
        # Convert to WGS84 for H3 operations
        segments_wgs84 = segments_gdf.to_crs('EPSG:4326')
        
        # Generate H3 cells covering the network extent
        intersecting_cells = self._find_intersecting_h3_cells(segments_wgs84, h3_resolution)
        
        if not intersecting_cells:
            self._log("No H3 cells intersect with sidewalk segments")
            return set()
        
        self._log(f"Generated {len(intersecting_cells)} H3-{h3_resolution} cells")
        return intersecting_cells
    
    def _convert_h3_cells_to_points(self, h3_cells: Set[str], h3_resolution: int, segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Convert H3 cells to points with clipped segment geometries (optimized batch version).
        
        Parameters:
        -----------
        h3_cells : Set[str]
            Set of H3 cell IDs
        h3_resolution : int
            H3 resolution
        segments_gdf : GeoDataFrame
            Original sidewalk segments
            
        Returns:
        --------
        GeoDataFrame
            Points representing H3 cells with clipped segment geometries
        """
        import time
        start_time = time.time()
        
        self._log("Converting H3 cells to points with clipped segment geometries (batch optimized)...")
        self._log(f"DEBUG: Input - {len(h3_cells)} H3 cells, segments_gdf shape: {segments_gdf.shape}, CRS: {segments_gdf.crs}")
        
        if not h3_cells:
            self._log("DEBUG: No H3 cells provided, returning empty GeoDataFrame")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        # Use optimized processing based on dataset size
        result = None
        processing_method = None
        
        try:
            if len(h3_cells) > 20000:
                # Use parallel processing for very large datasets
                processing_method = "parallel"
                self._log(f"DEBUG: Using parallel processing for {len(h3_cells)} cells (>20k)")
                result = self._convert_h3_cells_to_points_parallel(h3_cells, h3_resolution, segments_gdf)
            elif len(h3_cells) > 5000:
                # Use batch processing for large datasets
                processing_method = "batch_optimized"
                self._log(f"DEBUG: Using batch-optimized processing for {len(h3_cells)} cells (>5k)")
                result = self._convert_h3_cells_to_points_batch_optimized(h3_cells, h3_resolution, segments_gdf)
            else:
                # Use vectorized processing for smaller datasets
                processing_method = "vectorized"
                self._log(f"DEBUG: Using vectorized processing for {len(h3_cells)} cells (<=5k)")
                result = self._convert_h3_cells_to_points_vectorized(h3_cells, h3_resolution, segments_gdf)
        
            # Validate result
            if result is None:
                self._log(f"ERROR: {processing_method} processing returned None")
                return gpd.GeoDataFrame(
                    columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                            'clipped_segments', 'segment_count', 'total_clipped_length'],
                    crs=segments_gdf.crs
                )
            
            self._log(f"DEBUG: {processing_method} processing completed successfully")
            self._log(f"DEBUG: Result type: {type(result)}, shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
            if hasattr(result, 'columns'):
                self._log(f"DEBUG: Result columns: {list(result.columns)}")
            if hasattr(result, 'crs'):
                self._log(f"DEBUG: Result CRS: {result.crs}")
            
        except Exception as e:
            self._log(f"ERROR: {processing_method} processing failed: {e}")
            self._log(f"DEBUG: Exception details: {str(e)}")
            import traceback
            self._log(f"DEBUG: Traceback: {traceback.format_exc()}")
            
            # Return empty GeoDataFrame on error
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        elapsed_time = time.time() - start_time
        self._log(f"Step 3 completed in {elapsed_time:.2f} seconds ({len(h3_cells)} cells -> {len(result)} points)")
        
        return result
    
    def _convert_h3_cells_to_points_batch_optimized(self, h3_cells: Set[str], h3_resolution: int, segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Batch-optimized version for large datasets using vectorized operations.
        """
        self._log(f"Using batch-optimized processing for {len(h3_cells)} H3 cells...")
        
        # Import required libraries
        from shapely.geometry import Polygon, Point
        from shapely.strtree import STRtree
        
        # Create spatial index once for all operations
        segment_geometries = list(segments_gdf.geometry)
        spatial_index = STRtree(segment_geometries)
        
        # Step 1: Batch create all H3 geometries in WGS84
        self._log("Step 1: Batch creating H3 geometries in WGS84...")
        h3_data = []
        h3_cells_list = list(h3_cells)
        
        for i, cell_id in enumerate(h3_cells_list):
            # Progress logging every 2000 cells
            if i % 2000 == 0 and i > 0:
                self._log(f"  Created {i}/{len(h3_cells_list)} H3 geometries ({i/len(h3_cells_list)*100:.1f}%)")
            
            try:
                # Get H3 cell boundary and centroid
                cell_boundary = h3.cell_to_boundary(cell_id)
                centroid = h3.cell_to_latlng(cell_id)
                
                # Convert boundary to Shapely polygon
                cell_boundary_geojson = [[coord[1], coord[0]] for coord in cell_boundary]
                h3_polygon = Polygon(cell_boundary_geojson)
                
                # Create centroid point
                centroid_point = Point(centroid[1], centroid[0])
                
                h3_data.append({
                    'h3_cell_id': cell_id,
                    'boundary_wgs84': h3_polygon,
                    'centroid_wgs84': centroid_point
                })
                
            except Exception as e:
                self._log(f"Error processing H3 cell {cell_id}: {e}")
                continue
        
        if not h3_data:
            self._log("No valid H3 geometries created")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        # Step 2: Batch convert all geometries to target CRS
        self._log("Step 2: Batch converting all H3 geometries to target CRS...")
        self._log(f"  Converting {len(h3_data)} boundaries and centroids to {segments_gdf.crs}...")
        
        # Create GeoDataFrames for batch conversion
        boundaries_gdf = gpd.GeoDataFrame(
            data=[{'h3_cell_id': item['h3_cell_id']} for item in h3_data],
            geometry=[item['boundary_wgs84'] for item in h3_data],
            crs='EPSG:4326'
        ).to_crs(segments_gdf.crs)
        
        centroids_gdf = gpd.GeoDataFrame(
            data=[{'h3_cell_id': item['h3_cell_id']} for item in h3_data],
            geometry=[item['centroid_wgs84'] for item in h3_data],
            crs='EPSG:4326'
        ).to_crs(segments_gdf.crs)
        
        self._log("  Completed: CRS conversion finished")
        
        # Step 3: Process cells in batches for clipping
        self._log("Step 3: Processing cells in batches for segment clipping...")
        
        # Adjust batch size for very large datasets
        if len(h3_data) > 5000000:  # 5M+ cells
            batch_size = 500
            self._log(f"  Using smaller batch size ({batch_size}) for large dataset ({len(h3_data)} cells)")
        elif len(h3_data) > 1000000:  # 1M+ cells
            batch_size = 750
            self._log(f"  Using reduced batch size ({batch_size}) for large dataset ({len(h3_data)} cells)")
        else:
            batch_size = 1000
        
        h3_points = []
        total_batches = (len(h3_data) + batch_size - 1) // batch_size
        
        for batch_start in range(0, len(h3_data), batch_size):
            batch_end = min(batch_start + batch_size, len(h3_data))
            batch_num = batch_start // batch_size + 1
            
            # More frequent progress logging (every 10 batches or every 1000 cells)
            if batch_num % 10 == 0 or batch_num == 1 or batch_num == total_batches:
                self._log(f"  Processing batch {batch_num}/{total_batches} ({batch_start + 1}-{batch_end} of {len(h3_data)} cells)")
            
            # Process batch
            batch_points = []
            for i in range(batch_start, batch_end):
                cell_id = h3_data[i]['h3_cell_id']
                h3_polygon_projected = boundaries_gdf.geometry.iloc[i]
                centroid_projected = centroids_gdf.geometry.iloc[i]
                
                # Validate geometries before processing
                if h3_polygon_projected is None or centroid_projected is None:
                    continue
                if not (hasattr(h3_polygon_projected, 'is_valid') and h3_polygon_projected.is_valid):
                    continue
                if not (hasattr(centroid_projected, 'is_valid') and centroid_projected.is_valid):
                    continue
                
                # Find intersecting segments
                intersecting_segments = self._clip_segments_to_h3_cell_optimized(
                    segments_gdf, h3_polygon_projected, cell_id, h3_resolution, spatial_index
                )
                
                if intersecting_segments:
                    # Extract clipped geometries for parquet storage
                    clipped_geometries = []
                    segment_widths = []
                    
                    for seg in intersecting_segments:
                        clipped_geom = seg.get('clipped_geometry')
                        if clipped_geom and not clipped_geom.is_empty:
                            clipped_geometries.append(clipped_geom)
                        
                        # Extract width from original segment attributes
                        if 'original_width' in seg:
                            segment_widths.append(seg['original_width'])
                    
                    # Create MultiLineString from clipped segments for storage
                    if clipped_geometries:
                        if len(clipped_geometries) == 1:
                            clipped_multigeom = clipped_geometries[0]
                        else:
                            from shapely.geometry import MultiLineString
                            clipped_multigeom = MultiLineString(clipped_geometries)
                    else:
                        clipped_multigeom = None
                    
                    # Calculate average width
                    avg_width = sum(segment_widths) / len(segment_widths) if segment_widths else None
                    
                    point_record = {
                        'geometry': centroid_projected,  # Keep centroid as main geometry
                        'h3_cell_id': cell_id,
                        'h3_resolution': h3_resolution,
                        'clipped_segments_geometry': clipped_multigeom,  # Store clipped segments for parquet
                        'avg_sidewalk_width': avg_width,  # Store average width
                        'clipped_segments': intersecting_segments,  # Keep detailed info (removed in parquet prep)
                        'segment_count': len(intersecting_segments),
                        'total_clipped_length': sum(
                            seg.get('clipped_length', 0) for seg in intersecting_segments
                        )
                    }
                    batch_points.append(point_record)
            
            h3_points.extend(batch_points)
            
            # Memory management for large datasets - garbage collect every 100 batches
            if batch_num % 100 == 0:
                import gc
                gc.collect()
                self._log(f"  Memory cleanup completed at batch {batch_num}")
        
        # Final memory cleanup
        import gc
        gc.collect()
        
        self._log(f"  Completed: Processed all {total_batches} batches, generated {len(h3_points)} H3 points")
        
        if not h3_points:
            self._log("No H3 points generated")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        # Validate geometries before creating GeoDataFrame
        valid_h3_points = []
        invalid_count = 0
        
        for point_record in h3_points:
            if point_record.get('geometry') is not None:
                try:
                    # Check if geometry is valid
                    if hasattr(point_record['geometry'], 'is_valid') and point_record['geometry'].is_valid:
                        valid_h3_points.append(point_record)
                    else:
                        invalid_count += 1
                except Exception as e:
                    invalid_count += 1
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            self._log(f"Filtered out {invalid_count} invalid geometries out of {len(h3_points)} total points")
        
        if not valid_h3_points:
            self._log("No valid H3 points after geometry validation")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        result = gpd.GeoDataFrame(valid_h3_points, crs=segments_gdf.crs)
        self._log(f"Generated {len(result)} H3-{h3_resolution} points using batch optimization")
        return result
    
    def _convert_h3_cells_to_points_vectorized(self, h3_cells: Set[str], h3_resolution: int, segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Vectorized version for smaller datasets using pandas operations.
        """
        self._log(f"Using vectorized processing for {len(h3_cells)} H3 cells...")
        
        # Import required libraries
        from shapely.geometry import Polygon, Point
        from shapely.strtree import STRtree
        
        # Create spatial index
        segment_geometries = list(segments_gdf.geometry)
        spatial_index = STRtree(segment_geometries)
        
        # Step 1: Vectorized creation of all H3 geometries
        self._log("Step 1: Creating H3 geometries in WGS84...")
        h3_cells_list = list(h3_cells)
        
        # Get all boundaries and centroids in one go
        boundaries_data = []
        centroids_data = []
        valid_cells = []
        
        for i, cell_id in enumerate(h3_cells_list):
            # Progress logging every 1000 cells for smaller datasets
            if i % 1000 == 0 and i > 0:
                self._log(f"  Created {i}/{len(h3_cells_list)} H3 geometries ({i/len(h3_cells_list)*100:.1f}%)")
            
            try:
                # Get H3 cell boundary and centroid
                cell_boundary = h3.cell_to_boundary(cell_id)
                centroid = h3.cell_to_latlng(cell_id)
                
                # Convert boundary to Shapely polygon
                cell_boundary_geojson = [[coord[1], coord[0]] for coord in cell_boundary]
                h3_polygon = Polygon(cell_boundary_geojson)
                
                # Create centroid point
                centroid_point = Point(centroid[1], centroid[0])
                
                boundaries_data.append(h3_polygon)
                centroids_data.append(centroid_point)
                valid_cells.append(cell_id)
                
            except Exception as e:
                self._log(f"Error processing H3 cell {cell_id}: {e}")
                continue
        
        self._log(f"  Completed: Created {len(valid_cells)} valid H3 geometries")
        
        if not valid_cells:
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        # Step 2: Vectorized CRS conversion
        self._log("Step 2: Converting H3 geometries to target CRS...")
        boundaries_gdf = gpd.GeoDataFrame(
            data={'h3_cell_id': valid_cells},
            geometry=boundaries_data,
            crs='EPSG:4326'
        ).to_crs(segments_gdf.crs)
        
        centroids_gdf = gpd.GeoDataFrame(
            data={'h3_cell_id': valid_cells},
            geometry=centroids_data,
            crs='EPSG:4326'
        ).to_crs(segments_gdf.crs)
        
        self._log("  Completed: CRS conversion finished")
        
        # Step 3: Vectorized processing
        self._log("Step 3: Processing cells for segment clipping...")
        h3_points = []
        
        for idx, (_, boundary_row) in enumerate(boundaries_gdf.iterrows()):
            # Progress logging every 500 cells for smaller datasets
            if idx % 500 == 0 and idx > 0:
                self._log(f"  Processed {idx}/{len(boundaries_gdf)} cells ({idx/len(boundaries_gdf)*100:.1f}%)")
            
            try:
                cell_id = boundary_row['h3_cell_id']
                h3_polygon_projected = boundary_row.geometry
                centroid_projected = centroids_gdf.geometry.iloc[idx]
                
                # Find intersecting segments
                intersecting_segments = self._clip_segments_to_h3_cell_optimized(
                    segments_gdf, h3_polygon_projected, cell_id, h3_resolution, spatial_index
                )
                
                if intersecting_segments:
                    # Extract clipped geometries for parquet storage
                    clipped_geometries = []
                    segment_widths = []
                    
                    for seg in intersecting_segments:
                        clipped_geom = seg.get('clipped_geometry')
                        if clipped_geom and not clipped_geom.is_empty:
                            clipped_geometries.append(clipped_geom)
                        
                        # Extract width from original segment attributes
                        if 'original_width' in seg:
                            segment_widths.append(seg['original_width'])
                    
                    # Create MultiLineString from clipped segments for storage
                    if clipped_geometries:
                        if len(clipped_geometries) == 1:
                            clipped_multigeom = clipped_geometries[0]
                        else:
                            from shapely.geometry import MultiLineString
                            clipped_multigeom = MultiLineString(clipped_geometries)
                    else:
                        clipped_multigeom = None
                    
                    # Calculate average width
                    avg_width = sum(segment_widths) / len(segment_widths) if segment_widths else None
                    
                    point_record = {
                        'geometry': centroid_projected,  # Keep centroid as main geometry
                        'h3_cell_id': cell_id,
                        'h3_resolution': h3_resolution,
                        'clipped_segments_geometry': clipped_multigeom,  # Store clipped segments for parquet
                        'avg_sidewalk_width': avg_width,  # Store average width
                        'clipped_segments': intersecting_segments,  # Keep detailed info (removed in parquet prep)
                        'segment_count': len(intersecting_segments),
                        'total_clipped_length': sum(
                            seg.get('clipped_length', 0) for seg in intersecting_segments
                        )
                    }
                    h3_points.append(point_record)
            
            except Exception as e:
                self._log(f"Error processing H3 cell {cell_id}: {e}")
                continue
        
        self._log(f"  Completed: Processed all {len(boundaries_gdf)} cells, generated {len(h3_points)} H3 points")
        
        if not h3_points:
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        # Validate geometries before creating GeoDataFrame
        valid_h3_points = []
        invalid_count = 0
        
        for point_record in h3_points:
            if point_record.get('geometry') is not None:
                try:
                    # Check if geometry is valid
                    if hasattr(point_record['geometry'], 'is_valid') and point_record['geometry'].is_valid:
                        valid_h3_points.append(point_record)
                    else:
                        invalid_count += 1
                except Exception as e:
                    invalid_count += 1
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            self._log(f"Filtered out {invalid_count} invalid geometries out of {len(h3_points)} total points")
        
        if not valid_h3_points:
            self._log("No valid H3 points after geometry validation")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        result = gpd.GeoDataFrame(valid_h3_points, crs=segments_gdf.crs)
        self._log(f"Generated {len(result)} H3-{h3_resolution} points using vectorized processing")
        return result
    
    def _convert_h3_cells_to_points_parallel(self, h3_cells: Set[str], h3_resolution: int, segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Parallel processing version using multiple CPU cores for maximum performance.
        """
        self._log(f"Using parallel processing for {len(h3_cells)} H3 cells...")
        
        # Import required libraries
        from shapely.geometry import Polygon, Point
        
        # Step 1: Batch create all H3 geometries in WGS84
        self._log("Step 1: Batch creating H3 geometries in WGS84...")
        h3_data = []
        h3_cells_list = list(h3_cells)
        
        for i, cell_id in enumerate(h3_cells_list):
            # Progress logging every 5000 cells for very large datasets
            if i % 5000 == 0 and i > 0:
                self._log(f"  Created {i}/{len(h3_cells_list)} H3 geometries ({i/len(h3_cells_list)*100:.1f}%)")
            
            try:
                # Get H3 cell boundary and centroid
                cell_boundary = h3.cell_to_boundary(cell_id)
                centroid = h3.cell_to_latlng(cell_id)
                
                # Convert boundary to Shapely polygon
                cell_boundary_geojson = [[coord[1], coord[0]] for coord in cell_boundary]
                h3_polygon = Polygon(cell_boundary_geojson)
                
                # Create centroid point
                centroid_point = Point(centroid[1], centroid[0])
                
                h3_data.append({
                    'h3_cell_id': cell_id,
                    'boundary_wgs84': h3_polygon,
                    'centroid_wgs84': centroid_point
                })
                
            except Exception as e:
                self._log(f"Error processing H3 cell {cell_id}: {e}")
                continue
        
        if not h3_data:
            self._log("No valid H3 geometries created")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        self._log(f"  Completed: Created {len(h3_data)} valid H3 geometries")
        
        # Step 2: Batch convert all geometries to target CRS
        self._log("Step 2: Batch converting all H3 geometries to target CRS...")
        self._log(f"  Converting {len(h3_data)} boundaries and centroids to {segments_gdf.crs}...")
        
        # Create GeoDataFrames for batch conversion
        boundaries_gdf = gpd.GeoDataFrame(
            data=[{'h3_cell_id': item['h3_cell_id']} for item in h3_data],
            geometry=[item['boundary_wgs84'] for item in h3_data],
            crs='EPSG:4326'
        ).to_crs(segments_gdf.crs)
        
        centroids_gdf = gpd.GeoDataFrame(
            data=[{'h3_cell_id': item['h3_cell_id']} for item in h3_data],
            geometry=[item['centroid_wgs84'] for item in h3_data],
            crs='EPSG:4326'
        ).to_crs(segments_gdf.crs)
        
        self._log("  Completed: CRS conversion finished")
        
        # Step 3: Parallel processing for segment clipping
        self._log("Step 3: Parallel processing for segment clipping...")
        
        # Determine number of workers (locked to 8 for optimal performance)
        num_workers = 8
        self._log(f"Using {num_workers} parallel workers for processing (locked to 8)")
        
        # Split data into chunks for parallel processing
        chunk_size = max(1, len(h3_data) // num_workers)
        chunks = []
        
        for i in range(0, len(h3_data), chunk_size):
            chunk_end = min(i + chunk_size, len(h3_data))
            chunk_data = {
                'h3_data': h3_data[i:chunk_end],
                'boundaries_geoms': [boundaries_gdf.geometry.iloc[j] for j in range(i, chunk_end)],
                'centroids_geoms': [centroids_gdf.geometry.iloc[j] for j in range(i, chunk_end)],
                'segments_data': {
                    'geometries': list(segments_gdf.geometry),
                    'columns': list(segments_gdf.columns),
                    'data': segments_gdf.to_dict('records'),
                    'crs': str(segments_gdf.crs)
                },
                'h3_resolution': h3_resolution,
                'chunk_id': i // chunk_size
            }
            chunks.append(chunk_data)
        
        # Process chunks in parallel
        h3_points = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(H3NetworkGenerator._process_h3_chunk_parallel, chunk): chunk['chunk_id']
                for chunk in chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    h3_points.extend(chunk_results)
                    self._log(f"Completed chunk {chunk_id + 1}/{len(chunks)}")
                except Exception as e:
                    self._log(f"Error processing chunk {chunk_id}: {e}")
        
        self._log(f"  Completed: All {len(chunks)} chunks processed, generated {len(h3_points)} H3 points")
        
        if not h3_points:
            self._log("No H3 points generated")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        # Validate geometries before creating GeoDataFrame
        valid_h3_points = []
        invalid_count = 0
        
        for point_record in h3_points:
            if point_record.get('geometry') is not None:
                try:
                    # Check if geometry is valid
                    if hasattr(point_record['geometry'], 'is_valid') and point_record['geometry'].is_valid:
                        valid_h3_points.append(point_record)
                    else:
                        invalid_count += 1
                except Exception as e:
                    invalid_count += 1
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            self._log(f"Filtered out {invalid_count} invalid geometries out of {len(h3_points)} total points")
        
        if not valid_h3_points:
            self._log("No valid H3 points after geometry validation")
            return gpd.GeoDataFrame(
                columns=['geometry', 'h3_cell_id', 'h3_resolution', 'clipped_segments_geometry', 'avg_sidewalk_width', 
                        'clipped_segments', 'segment_count', 'total_clipped_length'],
                crs=segments_gdf.crs
            )
        
        result = gpd.GeoDataFrame(valid_h3_points, crs=segments_gdf.crs)
        self._log(f"Generated {len(result)} H3-{h3_resolution} points using parallel processing")
        return result
    
    @staticmethod
    def _process_h3_chunk_parallel(chunk_data: Dict) -> List[Dict]:
        """
        Process a chunk of H3 cells in parallel (static method for multiprocessing).
        """
        from shapely.strtree import STRtree
        import geopandas as gpd
        
        # Extract chunk data
        h3_data = chunk_data['h3_data']
        boundaries_geoms = chunk_data['boundaries_geoms']
        centroids_geoms = chunk_data['centroids_geoms']
        segments_data = chunk_data['segments_data']
        h3_resolution = chunk_data['h3_resolution']
        
        # Reconstruct segments GeoDataFrame
        segments_gdf = gpd.GeoDataFrame(
            data=segments_data['data'],
            geometry=segments_data['geometries'],
            crs=segments_data['crs']
        )
        
        # Create spatial index for this chunk
        spatial_index = STRtree(segments_data['geometries'])
        
        # Process each H3 cell in this chunk
        chunk_points = []
        
        for i, h3_item in enumerate(h3_data):
            try:
                cell_id = h3_item['h3_cell_id']
                h3_polygon_projected = boundaries_geoms[i]
                centroid_projected = centroids_geoms[i]
                
                # Find intersecting segments (optimized clipping)
                intersecting_segments = H3NetworkGenerator._clip_segments_to_h3_cell_static(
                    segments_gdf, h3_polygon_projected, cell_id, h3_resolution, spatial_index
                )
                
                if intersecting_segments:
                    # Extract clipped geometries for parquet storage
                    clipped_geometries = []
                    segment_widths = []
                    
                    for seg in intersecting_segments:
                        clipped_geom = seg.get('clipped_geometry')
                        if clipped_geom and not clipped_geom.is_empty:
                            clipped_geometries.append(clipped_geom)
                        
                        # Extract width from original segment attributes
                        if 'original_width' in seg:
                            segment_widths.append(seg['original_width'])
                    
                    # Create MultiLineString from clipped segments for storage
                    if clipped_geometries:
                        if len(clipped_geometries) == 1:
                            clipped_multigeom = clipped_geometries[0]
                        else:
                            from shapely.geometry import MultiLineString
                            clipped_multigeom = MultiLineString(clipped_geometries)
                    else:
                        clipped_multigeom = None
                    
                    # Calculate average width
                    avg_width = sum(segment_widths) / len(segment_widths) if segment_widths else None
                    
                    point_record = {
                        'geometry': centroid_projected,  # Keep centroid as main geometry
                        'h3_cell_id': cell_id,
                        'h3_resolution': h3_resolution,
                        'clipped_segments_geometry': clipped_multigeom,  # Store clipped segments for parquet
                        'avg_sidewalk_width': avg_width,  # Store average width
                        'clipped_segments': intersecting_segments,  # Keep detailed info (removed in parquet prep)
                        'segment_count': len(intersecting_segments),
                        'total_clipped_length': sum(
                            seg.get('clipped_length', 0) for seg in intersecting_segments
                        )
                    }
                    chunk_points.append(point_record)
                    
            except Exception as e:
                # Can't use self._log in static method, so continue silently
                continue
        
        return chunk_points
    
    @staticmethod
    def _clip_segments_to_h3_cell_static(segments_gdf: gpd.GeoDataFrame,
                                        h3_polygon: 'Polygon',
                                        cell_id: str,
                                        h3_resolution: int,
                                        spatial_index: 'STRtree') -> List[Dict]:
        """
        Static version of segment clipping for parallel processing.
        """
        clipped_segments = []
        
        try:
            # Use spatial index to find candidate segments efficiently
            candidate_segments = spatial_index.query(h3_polygon)
            
            for segment_idx in candidate_segments:
                if segment_idx >= len(segments_gdf):
                    continue
                    
                segment_row = segments_gdf.iloc[segment_idx]
                segment_geom = segment_row.geometry
                
                # Check if segment intersects with H3 cell
                if h3_polygon.intersects(segment_geom):
                    try:
                        # Clip segment to H3 cell boundary
                        clipped_geom = h3_polygon.intersection(segment_geom)
                        
                        if not clipped_geom.is_empty:
                            # Calculate clipped length
                            clipped_length = clipped_geom.length
                            
                            # Create clipped segment record
                            clipped_segment = {
                                'original_segment_id': segment_idx,
                                'h3_cell_id': cell_id,
                                'h3_resolution': h3_resolution,
                                'clipped_geometry': clipped_geom,
                                'clipped_length': clipped_length,
                                'original_length': segment_geom.length,
                                'clipping_ratio': clipped_length / segment_geom.length if segment_geom.length > 0 else 0
                            }
                            
                            # Add original segment attributes
                            for col in segments_gdf.columns:
                                if col != 'geometry':
                                    clipped_segment[f'original_{col}'] = segment_row[col]
                            
                            clipped_segments.append(clipped_segment)
                            
                    except Exception as e:
                        continue
            
            return clipped_segments
            
        except Exception as e:
            return []
    
    def _test_h3_functionality(self):
        """Test basic H3 functionality to ensure the API is working correctly."""
        try:
            # Test basic H3 functions using current h3-py API
            test_cell = h3.latlng_to_cell(40.7128, -74.0060, 9)  # NYC coordinates
            test_boundary = h3.cell_to_boundary(test_cell)
            test_centroid = h3.cell_to_latlng(test_cell)
            test_neighbors = h3.grid_disk(test_cell, 1)
            
            # Validate the results
            if not test_cell:
                raise ValueError("H3 cell generation failed")
            if len(test_boundary) != 6:
                raise ValueError(f"H3 boundary should have 6 points, got {len(test_boundary)}")
            if len(test_centroid) != 2:
                raise ValueError(f"H3 centroid should have 2 coordinates, got {len(test_centroid)}")
            if len(test_neighbors) != 7:  # includes the center cell
                raise ValueError(f"H3 grid_disk should return 7 cells, got {len(test_neighbors)}")
            
            self._log("✓ H3 functionality test passed")
            self._log(f"  - Test cell: {test_cell}")
            self._log(f"  - Boundary points: {len(test_boundary)}")
            self._log(f"  - Boundary format: (lat, lng) pairs")
            self._log(f"  - Centroid: {test_centroid}")
            self._log(f"  - Neighbors: {len(test_neighbors)}")
            
            return True
            
        except Exception as e:
            self._log(f"ERROR: H3 functionality test failed: {e}")
            self._log("This may indicate an H3 API version compatibility issue")
            self._log("Please ensure you have h3-py >= 3.7.0 installed")
            return False
    
    def _configure_gpu_batch_sizes(self, dataset_size: int) -> Dict[str, int]:
        """
        Dynamically configure batch sizes and CUDA stream counts based on GPU memory and dataset size.
        
        Parameters:
        -----------
        dataset_size : int
            Size of the dataset to process
            
        Returns:
        --------
        Dict[str, int]
            Configuration with batch sizes and stream counts
        """
        try:
            # Get GPU memory info
            gpu_memory = cp.cuda.runtime.memGetInfo()
            free_memory_gb = gpu_memory[0] / (1024**3)
            total_memory_gb = gpu_memory[1] / (1024**3)
            
            self._log(f"GPU Memory: {free_memory_gb:.1f}GB free / {total_memory_gb:.1f}GB total")
            
            # Configure based on available memory and dataset size
            if free_memory_gb >= 8.0:  # High memory GPU
                config = {
                    'batch_size': 2000,
                    'super_batch_size': 50000,
                    'num_streams': 6,
                    'adjacency_batch_size': 1000,
                    'adjacency_super_batch_size': 20000
                }
            elif free_memory_gb >= 4.0:  # Medium memory GPU
                config = {
                    'batch_size': 1500,
                    'super_batch_size': 30000,
                    'num_streams': 4,
                    'adjacency_batch_size': 800,
                    'adjacency_super_batch_size': 15000
                }
            else:  # Low memory GPU
                config = {
                    'batch_size': 1000,
                    'super_batch_size': 20000,
                    'num_streams': 2,
                    'adjacency_batch_size': 500,
                    'adjacency_super_batch_size': 10000
                }
            
            # Adjust for very large datasets
            if dataset_size > 1000000:  # 1M+ segments
                config['batch_size'] = max(500, config['batch_size'] // 2)
                config['super_batch_size'] = max(10000, config['super_batch_size'] // 2)
                config['num_streams'] = max(2, config['num_streams'] - 1)
            
            # Adjust for very small datasets
            if dataset_size < 10000:  # < 10K segments
                config['batch_size'] = min(5000, config['batch_size'] * 2)
                config['super_batch_size'] = min(100000, config['super_batch_size'] * 2)
                config['num_streams'] = min(8, config['num_streams'] + 1)
            
            self._log(f"GPU Configuration: {config}")
            return config
            
        except Exception as e:
            self._log(f"Could not get GPU memory info, using default configuration: {e}")
            # Default configuration
            return {
                'batch_size': 2000,
                'super_batch_size': 50000,
                'num_streams': 4,
                'adjacency_batch_size': 1000,
                'adjacency_super_batch_size': 20000
            }
    
    def _optimize_gpu_memory_usage(self):
        """
        Optimize GPU memory usage by clearing cache and setting memory pool.
        """
        try:
            # Clear GPU memory cache
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            # Set memory pool for better memory management
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            
            self._log("GPU memory optimized and cache cleared")
            
        except Exception as e:
            self._log(f"Could not optimize GPU memory: {e}")
    
    def _monitor_gpu_memory(self, stage: str):
        """
        Monitor GPU memory usage during processing.
        
        Parameters:
        -----------
        stage : str
            Current processing stage for logging
        """
        try:
            gpu_memory = cp.cuda.runtime.memGetInfo()
            free_memory_gb = gpu_memory[0] / (1024**3)
            used_memory_gb = (gpu_memory[1] - gpu_memory[0]) / (1024**3)
            
            self._log(f"GPU Memory at {stage}: {free_memory_gb:.1f}GB free, {used_memory_gb:.1f}GB used")
            
        except Exception as e:
            self._log(f"Could not monitor GPU memory: {e}")
    
    def _prepare_for_parquet_saving(self, h3_network_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Prepare H3 network data for parquet saving by handling non-serializable columns.
        
        Parameters:
        -----------
        h3_network_gdf : GeoDataFrame
            H3 network with all data including non-serializable columns
            
        Returns:
        --------
        GeoDataFrame
            Parquet-safe version of the H3 network
        """
        # Create a copy for saving
        save_gdf = h3_network_gdf.copy()
        
        # Remove clipped_segments column (detailed dictionaries) but keep clipped_segments_geometry
        if 'clipped_segments' in save_gdf.columns:
            self._log("Removing clipped_segments column for parquet compatibility")
            save_gdf = save_gdf.drop(columns=['clipped_segments'])
        
        # Convert clipped_segments_geometry from Shapely objects to WKT for parquet compatibility
        if 'clipped_segments_geometry' in save_gdf.columns:
            self._log("Converting clipped_segments_geometry to WKT for parquet storage")
            save_gdf['clipped_segments_geometry'] = save_gdf['clipped_segments_geometry'].apply(
                lambda geom: geom.wkt if geom is not None and not geom.is_empty else None
            )
        
        # avg_sidewalk_width is already parquet-compatible (numeric)
        if 'avg_sidewalk_width' in save_gdf.columns:
            self._log("Keeping avg_sidewalk_width for parquet storage")
        
        # Convert h3_adjacent_cells from list to string for parquet compatibility
        if 'h3_adjacent_cells' in save_gdf.columns:
            self._log("Converting h3_adjacent_cells to string for parquet compatibility")
            save_gdf['h3_adjacent_cells'] = save_gdf['h3_adjacent_cells'].apply(
                lambda x: ','.join(map(str, x)) if x and len(x) > 0 else ''
            )
        
        return save_gdf


if __name__ == "__main__":
    def main():
        """Command-line interface for H3NetworkGenerator."""
        parser = argparse.ArgumentParser(
            description="Generate H3-based sidewalk networks from segment data",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic usage with auto-generated filename (includes resolution and timestamp)
  python h3_network_generator.py input_segments.parquet
  
  # Basic usage with custom output filename (still adds resolution and timestamp)
  python h3_network_generator.py input_segments.parquet --output output_network.parquet
  
  # Custom H3 resolution (filename will include resolution)
  python h3_network_generator.py input_segments.parquet --resolution 10
  
  # Save intermediate results with timestamped filenames
  python h3_network_generator.py input_segments.parquet --save-intermediate --intermediate-dir ./intermediate
  
  # Disable GPU acceleration
  python h3_network_generator.py input_segments.parquet --no-gpu
  
  # Disable graph analytics  
  python h3_network_generator.py input_segments.parquet --no-graph-analytics
            """
        )
        
        parser.add_argument(
            'input_path',
            help='Path to input GeoParquet file containing sidewalk segments'
        )
        
        parser.add_argument(
            '--output', '-o',
            required=False,
            help='Path to output GeoParquet file for the H3 network (if not provided, generates dynamic filename with resolution and timestamp)'
        )
        
        parser.add_argument(
            '--resolution', '-r',
            type=int,
            default=9,
            help='H3 resolution (default: 9, higher = smaller cells)'
        )
        
        parser.add_argument(
            '--save-intermediate',
            action='store_true',
            help='Save intermediate results during processing'
        )
        
        parser.add_argument(
            '--intermediate-dir',
            help='Directory to save intermediate results (required if --save-intermediate is used)'
        )
        
        parser.add_argument(
            '--no-gpu',
            action='store_true',
            help='Disable GPU acceleration (use CPU only)'
        )
        
        parser.add_argument(
            '--no-graph-analytics',
            action='store_true',
            help='Disable advanced graph analytics with cuGraph'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        args = parser.parse_args()
        
        # Validate arguments
        if args.save_intermediate and not args.intermediate_dir:
            parser.error("--intermediate-dir is required when --save-intermediate is used")
        
        if args.intermediate_dir and not os.path.exists(args.intermediate_dir):
            try:
                os.makedirs(args.intermediate_dir)
            except Exception as e:
                parser.error(f"Could not create intermediate directory: {e}")
        
        # Create generator instance
        generator = H3NetworkGenerator(verbose=args.verbose)
        
        # Test H3 functionality
        if not generator._test_h3_functionality():
            print("ERROR: H3 functionality test failed. Please check your H3 installation.", flush=True)
            return 1
        
        # Process segments to H3 network
        try:
            result = generator.process_segments_to_h3_network(
                segments_input=args.input_path,
                h3_resolution=args.resolution,
                output_path=args.output,
                save_intermediate=args.save_intermediate,
                intermediate_dir=args.intermediate_dir,
                use_gpu=not args.no_gpu,
                enable_graph_analytics=not args.no_graph_analytics
            )
            
            if len(result) > 0:
                actual_output = result.attrs.get('output_path', 'Unknown')
                print(f"\nSUCCESS: Generated H3 network with {len(result)} points", flush=True)
                print(f"H3 Resolution: {args.resolution}", flush=True)
                print(f"Output saved to: {actual_output}", flush=True)
                return 0
            else:
                print("ERROR: No H3 points generated", flush=True)
                return 1
            
        except Exception as e:
            print(f"ERROR: Processing failed: {e}", flush=True)
            return 1
    
    # Run the CLI
    main() 