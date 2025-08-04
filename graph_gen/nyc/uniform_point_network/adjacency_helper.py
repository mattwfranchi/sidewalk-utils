import cupy as cp
import cudf  
import cuspatial
import cugraph
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from typing import Dict, List, Optional, Tuple
import random
import sys
sys.path.append('/share/ju/sidewalk_utils')
from utils.logger import get_logger
from collections import defaultdict
import momepy

# NetworkX imports for graph operations
import networkx as nx
try:
    import nx_cugraph as nxcg
    NX_CUGRAPH_AVAILABLE = True
except ImportError:
    NX_CUGRAPH_AVAILABLE = False

class AdjacencyHelper:
    """Helper class for computing adjacency relationships in sampled networks."""
    
    def __init__(self):
        self.logger = get_logger("AdjacencyHelper")
        
        # Configure NetworkX to use GPU backend if available
        if NX_CUGRAPH_AVAILABLE:
            self.logger.info("nx-cugraph available - will use GPU acceleration for NetworkX operations")
        else:
            self.logger.warning("nx-cugraph not available - using CPU NetworkX operations")
        
        # Initialize graph storage
        self.segment_graph = None
        self._cached_segment_connectivity = None
    
    def check_gpu_backend_status(self) -> Dict:
        """
        Check the status of nx-cugraph GPU backend configuration.
        
        Returns:
        --------
        Dict
            Status information about GPU backend availability
        """
        status = {
            'nx_cugraph_available': NX_CUGRAPH_AVAILABLE,
            'gpu_backend_active': False
        }
        
        if NX_CUGRAPH_AVAILABLE:
            try:
                # Test if we can create a simple nx-cugraph graph
                test_graph = nx.Graph()
                test_graph.add_edge(1, 2)
                gpu_graph = nxcg.from_networkx(test_graph)
                status['gpu_backend_active'] = True
                status['test_graph_nodes'] = gpu_graph.number_of_nodes()
                status['test_graph_edges'] = gpu_graph.number_of_edges()
            except Exception as e:
                status['gpu_backend_error'] = str(e)
                self.logger.warning(f"nx-cugraph backend test failed: {e}")
        
        self.logger.info(f"GPU backend status: {status}")
        return status

    def create_segment_network_graph(self, segments_gdf: gpd.GeoDataFrame, 
                                    points: List[Dict] = None) -> nx.Graph:
        """
        Convert uniform point network to NetworkX Graph using segment-first spatial overlay.
        
        This creates a point-based network graph where:
        1. Nodes = uniformly sampled points (already placed)
        2. Edges = connections between spatially related points
        3. Connections determined by segment-to-segment connectivity first, then point-level logic
        
        Parameters:
        -----------
        segments_gdf : gpd.GeoDataFrame
            Segment network with LineString geometries (may include crosswalks)
        points : List[Dict], optional
            Already-placed uniform points. If None, will be set during adjacency computation.
            
        Returns:
        --------
        nx.Graph
            NetworkX graph with sampled points as nodes
        """
        self.logger.info(f"Creating point-based NetworkX graph from segments and uniform points...")
        
        # If we have points, use the segment-first spatial overlay workflow
        if points is not None and len(points) > 0:
            self.logger.info(f"Using segment-first approach with {len(segments_gdf)} segments and {len(points)} points")
            
            # Step 1: Perform segment-first spatial overlay
            overlay_result = self._perform_spatial_overlay(points, segments_gdf)
            
            # Step 2: Build NetworkX graph from overlay results
            G = self._build_graph_from_overlay(points, overlay_result)
            
            # Store reference to points for later use
            self._current_points = points
        else:
            # Fallback: create empty graph if no points provided yet
            if NX_CUGRAPH_AVAILABLE:
                self.logger.info("nx-cugraph available for empty graph - backend dispatch will be used")
                G = momepy.gdf_to_nx(segments_gdf, approach='primal')
                total_adjacent_edges = 0

                for u, v in G.edges():
                    adjacent_edges = set(G.edges(u)) | set(G.edges(v))
                    adjacent_edges.discard((u, v))
                    adjacent_edges.discard((v, u))  # For undirected graphs
                    total_adjacent_edges += len(adjacent_edges)

                self.logger.info(f"Total sum of adjacent edges for all edges: {total_adjacent_edges}")

        
        self.logger.info(f"Created point-based NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def _perform_spatial_overlay(self, points: List[Dict], segments_gdf: gpd.GeoDataFrame) -> Dict:
        """
        Perform spatial overlay to assign points to segments and determine connections.
        
        This uses a segment-first approach:
        1. Compute segment-to-segment connectivity (which segments connect to which)
        2. Group points by segment and sort them along each segment
        3. Use segment connectivity to determine cross-segment point connections
        4. Add intersection-based connections
        
        Parameters:
        -----------
        points : List[Dict]
            Uniformly sampled points with geometry and metadata
        segments_gdf : gpd.GeoDataFrame
            Segment network for spatial reference
            
        Returns:
        --------
        Dict
            Dictionary containing point assignments and connection rules
        """
        self.logger.info(f"Performing segment-first spatial overlay for {len(points)} points and {len(segments_gdf)} segments...")
        
        overlay_result = {
            'segment_assignments': {},  # point_id -> segment_info
            'all_connections': []  # [(point1, point2, distance), ...] - unified nearest neighbor connections
        }
        
        # Step 1: Extract segment-to-segment connectivity from pre-computed NetworkX graph
        self.logger.info("Step 1: Extracting segment-to-segment connectivity from existing NetworkX graph...")
        if not hasattr(self, 'segment_graph') or self.segment_graph is None:
            self.logger.error("No NetworkX graph found! Graph must be created first with create_segment_network_graph()")
            raise ValueError("NetworkX graph must be created before performing spatial overlay")
        
        # Extract segment connectivity from the NetworkX graph
        segment_connectivity = self._extract_segment_connectivity_from_graph(segments_gdf)
        
        # Step 2: Index points by segment for fast lookup
        self.logger.info("Step 2: Indexing points by segment...")
        points_by_segment = {}
        
        for point in points:
            point_id = point['point_id']
            segment_idx = point.get('source_segment_idx')
            
            # Store segment assignment
            overlay_result['segment_assignments'][point_id] = {
                'segment_idx': segment_idx,
                'distance_along_segment': point.get('distance_along_segment', 0.0),
                'is_intersection': point.get('is_intersection', False)
            }
            
            # Group by segment (only if segment exists in our segments_gdf)
            if segment_idx is not None and segment_idx < len(segments_gdf):
                if segment_idx not in points_by_segment:
                    points_by_segment[segment_idx] = []
                points_by_segment[segment_idx].append(point)
                
        # Sort points within each segment by distance along segment
        for segment_idx in points_by_segment:
            points_by_segment[segment_idx].sort(key=lambda p: p.get('distance_along_segment', 0.0))
        
        self.logger.info(f"Indexed points across {len(points_by_segment)} segments")
        
        # Step 3: For each point, find nearest neighbors in all directions using segment network
        self.logger.info("Step 3: Building network topology using directional nearest neighbors...")
        all_connections = self._find_nearest_neighbors_via_segment_network(
            points, points_by_segment, segment_connectivity, max_search_distance=1000.0  # Increased for debugging
        )
        
        # Store all connections in the result - these will be used to create NetworkX edges
        overlay_result['all_connections'] = all_connections
        self.logger.info(f"Stored {len(all_connections)} connections for NetworkX graph creation")
        
        # Don't log here - already logged in the method above
        
        self.logger.info("Segment-first spatial overlay complete")
        return overlay_result



    def _find_nearest_neighbors_via_segment_network(self, points: List[Dict], 
                                                   points_by_segment: Dict[int, List[Dict]], 
                                                       segment_connectivity: Dict[int, List[int]],
                                                   max_search_distance: float = 200.0) -> List[Tuple]:
        """
        Find nearest neighbors using SSSP (Single Source Shortest Path) on the segment network.
        
        This approach:
        1. Creates a NetworkX graph from the segment connectivity
        2. For each point, runs SSSP to find all reachable points within max_search_distance
        3. For each direction, selects the nearest neighbor and stops
        4. Uses cuGraph backend for GPU acceleration when available
        
        Parameters:
        -----------
        points : List[Dict]
            All points in the network
        points_by_segment : Dict[int, List[Dict]]
            Points indexed by segment (sorted by distance along segment)
        segment_connectivity : Dict[int, List[int]]
            Pre-computed segment-to-segment connectivity
        max_search_distance : float
            Maximum network distance to search for neighbors
            
        Returns:
        --------
        List[Tuple]
            List of (point1_id, point2_id, network_distance) tuples for all connections
        """
        self.logger.info(f"Building network topology for {len(points)} points using SSSP nearest neighbors...")
        
        # Step 1: Create a complete segment+points network graph for SSSP
        network_graph = self._create_complete_network_graph(points, points_by_segment, segment_connectivity)
        
        # Step 2: Use SSSP to find nearest neighbors for each point
        all_connections = self._compute_sssp_nearest_neighbors(points, network_graph, max_search_distance)
        
        self.logger.info(f"Network topology complete: {len(all_connections)} edges connecting points")
        return all_connections

    def _create_complete_network_graph(self, points: List[Dict], 
                                     points_by_segment: Dict[int, List[Dict]], 
                                     segment_connectivity: Dict[int, List[int]]) -> nx.Graph:
        """
        Create a complete NetworkX graph that includes both segments and points for SSSP.
        
        The graph structure:
        - Segment endpoints are nodes (with special IDs to avoid collision with point IDs)
        - Points are nodes (using their point_id)  
        - Segment-to-segment connections are edges (weighted by segment length)
        - Point-to-segment connections are edges (weighted by distance along segment)
        - Point-to-point connections within segments are edges (weighted by distance between points)
        """
        self.logger.info("Creating complete network graph for SSSP calculations...")
        
        G = nx.Graph()
        
        # Step 1: Add segment endpoint nodes and segment-to-segment edges
        segment_endpoints = {}  # segment_idx -> {'start_node': node_id, 'end_node': node_id}
        
        # Use ALL segment indices that have points, not just those in connectivity
        all_segment_indices = set(segment_connectivity.keys()) | set(points_by_segment.keys())
        
        for segment_idx in all_segment_indices:
            # Create unique node IDs for segment endpoints (use negative numbers to avoid collision)
            start_node_id = f"seg_{segment_idx}_start"
            end_node_id = f"seg_{segment_idx}_end"
            
            segment_endpoints[segment_idx] = {
                'start_node': start_node_id,
                'end_node': end_node_id
            }
            
            # Add segment endpoint nodes
            G.add_node(start_node_id, node_type='segment_endpoint', segment_idx=segment_idx, endpoint_type='start')
            G.add_node(end_node_id, node_type='segment_endpoint', segment_idx=segment_idx, endpoint_type='end')
            
            # Add edge between segment endpoints (represents traversing the entire segment)
            if segment_idx in points_by_segment and len(points_by_segment[segment_idx]) > 0:
                # Use the total segment length from the points data
                segment_points = points_by_segment[segment_idx]
                segment_length = segment_points[-1]['segment_total_length']
            else:
                # Fallback: use a default segment length
                segment_length = 100.0  # feet
            
            G.add_edge(start_node_id, end_node_id, weight=segment_length, connection_type='segment_traversal')
        
        # Step 2: Add segment-to-segment connectivity edges
        for segment_idx, connected_segments in segment_connectivity.items():
            if segment_idx not in segment_endpoints:
                continue
                
            current_endpoints = segment_endpoints[segment_idx]
                
                for connected_segment_idx in connected_segments:
                    if connected_segment_idx not in segment_endpoints:
                        continue
                        
                    connected_endpoints = segment_endpoints[connected_segment_idx]
                    
                # Connect endpoints between segments (representing intersections)
                # We connect each endpoint to the closest endpoint on the connected segment
                for current_endpoint_type, current_node in current_endpoints.items():
                    for connected_endpoint_type, connected_node in connected_endpoints.items():
                        # Small weight for segment-to-segment transitions (intersection traversal)
                        intersection_weight = 5.0  # feet - cost of crossing an intersection
                        G.add_edge(current_node, connected_node, 
                                 weight=intersection_weight, 
                                 connection_type='intersection')
        
        # Step 3: Add points and connect them to segment endpoints
        points_added = 0
        points_skipped = 0
        
        for segment_idx, segment_points in points_by_segment.items():
            if segment_idx not in segment_endpoints:
                self.logger.warning(f"Segment {segment_idx} not in segment_endpoints, skipping {len(segment_points)} points")
                points_skipped += len(segment_points)
                continue
                
            segment_info = segment_endpoints[segment_idx]
            points_added += len(segment_points)
            start_node = segment_info['start_node']
            end_node = segment_info['end_node']
            
            for point in segment_points:
                point_id = point['point_id']
                distance_along = point.get('distance_along_segment', 0.0)
                total_length = point.get('segment_total_length', 100.0)
                
                # Add point as node
                G.add_node(point_id, node_type='point', 
                          segment_idx=segment_idx,
                          distance_along_segment=distance_along,
                          geometry=point['geometry'])
                points_added += 1
                
                # Connect point to segment start
                distance_to_start = distance_along
                if distance_to_start > 0:
                    G.add_edge(point_id, start_node, 
                             weight=distance_to_start, 
                             connection_type='point_to_segment')
                else:
                    # If point is at segment start, connect with small weight
                    G.add_edge(point_id, start_node, 
                             weight=0.1, 
                             connection_type='point_to_segment')
                
                # Connect point to segment end  
                distance_to_end = total_length - distance_along
                if distance_to_end > 0:
                    G.add_edge(point_id, end_node, 
                             weight=distance_to_end, 
                             connection_type='point_to_segment')
                else:
                    # If point is at segment end, connect with small weight  
                    G.add_edge(point_id, end_node, 
                             weight=0.1, 
                             connection_type='point_to_segment')
        
        # Step 4: Add point-to-point connections within segments
        for segment_idx, segment_points in points_by_segment.items():
            if len(segment_points) < 2:
                continue
                
            # Connect consecutive points along the segment
            for i in range(len(segment_points) - 1):
                point1 = segment_points[i]
                point2 = segment_points[i + 1]
                
                # Calculate distance between consecutive points along the segment
                distance = abs(point2['distance_along_segment'] - point1['distance_along_segment'])
                
                G.add_edge(point1['point_id'], point2['point_id'], 
                         weight=distance, 
                         connection_type='point_to_point')
        
        # Debug: Count different node types
        point_nodes = sum(1 for n, data in G.nodes(data=True) if data.get('node_type') == 'point')
        segment_nodes = sum(1 for n, data in G.nodes(data=True) if data.get('node_type') == 'segment_endpoint')
        
        self.logger.info(f"Created complete network graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        self.logger.info(f"  - Point nodes: {point_nodes}, Segment endpoint nodes: {segment_nodes}")
        self.logger.info(f"  - Points added: {points_added}, Points skipped: {points_skipped}")
        
        # Debug: Check if graph is connected
        if G.number_of_nodes() > 0:
            is_connected = nx.is_connected(G)
            num_components = nx.number_connected_components(G)
            self.logger.info(f"  - Graph connected: {is_connected}, Components: {num_components}")
        
        return G

    def _compute_sssp_nearest_neighbors(self, points: List[Dict], 
                                      network_graph: nx.Graph, 
                                      max_search_distance: float) -> List[Tuple]:
        """
        Use SSSP to find nearest neighbors for each point within the network.
        
        For each point, runs single_source_dijkstra_path_length to find all reachable
        points within max_search_distance, then selects nearest neighbors.
        """
        self.logger.info("Computing SSSP nearest neighbors with cuGraph acceleration...")
        
        all_connections = []
        processed_pairs = set()
        
        # Use cuGraph backend if available for SSSP acceleration
        import networkx as nx
        
        for i, point in enumerate(points):
            if i % 100 == 0:
                self.logger.info(f"Processing SSSP for point {i+1}/{len(points)}")
                
            point_id = point['point_id']
            
            if not network_graph.has_node(point_id):
                self.logger.warning(f"Point {point_id} not found in network graph")
                continue
            
            try:
                # Use NetworkX dijkstra with weight - cuGraph backend will accelerate automatically
                distances = nx.single_source_dijkstra_path_length(
                    network_graph, point_id, cutoff=max_search_distance, weight='weight'
                )
                
                # Debug: Log the first few points to see what's happening
                if i < 3:
                    self.logger.info(f"DEBUG: Point {point_id} found {len(distances)} reachable nodes within {max_search_distance}ft")
                    point_neighbors = [n for n, d in distances.items() if network_graph.nodes.get(n, {}).get('node_type') == 'point' and n != point_id]
                    self.logger.info(f"DEBUG: Of those, {len(point_neighbors)} are point nodes (excluding self)")
                
                # Find nearest neighbors (excluding the point itself)
                neighbors = []
                for neighbor_id, distance in distances.items():
                    # Skip if it's the same point
                    if neighbor_id == point_id:
                        continue
                        
                    # Skip if it's not a point node (segment endpoints)
                    neighbor_data = network_graph.nodes.get(neighbor_id, {})
                    if neighbor_data.get('node_type') != 'point':
                        continue
                    
                    neighbors.append((neighbor_id, distance))
                
                # Sort by distance and take the closest neighbors
                neighbors.sort(key=lambda x: x[1])
                
                # For now, connect to the K nearest neighbors (where K is small)
                max_neighbors_per_point = 4  # Limit connections to avoid dense graph
                
                for neighbor_id, distance in neighbors[:max_neighbors_per_point]:
                    # Avoid duplicate edges (A->B and B->A)
                    pair = tuple(sorted([point_id, neighbor_id]))
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        all_connections.append((point_id, neighbor_id, distance))
                        
            except Exception as e:
                self.logger.warning(f"SSSP failed for point {point_id}: {e}")
                continue
        
        self.logger.info(f"SSSP computation complete: found {len(all_connections)} nearest neighbor connections")
        return all_connections




    
    def _extract_segment_connectivity_from_graph(self, segments_gdf: gpd.GeoDataFrame) -> Dict[int, List[int]]:
        """
        Extract segment-to-segment connectivity from the pre-computed NetworkX graph.
        
        This method reads the segment connectivity that was already computed when
        the NetworkX graph was created, avoiding redundant computation.
        
        Parameters:
        -----------
        segments_gdf : gpd.GeoDataFrame
            Segment network with LineString geometries
            
        Returns:
        --------
        Dict[int, List[int]]
            Dictionary mapping segment_idx -> [list of connected segment indices]
        """
        # Return cached connectivity if available
        if self._cached_segment_connectivity is not None:
            self.logger.info(f"Using cached segment connectivity for {len(self._cached_segment_connectivity)} segments")
            return self._cached_segment_connectivity
        
        self.logger.info("Extracting segment connectivity from NetworkX graph...")
        
        # For a segments-based graph created with momepy.gdf_to_nx, 
        # nodes represent segments and edges represent connectivity
        segment_connectivity = defaultdict(list)
        
        # The graph created in step 4 uses momepy.gdf_to_nx with approach='primal'
        # In this approach, each segment becomes a node and connections are edges
        for edge in self.segment_graph.edges():
            segment1_idx, segment2_idx = edge
            
            # Add bidirectional connectivity
            segment_connectivity[segment1_idx].append(segment2_idx)
            segment_connectivity[segment2_idx].append(segment1_idx)
        
        # Convert to regular dict and remove duplicates
        result = {}
        for segment_idx, connected_segments in segment_connectivity.items():
            result[segment_idx] = list(set(connected_segments))
        
        # Cache the result for future use
        self._cached_segment_connectivity = result
        
        self.logger.info(f"Extracted and cached connectivity for {len(result)} segments from NetworkX graph")
        return result

        

    def _build_graph_from_overlay(self, points: List[Dict], overlay_result: Dict) -> nx.Graph:
        """
        Build NetworkX graph from spatial overlay results.
        
        Creates the actual graph structure using the connection rules determined
        by spatial overlay between points and segments.
        
        Parameters:
        -----------
        points : List[Dict]
            Uniformly sampled points
        overlay_result : Dict
            Results from spatial overlay containing connection rules
            
        Returns:
        --------
        nx.Graph
            Complete NetworkX graph with points as nodes and spatial connections as edges
        """
        self.logger.info("Creating NetworkX graph from network topology...")
        
        # Create graph
        if NX_CUGRAPH_AVAILABLE:
            G = nx.Graph()
        else:
            G = nx.Graph()
        
        # Add nodes (points)
        for point in points:
            G.add_node(
                point['point_id'],
                geometry=point['geometry'],
                x=point['geometry'].x,
                y=point['geometry'].y,
                source_segment_idx=point.get('source_segment_idx'),
                distance_along_segment=point.get('distance_along_segment', 0.0),
                is_intersection=point.get('is_intersection', False),
                parent_id=point.get('parent_id', ''),
                # Add other point attributes as needed
            )
        
        # Add edges from directional nearest neighbor connections
        edge_count = 0
        for point1_id, point2_id, distance in overlay_result['all_connections']:
            G.add_edge(point1_id, point2_id, 
                      distance=distance, 
                      connection_type='nearest_neighbor')
            edge_count += 1
        
        self.logger.info(f"Added {edge_count} edges from nearest neighbor connections")
        
        # Keep as regular NetworkX graph - let NetworkX handle GPU backend dispatch automatically
        if NX_CUGRAPH_AVAILABLE:
            self.logger.info("nx-cugraph available - NetworkX will use GPU backend for supported algorithms")
        
        self.logger.info(f"NetworkX graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def get_network_statistics(self, graph: nx.Graph) -> Dict:
        """
        Get basic network statistics from the segment graph.
        
        Parameters:
        -----------
        graph : nx.Graph
            The segment network graph
            
        Returns:
        --------
        Dict
            Dictionary containing network statistics
        """
        self.logger.info("Computing network statistics...")
        
        try:
            stats = {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'is_connected': nx.is_connected(graph) if graph.number_of_nodes() > 0 else False,
                'connected_components': nx.number_connected_components(graph) if graph.number_of_nodes() > 0 else 0
            }
            
            if graph.number_of_nodes() > 0:
                # Check if this is a CudaGraph (nx-cugraph) or regular NetworkX
                graph_type = type(graph).__name__
                
                if 'Cuda' in graph_type:
                    # nx-cugraph objects have limited API
                    stats['graph_type'] = 'nx-cugraph (GPU)'
                    stats['average_degree'] = 'N/A (CudaGraph)'
                    stats['density'] = 'N/A (CudaGraph)'
                else:
                    # Regular NetworkX - full API available
                    stats['graph_type'] = 'NetworkX (CPU)'
                    
                    # These operations might be expensive for large graphs
                    if graph.number_of_nodes() < 100000:  # Only for reasonably sized graphs
                        stats['average_degree'] = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
                        stats['density'] = nx.density(graph)
                    else:
                        stats['average_degree'] = 'N/A (large graph)'
                        stats['density'] = 'N/A (large graph)'
            
            self.logger.info(f"Network stats: {stats['nodes']} nodes, {stats['edges']} edges, "
                           f"{stats['connected_components']} components, connected: {stats['is_connected']}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error computing network statistics: {e}")
            return {'error': str(e)}

