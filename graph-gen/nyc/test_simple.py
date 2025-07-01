#!/usr/bin/env python3
"""
Simple test script for intersection connectivity without external dependencies.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the fire module if it's not available
try:
    import fire
except ImportError:
    # Create a mock fire module
    class MockFire:
        def Fire(self, *args, **kwargs):
            pass
    fire = MockFire()

# Mock other dependencies if needed
try:
    from data.nyc.c import PROJ_FT
except ImportError:
    PROJ_FT = "EPSG:2263"  # Default NYC projection

try:
    from geo_processor_base import GeoDataProcessor
except ImportError:
    # Create a minimal mock
    class GeoDataProcessor:
        def __init__(self, name=None):
            self.logger = self
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}")

try:
    from user import INSTALL_DIR
except ImportError:
    INSTALL_DIR = "/tmp"

def test_intersection_connectivity():
    """Test intersection point connectivity with a simple example."""
    print("=== Testing Intersection Connectivity ===")
    
    # Create a simple intersection test case
    lines = [
        LineString([(0, 0), (10, 0)]),      # Horizontal line
        LineString([(10, 0), (20, 0)]),     # Connected horizontal line
        LineString([(10, 0), (10, 10)]),    # Vertical line from intersection
        LineString([(10, 10), (20, 10)]),   # Horizontal line at top
        LineString([(20, 0), (20, 10)]),    # Vertical line at right
    ]
    
    # Create GeoDataFrame
    test_gdf = gpd.GeoDataFrame({
        'id': range(len(lines)),
        'geometry': lines
    }, crs=PROJ_FT)
    
    print(f"Created intersection test with {len(test_gdf)} line segments")
    
    # Test segment adjacency computation
    print("\nTesting segment adjacency...")
    from segmentize_utils import compute_adjacency
    
    test_gdf_with_adj = compute_adjacency(test_gdf, tolerance=1.0, logger=None)
    if test_gdf_with_adj is not None:
        adj_counts = test_gdf_with_adj['adjacent_ids'].apply(len)
        print(f"Segment adjacency: avg={adj_counts.mean():.2f}, max={adj_counts.max()}")
        
        # Show adjacency details
        for idx, row in test_gdf_with_adj.iterrows():
            print(f"  Segment {idx}: adjacent to {row['adjacent_ids']}")
    
    # Test point segmentization
    print("\nTesting point segmentization...")
    from segmentize_utils import segmentize_and_extract_points
    
    segmentized_points = segmentize_and_extract_points(test_gdf, distance=2.0, logger=None)
    if segmentized_points is not None:
        print(f"Segmentized into {len(segmentized_points)} points")
        
        # Test point adjacency computation
        print("\nTesting point adjacency...")
        from segmentize_cpu import SegmentizeCPUFallbacks
        
        # Create a mock parent for the CPU fallbacks
        class MockParent:
            def __init__(self):
                self.logger = self
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): print(f"DEBUG: {msg}")
        
        cpu_fallbacks = SegmentizeCPUFallbacks(MockParent())
        
        points_with_adj = cpu_fallbacks.compute_point_adjacency_parallel(
            segmentized_points, 
            test_gdf_with_adj, 
            point_distance_threshold=3.0
        )
        
        if points_with_adj is not None and 'point_adjacent_ids' in points_with_adj.columns:
            # Analyze intersection points
            adj_counts = points_with_adj['point_adjacent_ids'].apply(len)
            intersection_points = adj_counts >= 3
            
            print(f"Point adjacency results:")
            print(f"  Total points: {len(points_with_adj)}")
            print(f"  Intersection points (3+ connections): {intersection_points.sum()}")
            print(f"  Average connections: {adj_counts.mean():.2f}")
            print(f"  Max connections: {adj_counts.max()}")
            print(f"  Isolated points: {(adj_counts == 0).sum()}")
            
            # Show intersection point details
            if intersection_points.sum() > 0:
                print("  Intersection point details:")
                for idx in points_with_adj[intersection_points].index:
                    point_adj = points_with_adj.loc[idx, 'point_adjacent_ids']
                    point_geom = points_with_adj.loc[idx, points_with_adj.geometry.name]
                    print(f"    Point {idx}: {len(point_adj)} connections at {point_geom}")
            
            # Test consolidation
            print("\nTesting point consolidation...")
            from segmentize_utils import consolidate_corner_points
            
            consolidated_points = consolidate_corner_points(points_with_adj, min_distance=1.0, logger=None)
            
            if consolidated_points is not None and 'point_adjacent_ids' in consolidated_points.columns:
                cons_adj_counts = consolidated_points['point_adjacent_ids'].apply(len)
                cons_intersection_points = cons_adj_counts >= 3
                
                print(f"After consolidation:")
                print(f"  Total points: {len(consolidated_points)}")
                print(f"  Intersection points: {cons_intersection_points.sum()}")
                print(f"  Average connections: {cons_adj_counts.mean():.2f}")
                print(f"  Max connections: {cons_adj_counts.max()}")
                print(f"  Isolated points: {(cons_adj_counts == 0).sum()}")
                
                # Check if intersection connectivity was preserved
                if cons_intersection_points.sum() > 0:
                    print("  Consolidated intersection point details:")
                    for idx in consolidated_points[cons_intersection_points].index:
                        point_adj = consolidated_points.loc[idx, 'point_adjacent_ids']
                        point_geom = consolidated_points.loc[idx, consolidated_points.geometry.name]
                        print(f"    Point {idx}: {len(point_adj)} connections at {point_geom}")

def main():
    """Main function to run the test."""
    print("Simple Intersection Connectivity Test")
    print("=" * 40)
    
    test_intersection_connectivity()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 