#!/usr/bin/env python3
"""
Test script for validating adjacency calculations in the sidewalk segmentization tool.
This script helps diagnose issues with missing adjacent points.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentize import SidewalkSegmentizer
from data.nyc.c import PROJ_FT

def create_test_data():
    """Create a simple test dataset to validate adjacency calculations."""
    
    # Create a simple network of connected line segments
    lines = [
        LineString([(0, 0), (10, 0)]),      # Horizontal line
        LineString([(10, 0), (20, 0)]),     # Connected horizontal line
        LineString([(10, 0), (10, 10)]),    # Vertical line from intersection
        LineString([(10, 10), (20, 10)]),   # Horizontal line at top
        LineString([(20, 0), (20, 10)]),    # Vertical line at right
    ]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'id': range(len(lines)),
        'geometry': lines
    }, crs=PROJ_FT)
    
    return gdf

def test_segment_adjacency():
    """Test segment-level adjacency computation."""
    print("=== Testing Segment Adjacency ===")
    
    # Create test data
    test_gdf = create_test_data()
    print(f"Created test dataset with {len(test_gdf)} line segments")
    
    # Create segmentizer
    segmentizer = SidewalkSegmentizer()
    
    # Test different tolerance values
    for tolerance in [0.1, 1.0, 5.0]:
        print(f"\nTesting with tolerance = {tolerance}")
        
        # Compute adjacency
        result = segmentizer.compute_segment_adjacency(test_gdf, tolerance=tolerance)
        
        if result is not None:
            # Analyze results
            adj_counts = result['adjacent_ids'].apply(len)
            print(f"  Average adjacent segments: {adj_counts.mean():.2f}")
            print(f"  Max adjacent segments: {adj_counts.max()}")
            print(f"  Isolated segments: {(adj_counts == 0).sum()}")
            
            # Show adjacency details
            for idx, row in result.iterrows():
                print(f"  Segment {idx}: adjacent to {row['adjacent_ids']}")
        else:
            print("  Failed to compute adjacency")

def test_point_adjacency():
    """Test point-level adjacency computation."""
    print("\n=== Testing Point Adjacency ===")
    
    # Create test data
    test_gdf = create_test_data()
    print(f"Created test dataset with {len(test_gdf)} line segments")
    
    # Create segmentizer
    segmentizer = SidewalkSegmentizer()
    
    # Compute segment adjacency first
    test_gdf = segmentizer.compute_segment_adjacency(test_gdf, tolerance=1.0)
    if test_gdf is None:
        print("Failed to compute segment adjacency")
        return
    
    # Segmentize into points
    segmentized_points = segmentizer.convert_segments_to_points(test_gdf, distance=2.0)
    print(f"Segmentized into {len(segmentized_points)} points")
    
    # Test different distance thresholds
    for threshold in [2.0, 5.0, 10.0]:
        print(f"\nTesting with point distance threshold = {threshold}")
        
        # Compute point adjacency
        result = segmentizer.compute_point_level_adjacency(
            segmentized_points, 
            test_gdf, 
            point_distance_threshold=threshold
        )
        
        if result is not None and 'point_adjacent_ids' in result.columns:
            # Analyze results
            adj_counts = result['point_adjacent_ids'].apply(len)
            print(f"  Average adjacent points: {adj_counts.mean():.2f}")
            print(f"  Max adjacent points: {adj_counts.max()}")
            print(f"  Isolated points: {(adj_counts == 0).sum()}")
            
            # Show some adjacency details
            print("  Sample adjacency relationships:")
            for i in range(min(5, len(result))):
                point_adj = result.iloc[i]['point_adjacent_ids']
                print(f"    Point {i}: adjacent to {len(point_adj)} points")
        else:
            print("  Failed to compute point adjacency")

def test_intersection_connectivity():
    """Test intersection point connectivity specifically."""
    print("\n=== Testing Intersection Connectivity ===")
    
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
    
    # Create segmentizer
    segmentizer = SidewalkSegmentizer()
    
    # Compute segment adjacency
    test_gdf = segmentizer.compute_segment_adjacency(test_gdf, tolerance=1.0)
    if test_gdf is None:
        print("Failed to compute segment adjacency")
        return
    
    # Segmentize into points
    segmentized_points = segmentizer.convert_segments_to_points(test_gdf, distance=2.0)
    print(f"Segmentized into {len(segmentized_points)} points")
    
    # Compute point adjacency
    points_with_adj = segmentizer.compute_point_level_adjacency(
        segmentized_points, 
        test_gdf, 
        point_distance_threshold=3.0
    )
    
    if points_with_adj is not None and 'point_adjacent_ids' in points_with_adj.columns:
        # Analyze intersection points
        adj_counts = points_with_adj['point_adjacent_ids'].apply(len)
        intersection_points = adj_counts >= 3
        
        print(f"Intersection analysis:")
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
        consolidated_points = segmentizer.merge_nearby_corner_points(points_with_adj, min_distance=1.0)
        
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

def validate_real_data(input_path):
    """Validate adjacency calculations on real data."""
    print(f"\n=== Validating Real Data: {input_path} ===")
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return
    
    # Create segmentizer
    segmentizer = SidewalkSegmentizer()
    
    # Load data
    print("Loading data...")
    sidewalks_data = segmentizer.read_geodataframe(input_path, crs=PROJ_FT, name="sidewalk data")
    
    if sidewalks_data is None:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(sidewalks_data)} sidewalk segments")
    
    # Test segment adjacency
    print("\nTesting segment adjacency...")
    sidewalks_with_adj = segmentizer.compute_segment_adjacency(sidewalks_data, tolerance=5.0)
    
    if sidewalks_with_adj is not None:
        adj_counts = sidewalks_with_adj['adjacent_ids'].apply(len)
        print(f"Segment adjacency stats:")
        print(f"  Average adjacent segments: {adj_counts.mean():.2f}")
        print(f"  Max adjacent segments: {adj_counts.max()}")
        print(f"  Isolated segments: {(adj_counts == 0).sum()} ({(adj_counts == 0).sum()/len(sidewalks_with_adj)*100:.1f}%)")
    
    # Test point adjacency
    print("\nTesting point adjacency...")
    segmentized_points = segmentizer.convert_segments_to_points(sidewalks_data, distance=50.0)
    print(f"Segmentized into {len(segmentized_points)} points")
    
    if segmentized_points is not None:
        points_with_adj = segmentizer.compute_point_level_adjacency(
            segmentized_points, 
            sidewalks_with_adj, 
            point_distance_threshold=55.0
        )
        
        if points_with_adj is not None and 'point_adjacent_ids' in points_with_adj.columns:
            adj_counts = points_with_adj['point_adjacent_ids'].apply(len)
            print(f"Point adjacency stats:")
            print(f"  Average adjacent points: {adj_counts.mean():.2f}")
            print(f"  Max adjacent points: {adj_counts.max()}")
            print(f"  Isolated points: {(adj_counts == 0).sum()} ({(adj_counts == 0).sum()/len(points_with_adj)*100:.1f}%)")

def main():
    """Main function to run all tests."""
    print("Adjacency Calculation Validation Tool")
    print("=" * 50)
    
    # Run synthetic tests
    test_segment_adjacency()
    test_point_adjacency()
    test_intersection_connectivity()
    
    # Run validation on real data if provided
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        validate_real_data(input_path)
    else:
        print("\nTo test with real data, provide an input file path:")
        print("python test_adjacency.py /path/to/your/sidewalk/data.geojson")

if __name__ == "__main__":
    main() 