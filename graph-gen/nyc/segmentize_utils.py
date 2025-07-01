import os
import geopandas as gpd
import numpy as np
import warnings
import pandas as pd
from shapely.strtree import STRtree
from geo_processor_base import GeoDataProcessor
from typing import Optional, List, Set, Dict, Any, Union
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import nearest_points

def log_message(logger: Optional[Any], level: str, message: str) -> None:
    """Helper function to handle logging with optional logger"""
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "success":
            logger.success(message)
    else:
        print(f"[{level.upper()}] {message}")

def segmentize_and_extract_points(gdf: gpd.GeoDataFrame, distance: float = 50, 
                                 logger: Optional[Any] = None) -> Optional[gpd.GeoDataFrame]:
    """
    Segmentize geometries and extract unique points
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Input geodataframe
    distance : float
        Distance between points in segmentation
    logger : Logger, optional
        Logger object for status updates
        
    Returns:
    --------
    GeoDataFrame
        Geodataframe with point geometries
    """
    try:
        if gdf is None or gdf.empty:
            log_message(logger, "error", "Cannot segmentize empty geodataframe")
            return None
            
        log_message(logger, "info", f"Segmentizing geometries at {distance} foot intervals")
        
        # Log input geometry statistics
        total_length = gdf.geometry.length.sum()
        avg_length = gdf.geometry.length.mean()
        expected_points = int(total_length / distance)
        log_message(logger, "info", f"Total length to segmentize: {total_length:.2f} units")
        log_message(logger, "info", f"Average feature length: {avg_length:.2f} units")
        log_message(logger, "info", f"Expected point count (estimate): ~{expected_points} points")
        
        try:
            # Segmentize and extract points
            segmentized = gdf.segmentize(distance).extract_unique_points()
            
            # Explode multipoint geometries into individual points
            log_message(logger, "info", "Exploding multipoint geometries")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                segmentized = segmentized.explode(index_parts=True)
                
            actual_points = len(segmentized)
            point_ratio = actual_points / expected_points
            log_message(logger, "info", f"Generated {actual_points} points ({point_ratio:.2f}x the estimate)")
            
            # Log spatial distribution stats
            if len(segmentized) > 0:
                bounds = segmentized.total_bounds
                area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                point_density = actual_points / area if area > 0 else 0
                log_message(logger, "info", f"Point density: {point_density:.6f} points per square unit")
                
                # Calculate nearest-neighbor distances
                try:
                    sample_size = min(1000, len(segmentized))
                    if sample_size > 10:  # Only calculate for reasonably sized samples
                        sample = segmentized.sample(sample_size) if sample_size < len(segmentized) else segmentized
                        distances = []
                        for idx, point in sample.items():
                            if len(sample) > 1:  # Need at least 2 points
                                others = sample[sample.index != idx]
                                min_dist = others.distance(point).min()
                                distances.append(min_dist)
                        
                        if distances:
                            avg_nn_dist = sum(distances) / len(distances)
                            log_message(logger, "info", f"Average nearest neighbor distance (sample): {avg_nn_dist:.2f} units")
                            log_message(logger, "info", f"Min/Max nearest neighbor distance: {min(distances):.2f}/{max(distances):.2f} units")
                except Exception as e:
                    log_message(logger, "warning", f"Could not calculate point distribution stats: {e}")
            
            if segmentized.empty:
                log_message(logger, "error", "No points were generated during segmentization")
                return None
                
            return segmentized
            
        except Exception as e:
            log_message(logger, "error", f"Error during segmentization: {e}")
            return None
            
    except Exception as e:
        log_message(logger, "error", f"Error in segmentize_and_extract_points: {e}")
        return None

def prepare_segmentized_dataframe(segmentized_points: gpd.GeoDataFrame, 
                                 source_gdf: gpd.GeoDataFrame,
                                 logger: Optional[Any] = None) -> Optional[gpd.GeoDataFrame]:
    """
    Prepare the final segmentized dataframe with attributes
    
    Parameters:
    -----------
    segmentized_points : GeoDataFrame
        Segmentized point geometries
    source_gdf : GeoDataFrame
        Original geodataframe with attributes
    logger : Logger, optional
        Logger object for status updates
        
    Returns:
    --------
    GeoDataFrame
        Final geodataframe with points and attributes
    """
    try:
        if segmentized_points is None or segmentized_points.empty:
            log_message(logger, "error", "Segmentized data is empty")
            return None
            
        if source_gdf is None or source_gdf.empty:
            log_message(logger, "error", "Source data is empty")
            return None
            
        log_message(logger, "info", "Preparing final segmentized dataframe")
        log_message(logger, "info", f"Source data contains {len(source_gdf)} features with {len(source_gdf.columns)} attributes")
        
        # Save the original geometry column and CRS
        original_geom_col = segmentized_points.geometry.name
        original_crs = segmentized_points.crs
        
        # Get the geometries directly from the GeoDataFrame's geometry
        geometries = segmentized_points.geometry.copy()
        log_message(logger, "info", f"Saved {len(geometries)} geometries from geometry column")
        
        # Reset index to get level_0 and level_1 columns
        log_message(logger, "info", "Resetting index to get level_0 and level_1 columns")
        segmentized_df = segmentized_points.reset_index()
        
        # Merge with original dataframe
        try:
            log_message(logger, "info", "Merging with original dataframe")
            result = segmentized_df.merge(
                source_gdf, 
                left_on='level_0',
                right_index=True
            )
            log_message(logger, "info", f"Merged dataframe has {len(result)} rows and {len(result.columns)} columns")
        except Exception as e:
            log_message(logger, "error", f"Merge failed: {e}")
            return None
        
        # Drop unnecessary columns in one step
        drop_cols = ['level_0', 'level_1']
        
        # Also drop any geometry column from the merged result to avoid conflicts
        geom_columns_to_drop = []
        for col in result.columns:
            if col != original_geom_col and isinstance(result[col].iloc[0], (Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon)):
                geom_columns_to_drop.append(col)
        
        drop_cols.extend(geom_columns_to_drop)
        result = result.drop(columns=drop_cols, errors='ignore')
        
        # Create final GeoDataFrame with the saved point geometries
        result = gpd.GeoDataFrame(
            result, 
            geometry=geometries,  # Directly use the saved geometries
            crs=original_crs
        )
        
        log_message(logger, "info", f"Final dataframe has {len(result)} points with {len(result.columns)-1} attributes")
        return result
        
    except Exception as e:
        log_message(logger, "error", f"Error in prepare_segmentized_dataframe: {e}")
        log_message(logger, "error", f"Error details: {str(e)}")
        return None

def compute_adjacency(gdf: gpd.GeoDataFrame, tolerance: float = 0.1, 
                     logger: Optional[Any] = None) -> Optional[gpd.GeoDataFrame]:
    """
    Compute adjacency relationships between geometries using vectorized operations
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Input geodataframe with geometries
    tolerance : float
        Distance tolerance for considering geometries adjacent (in feet)
    logger : Logger, optional
        Logger object for status updates
        
    Returns:
    --------
    GeoDataFrame
        Input geodataframe with additional adjacency columns
    """
    try:
        if gdf is None or gdf.empty:
            log_message(logger, "error", "Cannot compute adjacency for empty geodataframe")
            return None
            
        log_message(logger, "info", f"Computing adjacency relationships with tolerance {tolerance}")
        
        # Create a spatial index
        log_message(logger, "info", "Building spatial index")
        sindex = gdf.sindex
        
        # Use a more comprehensive approach to find adjacent geometries
        log_message(logger, "info", f"Finding adjacent features with comprehensive spatial queries")
        
        # Initialize adjacency dictionary
        adjacency_dict = {idx: [] for idx in gdf.index}
        
        # Process each geometry to find its neighbors
        total_features = len(gdf)
        log_message(logger, "info", f"Processing {total_features} features for adjacency")
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Use intersection query with buffered geometry to find nearby features
            buffered_geom = geom.buffer(tolerance)
            candidate_indices = list(sindex.intersection(buffered_geom.bounds))
            
            # Check each candidate for actual adjacency
            for candidate_idx in candidate_indices:
                candidate_idx = gdf.index[candidate_idx]
                
                # Skip self
                if candidate_idx == idx:
                    continue
                    
                candidate_geom = gdf.loc[candidate_idx, 'geometry']
                
                # Check if geometries are actually within tolerance distance
                if geom.distance(candidate_geom) <= tolerance:
                    # Add bidirectional adjacency
                    if candidate_idx not in adjacency_dict[idx]:
                        adjacency_dict[idx].append(candidate_idx)
                    if idx not in adjacency_dict[candidate_idx]:
                        adjacency_dict[candidate_idx].append(idx)
            
            # Log progress every 10% of features
            if (gdf.index.get_loc(idx) + 1) % max(1, total_features // 10) == 0:
                progress = int(((gdf.index.get_loc(idx) + 1) / total_features) * 100)
                log_message(logger, "info", f"Adjacency progress: {progress}%")
        
        # Add adjacency information to the dataframe
        log_message(logger, "info", "Adding adjacency information to dataframe")
        gdf['adjacent_ids'] = gdf.index.map(adjacency_dict)
        gdf['adjacency_count'] = gdf['adjacent_ids'].apply(len)
        
        # Log adjacency statistics
        feature_count = len(gdf)
        avg_adjacency = gdf['adjacency_count'].mean()
        max_adjacency = gdf['adjacency_count'].max()
        isolated_count = (gdf['adjacency_count'] == 0).sum()
        
        log_message(logger, "info", f"Adjacency statistics:")
        log_message(logger, "info", f"  - Average adjacent features: {avg_adjacency:.2f}")
        log_message(logger, "info", f"  - Maximum adjacent features: {max_adjacency}")
        log_message(logger, "info", f"  - Isolated features: {isolated_count} ({isolated_count/feature_count*100:.1f}%)")
        
        return gdf
        
    except Exception as e:
        log_message(logger, "error", f"Error in compute_adjacency: {e}")
        return None

def consolidate_corner_points(gdf: gpd.GeoDataFrame, min_distance: float = 10, 
                             logger: Optional[Any] = None) -> gpd.GeoDataFrame:
    """
    Consolidate points that are too close together at corners.
    
    This function now preserves intersection connectivity by:
    1. Identifying intersection points (points with high connectivity)
    2. Merging nearby points while preserving their adjacency relationships
    3. Ensuring intersection points maintain their network connectivity
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Segmentized points with adjacency information
    min_distance : float
        Minimum distance between points (any closer will be consolidated)
    logger : Logger, optional
        Logger object for status updates
        
    Returns:
    --------
    GeoDataFrame
        Points with corner clusters consolidated while preserving connectivity
    """
    log_message(logger, "info", f"Consolidating points closer than {min_distance} feet")
    
    # Ensure we have a proper GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        log_message(logger, "error", f"Input is not a GeoDataFrame: {type(gdf)}")
        return gdf
    
    # Ensure the geometry column is properly set
    geometry_column = gdf._geometry_column_name
    if geometry_column is None:
        log_message(logger, "error", "No geometry column found in GeoDataFrame")
        return gdf
    
    log_message(logger, "debug", f"Using geometry column: '{geometry_column}'")
    
    # Check if we have adjacency information
    has_adjacency = 'point_adjacent_ids' in gdf.columns
    if has_adjacency:
        log_message(logger, "info", "Found adjacency information - preserving connectivity during consolidation")
    else:
        log_message(logger, "warning", "No adjacency information found - connectivity may be lost")
    
    # Create a copy to avoid modifying the original
    result = gdf.copy()
    
    # Build spatial index
    sindex = result.sindex
    
    # Track points to remove and their replacements
    points_to_remove: Set = set()
    point_replacements: Dict[Any, Any] = {}  # Maps removed point to replacement point
    
    # First pass: identify intersection points (high connectivity points)
    intersection_points = set()
    if has_adjacency:
        # Points with high connectivity are likely intersections
        high_connectivity_threshold = 3  # Points with 3+ connections are likely intersections
        for idx, row in result.iterrows():
            if len(row['point_adjacent_ids']) >= high_connectivity_threshold:
                intersection_points.add(idx)
        
        log_message(logger, "info", f"Identified {len(intersection_points)} potential intersection points")
    
    # Second pass: consolidate points while preserving intersection connectivity
    for idx, row in result.iterrows():
        # Skip if already marked for removal
        if idx in points_to_remove:
            continue
            
        # Access geometry safely using the column name
        point_geom = row[geometry_column]
        
        # Find nearby points
        point_buffer = point_geom.buffer(min_distance)
        possible_matches_idx = list(sindex.query(point_buffer))
        
        # Filter to points actually within distance
        close_points = []
        for match_idx in possible_matches_idx:
            match_idx = result.index[match_idx]
            if match_idx == idx:  # Skip self
                continue
            if match_idx in points_to_remove:  # Skip already marked points
                continue
                
            # Check actual distance using the column name to access geometry
            if point_geom.distance(result.loc[match_idx, geometry_column]) < min_distance:
                close_points.append(match_idx)
        
        # If we found close points, consolidate them
        if close_points:
            # Determine which point to keep (prefer intersection points)
            points_to_consider = [idx] + close_points
            best_point = idx
            
            # Prefer intersection points as the consolidation target
            for point in points_to_consider:
                if point in intersection_points:
                    best_point = point
                    break
            
            # If no intersection point, prefer the one with highest connectivity
            if best_point == idx and has_adjacency:
                max_connectivity = len(result.loc[idx, 'point_adjacent_ids'])
                for point in close_points:
                    connectivity = len(result.loc[point, 'point_adjacent_ids'])
                    if connectivity > max_connectivity:
                        max_connectivity = connectivity
                        best_point = point
            
            # Mark other points for removal
            points_to_remove.update([p for p in points_to_consider if p != best_point])
            
            # Record replacements
            for point in points_to_consider:
                if point != best_point:
                    point_replacements[point] = best_point
            
            # Merge adjacency information if available
            if has_adjacency:
                # Collect all unique adjacent points from all points being merged
                all_adjacent = set()
                for point in points_to_consider:
                    all_adjacent.update(result.loc[point, 'point_adjacent_ids'])
                
                # Remove references to points being removed
                all_adjacent = all_adjacent - set(points_to_remove)
                
                # Update the kept point's adjacency
                result.loc[best_point, 'point_adjacent_ids'] = list(all_adjacent)
    
    # Remove consolidated points
    log_message(logger, "info", f"Removing {len(points_to_remove)} duplicate points")
    result = result.drop(list(points_to_remove))
    
    # Update adjacency references to point to the correct consolidated points
    if has_adjacency and point_replacements:
        log_message(logger, "info", "Updating adjacency references to consolidated points")
        for idx, row in result.iterrows():
            updated_adjacent = []
            for adj_point in row['point_adjacent_ids']:
                # If this adjacent point was replaced, use the replacement
                if adj_point in point_replacements:
                    replacement = point_replacements[adj_point]
                    # If the replacement is also being removed, find the final replacement
                    while replacement in point_replacements:
                        replacement = point_replacements[replacement]
                    if replacement not in points_to_remove:
                        updated_adjacent.append(replacement)
                elif adj_point not in points_to_remove:
                    updated_adjacent.append(adj_point)
            
            # Remove duplicates and self-references
            updated_adjacent = list(set(updated_adjacent) - {idx})
            result.loc[idx, 'point_adjacent_ids'] = updated_adjacent
    
    # Update adjacency counts
    if has_adjacency:
        result['point_adjacency_count'] = result['point_adjacent_ids'].apply(len)
        
        # Log connectivity statistics
        avg_connectivity = result['point_adjacency_count'].mean()
        max_connectivity = result['point_adjacency_count'].max()
        isolated_count = (result['point_adjacency_count'] == 0).sum()
        
        log_message(logger, "info", f"After consolidation - avg connectivity: {avg_connectivity:.2f}, max: {max_connectivity}, isolated: {isolated_count}")
    
    log_message(logger, "info", f"Point consolidation complete: {len(gdf)} -> {len(result)} points")
    return result