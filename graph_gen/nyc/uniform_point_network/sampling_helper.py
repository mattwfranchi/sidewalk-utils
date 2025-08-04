import geopandas as gpd
import pandas as pd
import numpy as np
import cupy as cp
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from typing import Dict, List
from collections import defaultdict
import sys
sys.path.append('/share/ju/sidewalk_utils')
from utils.logger import get_logger

class SamplingHelper:
    """Helper class for point sampling and filtering operations."""
    
    def __init__(self):
        self.logger = get_logger("SamplingHelper")
        
    def load_and_process_pedestrian_ramps(self, pedestrian_ramps_path: str, 
                                         segments_gdf: gpd.GeoDataFrame) -> List[Dict]:
        """Load pedestrian ramps and create initial intersection points."""
        if not pedestrian_ramps_path:
            pedestrian_ramps_path = "/share/ju/sidewalk_utils/data/nyc/geo/nyc_pedestrian_ramps.geojson"
        
        try:
            self.logger.info(f"Loading pedestrian ramps from: {pedestrian_ramps_path}")
            ramps_gdf = gpd.read_file(pedestrian_ramps_path)
            
            if ramps_gdf.crs != segments_gdf.crs:
                ramps_gdf = ramps_gdf.to_crs(segments_gdf.crs)
            
            # Filter ramps to neighborhood area
            ramps_gdf = self._filter_pedestrian_ramps(ramps_gdf, segments_gdf)
            
            # Prepare segments for spatial join
            segments_with_idx = segments_gdf.copy()
            segments_with_idx['segment_idx'] = segments_with_idx.index
            
            # Perform spatial join to find nearest segment for each ramp
            joined = gpd.sjoin_nearest(
                ramps_gdf, 
                segments_with_idx, 
                how='left',
                distance_col='distance_to_segment'
            )
            
            # Remove failed joins
            joined = joined.dropna(subset=['parent_id'])
            
            # Convert to list of dictionaries
            ramp_points = []
            for idx, (_, row) in enumerate(joined.iterrows()):
                ramp_point = {
                    'geometry': row.geometry,
                    'point_id': f"ramp_{idx}",
                    'is_pedestrian_ramp': True,
                    'is_intersection': True,
                    'parent_id': row['parent_id'],
                    'source_segment_idx': row['segment_idx'],
                    'distance_to_segment': row['distance_to_segment'],
                    'distance_along_segment': 0.0,
                    'segment_total_length': 0.0,
                    'position_ratio': 0.0,
                    'buffer_zone': None,
                    'network_neighbors': [],
                }
                ramp_points.append(ramp_point)
            
            self.logger.info(f"Processed {len(ramp_points)} pedestrian ramp points")
            return ramp_points
            
        except Exception as e:
            self.logger.error(f"Failed to load pedestrian ramps: {e}")
            return []
    
    def _filter_pedestrian_ramps(self, ramps_gdf: gpd.GeoDataFrame, 
                                segments_gdf: gpd.GeoDataFrame, 
                                buffer_distance: float = 100.0) -> gpd.GeoDataFrame:
        """Filter pedestrian ramps to only include those within buffer_distance of segments."""
        segment_tree = STRtree(list(segments_gdf.geometry))
        filtered_ramps = []
        
        for idx, ramp_geom in enumerate(ramps_gdf.geometry):
            nearby_segments = segment_tree.query(ramp_geom.buffer(buffer_distance))
            
            if len(nearby_segments) > 0:
                min_distance = float('inf')
                for segment_idx in nearby_segments:
                    segment_geom = segments_gdf.iloc[segment_idx].geometry
                    distance = ramp_geom.distance(segment_geom)
                    min_distance = min(min_distance, distance)
                
                if min_distance <= buffer_distance:
                    filtered_ramps.append(idx)
        
        if filtered_ramps:
            return ramps_gdf.iloc[filtered_ramps].reset_index(drop=True)
        else:
            return ramps_gdf.iloc[:0].copy()

    def generate_candidate_points(self, segments_gdf: gpd.GeoDataFrame,
                                 sampling_params: Dict,
                                 max_points_per_segment: int) -> List[Dict]:
        """Generate candidate points along segments using uniform sampling."""
        candidates = []
        interval = sampling_params['sampling_interval']
        
        for idx, (_, row) in enumerate(segments_gdf.iterrows()):
            if idx % 1000 == 0:
                self.logger.info(f"Processing segment {idx + 1}/{len(segments_gdf)}")
            
            try:
                # Use the actual segment_id from the row instead of enumerate index
                segment_id = row.get('segment_id', idx)  # Fallback to idx if segment_id missing
                segment_candidates = self._generate_points_along_linestring(
                    row.geometry, interval, segment_id, row, max_points_per_segment
                )
                candidates.extend(segment_candidates)
            except Exception as e:
                self.logger.error(f"Error processing segment {idx}: {e}")
                continue
        
        self.logger.info(f"Generated {len(candidates)} candidate points")
        return candidates
    
    def _generate_points_along_linestring(self, linestring: LineString,
                                        interval: float,
                                        segment_idx: int,
                                        segment_row: pd.Series,
                                        max_points: int) -> List[Dict]:
        """Generate points along a single LineString at specified intervals."""
        points = []
        length = linestring.length
        
        if length < interval:
            # For short segments, place one point at midpoint
            midpoint = linestring.interpolate(length / 2.0)
            record = self._create_point_record(
                midpoint, segment_idx, segment_row, length / 2.0, length, 0.5
            )
            points.append(record)
        else:
            # Generate points at regular intervals
            num_points = min(int(length / interval) + 1, max_points)
            
            for i in range(num_points):
                distance = min(i * interval, length)
                point_geom = linestring.interpolate(distance)
                
                record = self._create_point_record(
                    point_geom, segment_idx, segment_row, distance, length, distance / length
                )
                points.append(record)
        
        return points

    def _create_point_record(self, point_geom: Point,
                           segment_idx: int,
                           segment_row: pd.Series,
                           distance: float,
                           total_length: float,
                           position_ratio: float) -> Dict:
        """Create a standardized point record with metadata."""
        record = {
            'geometry': point_geom,
            'source_segment_idx': segment_idx,
            'distance_along_segment': distance,
            'segment_total_length': total_length,
            'position_ratio': position_ratio,
            'point_id': None,
            'is_intersection': False,
            'buffer_zone': None,
            'network_neighbors': [],
        }
        
        # Add segment attributes with prefix
        for col in segment_row.index:
            if col != 'geometry':
                record[f'source_{col}'] = segment_row[col]
        
        # Ensure parent_id is directly accessible
        if 'parent_id' in segment_row.index:
            record['parent_id'] = segment_row['parent_id']
        
        return record

    def apply_buffer_filtering(self, candidate_points: List[Dict],
                             pedestrian_ramps_points: List[Dict],
                             sampling_params: Dict) -> List[Dict]:
        """Apply parent_id-aware buffer filtering."""
        self.logger.info("Applying parent_id-aware buffer filtering...")
        
        # Start with pedestrian ramp points (automatically accepted)
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
            # Get accepted points for this parent_id only
            parent_accepted_points = [
                pt for pt in accepted_points 
                if pt.get('parent_id') == parent_id
            ]
            
            # Apply buffer filtering within this parent_id group
            filtered_parent_points = self._apply_buffer_filtering_gpu(
                parent_candidates, parent_accepted_points, sampling_params
            )
            
            # Add filtered points to the main accepted list
            for point in filtered_parent_points:
                point['point_id'] = len(accepted_points)
                accepted_points.append(point)
        
        self.logger.info(f"Buffer filtering complete: {len(accepted_points)} total points")
        return accepted_points

    def _apply_buffer_filtering_gpu(self, candidate_points: List[Dict],
                                  existing_points: List[Dict],
                                  sampling_params: Dict) -> List[Dict]:
        """GPU-based buffer filtering within a parent_id group."""
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
            current_point = sorted_candidate_coords[i:i+1]
            
            # Check distance to existing points
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

    def generate_intersection_centerlines(self, intersection_points: List[Dict], 
                                        crosswalk_distance: float = 100.0) -> gpd.GeoDataFrame:
        """Generate crosswalk centerlines between pairs of intersection points."""
        if not intersection_points:
            return gpd.GeoDataFrame(
                columns=['geometry', 'width', 'segment_id', 'segment_length', 
                        'segment_type', 'parent_id', 'is_crosswalk'], 
                crs="EPSG:2263"
            )
        
        new_centerlines = []
        processed_pairs = set()
        
        # Create spatial index for intersection points
        point_geoms = [point['geometry'] for point in intersection_points]
        point_tree = STRtree(point_geoms)
        
        # Find pairs of intersection points within crosswalk distance
        for i, point1 in enumerate(intersection_points):
            point1_geom = point1['geometry']
            point1_id = point1['point_id']
            
            nearby_points = point_tree.query(point1_geom.buffer(crosswalk_distance))
            
            for j in nearby_points:
                if i != j:
                    point2 = intersection_points[j]
                    point2_geom = point2['geometry']
                    point2_id = point2['point_id']
                    
                    distance = point1_geom.distance(point2_geom)
                    
                    if distance <= crosswalk_distance and distance > 0:
                        pair_id = tuple(sorted([point1_id, point2_id]))
                        
                        if pair_id not in processed_pairs:
                            processed_pairs.add(pair_id)
                            
                            # Create crosswalk centerline
                            point1_coords = (point1_geom.x, point1_geom.y)
                            point2_coords = (point2_geom.x, point2_geom.y)
                            crosswalk_centerline = LineString([point1_coords, point2_coords])
                            
                            centerline_record = {
                                'geometry': crosswalk_centerline,
                                'width': 10.0,
                                'segment_id': f"crosswalk_{point1_id}_to_{point2_id}",
                                'segment_length': crosswalk_centerline.length,
                                'segment_type': 'LineString',
                                'parent_id': f"crosswalk_{point1_id}_{point2_id}",
                                'source_intersection_id': point1_id,
                                'target_intersection_id': point2_id,
                                'crosswalk_distance': distance,
                                'is_crosswalk': True
                            }
                            
                            new_centerlines.append(centerline_record)
        
        if new_centerlines:
            crosswalk_centerlines = gpd.GeoDataFrame(new_centerlines, crs="EPSG:2263")
            self.logger.info(f"Generated {len(crosswalk_centerlines)} crosswalk centerlines")
            return crosswalk_centerlines
        else:
            return gpd.GeoDataFrame(
                columns=['geometry', 'width', 'segment_id', 'segment_length', 
                        'segment_type', 'parent_id', 'is_crosswalk'], 
                crs="EPSG:2263"
            ) 