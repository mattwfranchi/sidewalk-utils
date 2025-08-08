# Sidewalk Network Generation Script 
# Claustrophobic Streets 
# Developer: @pendulating 

# (0a) import libraries & global constants 
import geopandas as gpd 
import pandas as pd 
import numpy as np 
import os 
import sys 
import networkx as nx
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from collections import defaultdict
import pygeoops
import libpysal
from tqdm import tqdm 
tqdm.pandas(desc="Splitting sidewalk planimetric at pedestrian ramps")


PROJ_CRS="EPSG:3627"


# (0b) import modules 
sys.path.append("../../../")
from utils.logger import get_logger
from utils.timer import time_it

log = get_logger(__name__)
log.setLevel("INFO")

from graph_gen.nyc.geo_processor_base import GeoDataProcessor
geobase = GeoDataProcessor()
from graph_gen.nyc.sidewalk_network.io import SIDEWALK_PLANIMETRIC, PEDESTRIAN_RAMPS, PARKS_PROPERTIES, STREET_CENTERLINES, SIDEWALK_CENTERLINES

# RAPIDS 
try: 
    import cudf as cf
    import cuspatial as cs
    import cupy
    import nx_cugraph as nx_cg
    import cugraph as cg
    log.success("RAPIDS libraries imported successfully")
    RAPIDS=True
except ImportError as e:
    log.error(f"Error importing RAPIDS libraries: {e}")
    sys.exit(1)


# DEBUGGING FLAG STATES 
RAPIDS=True
CENTERLINES_GEN=False

# (0c) load data 
@time_it()
def load_sidewalk_planimetric():
    sidewalk_planimetric = geobase.read_geodataframe(SIDEWALK_PLANIMETRIC)
    # explode multi-polygons into single polygons. 
    sidewalk_planimetric = sidewalk_planimetric.explode(index_parts=False)
    sidewalk_planimetric = geobase.ensure_crs(sidewalk_planimetric, PROJ_CRS)
    geobase.inspect_geometry_types(sidewalk_planimetric)
    return sidewalk_planimetric

sidewalk_planimetric = load_sidewalk_planimetric()
# Maintain a stable positional index to align with cuSpatial polygon_index (0..n-1)
sidewalk_planimetric = sidewalk_planimetric.copy().reset_index(drop=True)

if CENTERLINES_GEN:
    sidewalk_planimetric_centerlines = sidewalk_planimetric.geometry.progress_apply(lambda x: pygeoops.centerline(x))
    sidewalk_planimetric_centerlines.to_frame().to_parquet("sidewalk_planimetric_centerlines.parquet")

@time_it()
def sidewalk_planimetric_on_gpu(sidewalk_planimetric: gpd.GeoDataFrame) -> cs.GeoDataFrame:
    if RAPIDS:
        sidewalk_planimetric_gpu = geobase.gpd_gdf_to_cuspatial_gdf(sidewalk_planimetric)
    return sidewalk_planimetric_gpu

sidewalk_planimetric_gpu = sidewalk_planimetric_on_gpu(sidewalk_planimetric)

sidewalk_planimetric_global_bounds = sidewalk_planimetric.geometry.total_bounds
global_minx, global_miny, global_maxx, global_maxy = sidewalk_planimetric_global_bounds
log.info(f"Sidewalk planimetric global bounds: {sidewalk_planimetric_global_bounds}")

sidewalk_planimetric_gpu_bounding_boxes = cs.polygon_bounding_boxes(sidewalk_planimetric_gpu.geometry)
log.info(f"Sidewalk planimetric GPU bounding boxes: {len(sidewalk_planimetric_gpu_bounding_boxes)}")

@time_it()
def load_pedestrian_ramps():
    pedestrian_ramps = geobase.read_geodataframe(PEDESTRIAN_RAMPS)
    pedestrian_ramps = geobase.ensure_crs(pedestrian_ramps, PROJ_CRS)

    # collect ramps that are within 3 meters of eachother, and consolidate them into a single geometry at the midpoint of the points. 
    ramp_coords = np.array([(x, y) for x, y in zip(pedestrian_ramps.geometry.x, pedestrian_ramps.geometry.y)])
    cluster = DBSCAN(eps=3, min_samples=2, metric='euclidean')
    labels = cluster.fit_predict(ramp_coords)

    # IMPORTANT: Do NOT collapse all noise points (label == -1) into a single centroid
    # Treat clusters (label >= 0) by their centroid, and keep noise points as-is
    unique_geometries = []
    for label in sorted(set(labels)):
        indices_for_label = np.where(labels == label)[0]
        if label == -1:
            # Keep each noise point individually
            for idx in indices_for_label:
                unique_geometries.append(Point(ramp_coords[idx]))
        else:
            pts = [Point(ramp_coords[idx]) for idx in indices_for_label]
            unique_geometries.append(MultiPoint(pts).centroid)

    ramps_dissolved = gpd.GeoDataFrame(geometry=unique_geometries, crs=PROJ_CRS)
    log.info(f"Pedestrian ramps input: {len(pedestrian_ramps)} → after clustering: {len(ramps_dissolved)} (noise kept individually)")




    geobase.inspect_geometry_types(ramps_dissolved)
    return ramps_dissolved

pedestrian_ramps = load_pedestrian_ramps()

@time_it()
def pedestrian_ramps_on_gpu(pedestrian_ramps: gpd.GeoDataFrame) -> cs.GeoDataFrame:
    if RAPIDS:
        pedestrian_ramps_gpu = geobase.gpd_gdf_to_cuspatial_gdf(pedestrian_ramps)
    return pedestrian_ramps_gpu

pedestrian_ramps_gpu = pedestrian_ramps_on_gpu(pedestrian_ramps)

pedestrian_ramps_gpu_keys_to_points,pedestrian_ramps_gpu_quadtree = cs.quadtree_on_points(
    points=pedestrian_ramps_gpu.geometry,
    x_min=global_minx,
    x_max=global_maxx,
    y_min=global_miny,
    y_max=global_maxy,
    scale=max(global_maxx - global_minx, global_maxy - global_miny) // (1 << 4),
    max_depth=4, 
    max_size=128
)
log.success("Created pedestrian ramps quadtree.")

@time_it()
def load_park_properties():
    park_properties = geobase.read_geodataframe(PARKS_PROPERTIES)
    park_properties = geobase.ensure_crs(park_properties, PROJ_CRS)
    geobase.inspect_geometry_types(park_properties)
    return park_properties

park_properties = load_park_properties()

@time_it()
def park_properties_on_gpu(park_properties: gpd.GeoDataFrame) -> cs.GeoDataFrame:   
    if RAPIDS:
        park_properties_gpu = geobase.gpd_gdf_to_cuspatial_gdf(park_properties)
    return park_properties_gpu

park_properties_gpu = park_properties_on_gpu(park_properties)

@time_it()
def load_street_centerlines():

    street_centerlines = geobase.read_geodataframe(STREET_CENTERLINES)
    street_centerlines = geobase.ensure_crs(street_centerlines, PROJ_CRS)
    geobase.inspect_geometry_types(street_centerlines)
    return street_centerlines

street_centerlines = load_street_centerlines()

@time_it()
def find_street_intersections(street_centerlines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Find all intersection points between any pair of street centerlines using R-tree spatial indexing.
    
    Args:
        street_centerlines: GeoDataFrame containing street centerline geometries
        
    Returns:
        GeoDataFrame with intersection points as Point geometries
    """
    log.info(f"Finding intersections between {len(street_centerlines)} street centerlines using R-tree indexing...")
    
    from shapely.geometry import Point
    from shapely.ops import unary_union
    import rtree
    
    intersection_points = []
    
    # Create spatial index for efficient intersection queries
    spatial_index = rtree.index.Index()
    
    # Add all street geometries to the spatial index
    for idx, geom in enumerate(street_centerlines.geometry):
        spatial_index.insert(idx, geom.bounds)
    
    # Find potential intersections using spatial index
    processed_pairs = set()
    
    for i, street1 in enumerate(street_centerlines.geometry):
        # Get potential candidates using spatial index
        candidates = list(spatial_index.intersection(street1.bounds))
        
        for j in candidates:
            if i == j:
                continue
                
            # Create unique pair identifier to avoid duplicate processing
            pair_id = tuple(sorted([i, j]))
            if pair_id in processed_pairs:
                continue
            processed_pairs.add(pair_id)
            
            street2 = street_centerlines.iloc[j].geometry
            
            try:
                # Find intersection between the two streets
                intersection = street1.intersection(street2)
                
                # Handle different types of intersection results
                if intersection.is_empty:
                    continue
                elif intersection.geom_type == 'Point':
                    intersection_points.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    intersection_points.extend(list(intersection.geoms))
                elif intersection.geom_type == 'LineString':
                    # If streets overlap, we might get a LineString
                    # We'll take the start and end points
                    coords = list(intersection.coords)
                    if len(coords) > 0:
                        intersection_points.append(Point(coords[0]))
                        if len(coords) > 1:
                            intersection_points.append(Point(coords[-1]))
                elif intersection.geom_type == 'MultiLineString':
                    # Handle multiple line segments
                    for line in intersection.geoms:
                        coords = list(line.coords)
                        if len(coords) > 0:
                            intersection_points.append(Point(coords[0]))
                            if len(coords) > 1:
                                intersection_points.append(Point(coords[-1]))
                                
            except Exception as e:
                log.warning(f"Error finding intersection between streets {i} and {j}: {e}")
                continue
    
    # Remove duplicate points (within a small tolerance)
    if intersection_points:
        # Convert to GeoDataFrame for easier deduplication
        temp_gdf = gpd.GeoDataFrame(geometry=intersection_points, crs=PROJ_CRS)
        
        # Buffer points slightly and dissolve to merge nearby points
        buffered_points = temp_gdf.geometry.buffer(1.0)  # 1 meter buffer
        dissolved = unary_union(buffered_points)
        
        # Extract centroids of dissolved areas as unique intersection points
        if dissolved.geom_type == 'Point':
            unique_points = [dissolved]
        elif dissolved.geom_type == 'MultiPolygon':
            unique_points = [poly.centroid for poly in dissolved.geoms]
        else:
            unique_points = [dissolved.centroid]
    else:
        unique_points = []
    
    log.success(f"Found {len(unique_points)} unique intersection points")
    
    intersections = gpd.GeoDataFrame(
        geometry=unique_points,
        crs=PROJ_CRS
    )
    
    return intersections

# Find all street intersections
#intersections = find_street_intersections(street_centerlines)
#intersections.to_parquet("street_centerlines_intersections.parquet")

@time_it()
def street_centerlines_on_gpu(street_centerlines: gpd.GeoDataFrame) -> cs.GeoDataFrame:
    if RAPIDS:
        street_centerlines_gpu = geobase.gpd_gdf_to_cuspatial_gdf(street_centerlines)
    return street_centerlines_gpu

street_centerlines_gpu = street_centerlines_on_gpu(street_centerlines)

@time_it()
def load_sidewalk_centerlines():

    sidewalk_centerlines = geobase.read_geodataframe(SIDEWALK_CENTERLINES)
    sidewalk_centerlines = geobase.ensure_crs(sidewalk_centerlines, PROJ_CRS)
    geobase.inspect_geometry_types(sidewalk_centerlines)
    return sidewalk_centerlines

sidewalk_centerlines = load_sidewalk_centerlines()

@time_it()
def sidewalk_centerlines_on_gpu(sidewalk_centerlines: gpd.GeoDataFrame) -> cs.GeoDataFrame:

    if RAPIDS:
        sidewalk_centerlines_gpu = geobase.gpd_gdf_to_cuspatial_gdf(sidewalk_centerlines)
    return sidewalk_centerlines_gpu

sidewalk_centerlines_gpu = sidewalk_centerlines_on_gpu(sidewalk_centerlines)


# (1) start with raw sidewalk geojson and nyc street centerlines. 

# (2) generate intersection corner segments: 




sidewalk_exteriors = sidewalk_planimetric.copy().reset_index(drop=True)
sidewalk_exteriors.geometry = sidewalk_exteriors.geometry.exterior
# SANITY CHECK: all geometries need to be closed, and have the same first and last coordinate. 
assert sidewalk_exteriors.geometry.apply(lambda x: x.coords[0] == x.coords[-1]).all(), "Sidewalk exteriors are not closed."
# convert LineString to Polygon. 
sidewalk_exteriors.geometry = sidewalk_exteriors.geometry.apply(lambda x: Polygon(x.coords))

log.info("Made version of sidewalk_planimetric with exteriors as Shapely Polygons.")

sidewalk_exteriors_gpu = geobase.gpd_gdf_to_cuspatial_gdf(sidewalk_exteriors)


# custom function using shapely.ops.split to split sidewalk_planimetric at corners defined by pedestrian_ramps.
@time_it()
def find_ramps_of_interest(sidewalk_exteriors_gpu: cs.GeoDataFrame, pedestrian_ramps_gpu: cs.GeoDataFrame) -> gpd.GeoDataFrame:
    try: 
        # SANITY CHECK: pedestrian_ramps_gpu.geometry is a cs.GeoSeries, and sidewalk_planimetric_gpu.geometry is also a cs.GeoSeries. 
        # This is a requirement for cuspatial.point_in_polygon. 
        assert isinstance(pedestrian_ramps_gpu.geometry, cs.GeoSeries), "pedestrian_ramps_gpu.geometry is not a cs.GeoSeries"
        assert isinstance(sidewalk_exteriors_gpu.geometry, cs.GeoSeries), "sidewalk_exteriors_gpu.geometry is not a cs.GeoSeries"


        print(sidewalk_exteriors_gpu.geometry.head())
        print(pedestrian_ramps_gpu.geometry.head())
        points_in_polygons = cs.point_in_polygon(pedestrian_ramps_gpu.geometry, sidewalk_exteriors_gpu.geometry)
        print(points_in_polygons.shape)
        print(points_in_polygons)

        # write points_in_polygons to a file. 
        points_in_polygons.columns = points_in_polygons.columns.astype(str)
        points_in_polygons.to_parquet("points_in_polygons.parquet")

    except Exception as e:
        log.error(f"Error in find_ramps_of_interest: {e}")
        exit()

@time_it()
def find_ramps_of_interest_quadtree(sidewalk_exteriors_gpu: cs.GeoDataFrame, sidewalk_exteriors_gpu_bounding_boxes: cs.GeoSeries, pedestrian_ramps_gpu: cs.GeoDataFrame, pedestrian_ramps_gpu_keys_to_points: cf.Series, pedestrian_ramps_gpu_quadtree: cf.DataFrame, global_minx: float, global_miny: float, global_maxx: float, global_maxy: float) -> gpd.GeoDataFrame:

    poly_quad_pairs = cs.join_quadtree_and_bounding_boxes(
        pedestrian_ramps_gpu_quadtree, 
        sidewalk_exteriors_gpu_bounding_boxes, 
        x_min=global_minx,
        x_max=global_maxx,
        y_min=global_miny,
        y_max=global_maxy,
        scale=max(global_maxx - global_minx, global_maxy - global_miny) // (1 << 4),
        max_depth=4,
    )
    log.info("Joined pedestrian ramps quadtree and sidewalk planimetric bounding boxes.")


    ramp_sidewalk_join = cs.quadtree_point_in_polygon(
        poly_quad_pairs, 
        pedestrian_ramps_gpu_quadtree,
        pedestrian_ramps_gpu_keys_to_points,
        pedestrian_ramps_gpu.geometry,
        sidewalk_exteriors_gpu.geometry,
    )
    log.success(f"Joined pedestrian ramps quadtree and sidewalk planimetric bounding boxes. {len(ramp_sidewalk_join)} rows.")

    return ramp_sidewalk_join



# Slightly buffer sidewalks for the point-in-polygon join to include boundary-adjacent ramps
sidewalk_exteriors_join = sidewalk_exteriors.copy().reset_index(drop=True)
sidewalk_exteriors_join.geometry = sidewalk_exteriors_join.geometry.buffer(0.75)
sidewalk_exteriors_join_gpu = geobase.gpd_gdf_to_cuspatial_gdf(sidewalk_exteriors_join)

# Use bounding boxes derived from the actual polygon series used in the join
sidewalk_exteriors_gpu_bounding_boxes = cs.polygon_bounding_boxes(sidewalk_exteriors_join_gpu.geometry)

# find_ramps_of_interest(sidewalk_exteriors_gpu, pedestrian_ramps_gpu)
ramp_lookup_table = find_ramps_of_interest_quadtree(
    sidewalk_exteriors_join_gpu,
    sidewalk_exteriors_gpu_bounding_boxes,
    pedestrian_ramps_gpu,
    pedestrian_ramps_gpu_keys_to_points,
    pedestrian_ramps_gpu_quadtree,
    global_minx,
    global_miny,
    global_maxx,
    global_maxy,
)
log.info(f"Ramp to sidewalk candidate pairs after quadtree join: {len(ramp_lookup_table)}")

# Standardize returned column names and compute diagnostics
try:
    # Some cuSpatial versions may use different column names
    col_map = {}
    for candidate, standard in {
        "poly_index": "polygon_index",
        "poly_idx": "polygon_index",
        "polygon_idx": "polygon_index",
        "point_idx": "point_index",
    }.items():
        if candidate in ramp_lookup_table.columns and standard not in ramp_lookup_table.columns:
            col_map[candidate] = standard
    if col_map:
        ramp_lookup_table = ramp_lookup_table.rename(columns=col_map)
    log.info(f"ramp_lookup_table columns: {list(ramp_lookup_table.columns)}")

    # Light-weight coverage stats
    try:
        ramp_lookup_pd = ramp_lookup_table.to_pandas()
    except Exception:
        ramp_lookup_pd = ramp_lookup_table
    if "polygon_index" in ramp_lookup_pd.columns:
        per_poly_counts = ramp_lookup_pd["polygon_index"].value_counts()
        matched_polys = len(per_poly_counts)
        total_polys = len(sidewalk_exteriors_join)
        coverage = 100.0 * matched_polys / max(total_polys, 1)
        log.info(f"Sidewalk polygons with ≥1 ramp: {matched_polys}/{total_polys} ({coverage:.1f}%)")
        if len(per_poly_counts) > 0:
            log.info(f"Ramp count per matched polygon: mean={per_poly_counts.mean():.2f}, max={per_poly_counts.max()}, median={per_poly_counts.median()}\nTop 5: {per_poly_counts.head(5).to_dict()}")
except Exception as e:
    log.warning(f"Diagnostics for ramp_lookup_table failed: {e}")


pedestrian_ramps.geometry = pedestrian_ramps.geometry.buffer(10)

assert pedestrian_ramps.geometry.is_valid.all(), f"Pedestrian ramps are not valid. {pedestrian_ramps.geometry}"

def split_sidewalk_at_ramps(sidewalk_block: pd.Series , ramp_lookup_table: cf.DataFrame, pedestrian_ramps_gpu_keys_to_points: cf.Series, pedestrian_ramps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    blocks_with_errors = False

    # Ensure valid and precision-snapped sidewalk geometry to avoid topology errors
    try:
        geom_sidewalk = sidewalk_block.geometry
        if not geom_sidewalk.is_valid:
            geom_sidewalk = shapely.make_valid(geom_sidewalk)
        geom_sidewalk = shapely.set_precision(geom_sidewalk, 0.01)
    except Exception as e:
        log.error(f"Failed to validate/snap sidewalk geometry {sidewalk_block.name}: {e}")
        return MultiPolygon(), True
    


    # first, get the indices of the sidewalk_exteriors_gpu that are within the ramp_lookup_table. 
    on_sidewalk_points_idx = ramp_lookup_table[ramp_lookup_table['polygon_index'] == sidewalk_block.name]

    if len(on_sidewalk_points_idx) == 0:
        # No ramps associated with this sidewalk polygon; return it unchanged as a MultiPolygon of one
        try:
            return MultiPolygon([sidewalk_block.geometry]), False
        except Exception as e:
            log.error(f"Error wrapping sidewalk polygon into MultiPolygon for {sidewalk_block.name}: {e}")
            return MultiPolygon(), True

    on_sidewalk_points_idx = pedestrian_ramps_gpu_keys_to_points.iloc[on_sidewalk_points_idx.point_index].to_numpy()

    on_sidewalk_points = pedestrian_ramps.iloc[on_sidewalk_points_idx]
    # Merge ramp buffers into a single valid union to avoid overlapping multipolygon issues
    try:
        ramp_polygons = [g for g in on_sidewalk_points.geometry.tolist() if isinstance(g, (Polygon, MultiPolygon))]
        if len(ramp_polygons) == 0:
            return MultiPolygon([geom_sidewalk]), False
        ramp_union = unary_union(ramp_polygons)
        # Ensure valid and precision-snap the union
        ramp_union = shapely.make_valid(ramp_union)
        ramp_union = shapely.set_precision(ramp_union, 0.01)
    except Exception as e:
        log.error(f"Error creating union of ramp buffers for sidewalk {sidewalk_block.name}: {e}")
        return MultiPolygon(), True


    # now, get intersection of on_sidewalk_geometries and sidewalk_block.geometry. 
    try:
        sidewalk_corners = shapely.intersection(geom_sidewalk, ramp_union)
    except Exception as e:
        # Retry with buffer(0) fallback
        try:
            sidewalk_corners = shapely.intersection(geom_sidewalk.buffer(0), ramp_union.buffer(0))
        except Exception as e2:
            log.error(f"Error in split_sidewalk_at_ramps sidewalk corners construction: {e2}. This can occur if the input geometry is invalid.")
            return MultiPolygon(), True

    if (type(sidewalk_corners) == MultiPolygon):
        sidewalk_corners = sidewalk_corners.geoms
    else:
        sidewalk_corners = [sidewalk_corners]

    try:
        split_sidewalk_block = shapely.difference(geom_sidewalk, ramp_union)
    except Exception as e:
        # Retry with buffer(0) fallback
        try:
            split_sidewalk_block = shapely.difference(geom_sidewalk.buffer(0), ramp_union.buffer(0))
        except Exception as e2:
            log.error(f"Error in split_sidewalk_at_ramps split sidewalk block construction: {e2}. This can occur if the input geometry is invalid.")
            return MultiPolygon(), True

    if (type(split_sidewalk_block) == MultiPolygon):
        split_sidewalk_block = split_sidewalk_block.geoms
    else:
        split_sidewalk_block = [split_sidewalk_block]
 
    # Collect only polygonal parts; accepts single geometry or sequences/GeometrySequence
    def polygonal_parts(obj):
        parts = []
        if obj is None:
            return parts
        # If it's a list/tuple or a GeometrySequence (shapely), iterate over items
        if isinstance(obj, (list, tuple)) or obj.__class__.__name__ == 'GeometrySequence':
            for item in obj:
                parts.extend(polygonal_parts(item))
            return parts
        # Otherwise, expect a shapely geometry
        geom = obj
        try:
            if geom.is_empty:
                return parts
            gtype = geom.geom_type
        except Exception:
            return parts
        if gtype == 'Polygon':
            parts.append(geom)
        elif gtype == 'MultiPolygon':
            parts.extend(list(geom.geoms))
        elif gtype == 'GeometryCollection':
            for sub in geom.geoms:
                parts.extend(polygonal_parts(sub))
        return parts

    try:
        corner_parts = polygonal_parts(sidewalk_corners)
        split_parts = polygonal_parts(split_sidewalk_block)
        all_parts = corner_parts + split_parts
        if len(all_parts) == 0:
            # If nothing resulted, keep original sidewalk geometry
            all_geos = MultiPolygon([geom_sidewalk])
        else:
            all_geos = MultiPolygon(all_parts)
    except Exception as e:
        log.error(f"Error in split_sidewalk_at_ramps assembling polygonal parts: {e}")
        return MultiPolygon(), True
    
    return all_geos, blocks_with_errors


# vectorize the function across the sidewalk_planimetric. 
result = sidewalk_planimetric.progress_apply(lambda x: split_sidewalk_at_ramps(x, ramp_lookup_table, pedestrian_ramps_gpu_keys_to_points, pedestrian_ramps), axis=1)
# unpack the result
result = pd.DataFrame(result.tolist(), columns=['geometry', 'blocks_with_errors'])
result = gpd.GeoDataFrame(result, geometry='geometry', crs=PROJ_CRS)
# drop blocks with errors. 
result = result[~result['blocks_with_errors']]

# explode MultiPolygons into individual polygon rows so counts reflect true segments
pre_explode_count = len(result)
result = result.explode(index_parts=False).reset_index(drop=True)
post_explode_count = len(result)
log.success(f"Split sidewalk segments: before explode rows={pre_explode_count}, after explode rows={post_explode_count}")

result = result.to_parquet("sidewalk_planimetric_split.parquet")





# (2b) draw rectangular spatial buffer of dim (street 1 width, street 2 width) around each intersection point. 
# (2c) clip the block-level sidewalk geojson by this buffer, resulting in a set of corner sidewalk segments. 
# (2d) load the nyc pedestrian ramps dataset. 
# (2e) if at least one pedestrian ramp falls within a corner sidewalk segment, tag it as 'confirmed'. 

# (3) generate pedestrian island segments: 


# (4) generate across-road connections. 
# (4a) find points where two streets intersect. 
# (4b) draw rectangular spatial buffer of dim (street 1 width, street 2 width) around each intersection point. 
# (4c) draw a connection between every contained combination of corner sidewalk segments & pedestrian island segments. 
# (4d) then, remove connections if they form a vector more than 15 degrees deviated from one of the street centerlines involved in the intersection point. 

# At this point, evaluate network connectivity. Should be highly connected, if not, something is wrong. 

# (5) overlay park zones, and remove computed segments that fall within them. 

