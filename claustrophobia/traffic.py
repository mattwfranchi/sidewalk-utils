# %%
import pandas as pd 
import geopandas as gpd 
import numpy as np
from shapely import wkt 
from shapely.geometry import Point, Polygon, MultiPolygon

import matplotlib.pyplot as plt

from tqdm import tqdm 
from glob import glob 

import os
import sys 

# Import the logger
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)
logger.setLevel("DEBUG")

# %%
PROJ_CRS = 'EPSG:2263'
# increasing the distance acts as a smoothing kernal, as more points get to 'count' the traffic from an image
MAX_DISTANCE= 30 * 3.28084 # 25 meters in feet
FOV =  180 # Field of view in degrees
DEBUG_SAMPLE=False
USE_DIRECTION = False  # Set to False when using older data without camera_heading

logger.info(f"Using projection {PROJ_CRS} with max distance {MAX_DISTANCE:.2f} feet and FOV {FOV} degrees")

# %%
# load nyc ntas 
logger.info("Loading NYC NTA shapefile data...")
nyc_ntas = gpd.read_file("../data/nyc/sf/nynta2020_24b/nynta2020.shp").to_crs(PROJ_CRS)
logger.debug(f"Loaded {len(nyc_ntas)} NTA areas")
logger.debug(f"Available NTA names: {', '.join(nyc_ntas.NTAName.values[:5])}...")


# %%
logger.info("Loading traffic data from parquet files...")
traffic = pd.read_parquet("../data/nyc/processed/nexar2020_tuesthurs_subset_traffic.parquet")
traffic['image_name'] = traffic['image_name'].str.replace('.jpg', '', regex=False)
logger.debug(f"Loaded traffic data with {len(traffic)} records")

logger.info("Loading metadata from parquet files...")
md = gpd.read_parquet("/share/ju/periscopes/data/raw_export/nexar2020_geo_6077397_20250329_000919.parquet")
logger.debug(f"Loaded metadata with {len(md)} records")

logger.info("Merging traffic data with metadata...")
traffic = traffic.merge(md, left_on='image_name', right_on='frame_id', how='left')
# log columns in merged dataframe 
logger.debug(f"Columns in merged dataframe: {', '.join(traffic.columns)}")

# Log missing values instead of printing
missing_values = traffic.isna().sum()
logger.info("Missing values after merge:")
for column, count in missing_values[missing_values > 0].items():
    logger.info(f"  {column}: {count} missing values")

# Log summary statistics instead of printing
logger.info("Traffic data summary statistics:")
stats = traffic.describe().transpose()
for column in stats.index:
    logger.debug(f"  {column}: min={stats.loc[column, 'min']:.2f}, max={stats.loc[column, 'max']:.2f}, mean={stats.loc[column, 'mean']:.2f}")

# %%
# take a random sample 
if DEBUG_SAMPLE:
    logger.info(f"Taking random sample of 100,000 records for debug mode")
    traffic = traffic.sample(100000).reset_index(drop=True)

# %%
logger.info("Converting to GeoDataFrame...")
traffic = gpd.GeoDataFrame(traffic, geometry=traffic['geometry'])
logger.debug(f"Created GeoDataFrame with {len(traffic)} rows")

# Check if camera heading information is available
has_camera_heading = 'camera_heading' in traffic.columns
has_direction = 'direction' in traffic.columns

if not has_camera_heading and USE_DIRECTION:
    logger.warning("Camera heading column not found but USE_DIRECTION is True")
    logger.warning("Setting USE_DIRECTION to False")
    USE_DIRECTION = False

# Define semicircle function for direction-based filtering
def create_semicircle(point, heading, distance):
    # Convert the heading to radians
    heading_rad = np.deg2rad(heading)

    # Generate points for the semicircle
    num_points = 10  # Number of points to approximate the semicircle
    angles = np.linspace(heading_rad - np.pi / 2, heading_rad + np.pi / 2, num_points)
    
    semicircle_points = [point]
    for angle in angles:
        semicircle_points.append(Point(point.x + distance * np.cos(angle),
                                       point.y + distance * np.sin(angle)))
    semicircle_points.append(point)  # Close the semicircle

    # Create the semicircle polygon
    semicircle = Polygon(semicircle_points)

    return semicircle

# Load NYC sidewalk graph
logger.info("Loading NYC sidewalk graph...")
nyc_sidewalks = gpd.read_parquet("../data/nyc/processed/segmentized_with_widths.parquet")
logger.debug(f"Loaded {len(nyc_sidewalks)} sidewalk segments")

# Set first column to be named 'point index'
nyc_sidewalks.columns = ['point_index'] + list(nyc_sidewalks.columns[1:])

# Ensure both GeoDataFrames are in the same CRS
if traffic.crs != nyc_sidewalks.crs:
    logger.info(f"Converting traffic CRS to match sidewalk data CRS ({nyc_sidewalks.crs})")
    traffic = traffic.to_crs(nyc_sidewalks.crs)

if USE_DIRECTION:
    logger.info("Direction-based processing is enabled")
    if has_camera_heading:
        logger.debug(f"Camera heading stats: min={traffic['camera_heading'].min():.2f}, max={traffic['camera_heading'].max():.2f}, mean={traffic['camera_heading'].mean():.2f}")
    
    if has_direction:
        logger.debug(f"Direction value counts: {traffic['direction'].value_counts().to_dict()}")
        
        # Map direction column (NORTH_WEST, etc.) to a degree value 0-360 in new column
        dir_mapping = {
            'NORTH': 0,
            'NORTH_EAST': 45,
            'EAST': 90,
            'SOUTH_EAST': 135,
            'SOUTH': 180,
            'SOUTH_WEST': 225,
            'WEST': 270,
            'NORTH_WEST': 315
        }
        traffic['snapped_heading'] = traffic['direction'].map(dir_mapping)
        logger.debug(f"Created snapped_heading from direction with {traffic['snapped_heading'].notna().sum()} valid values")
        
        # Drop na rows on snapped_heading
        original_len = len(traffic)
        traffic = traffic.dropna(subset=['snapped_heading'])
        logger.info(f"Dropped {original_len - len(traffic)} rows missing direction information")
    
    # if original geometry column exists, swap it in and drop it 
    if 'original_geometry' in traffic.columns:
        traffic['geometry'] = traffic['original_geometry']
        traffic = traffic.drop(columns=['original_geometry'])

    # Store the original geometry
    traffic['original_geometry'] = traffic['geometry']

    # Create semicircle geometries
    logger.info(f"Creating semicircle geometries with {MAX_DISTANCE:.2f} feet distance and {FOV} degree FOV")
    traffic['geometry'] = traffic.apply(lambda row: create_semicircle(row['geometry'], row['camera_heading'], MAX_DISTANCE), axis=1)
    
    # Perform a spatial join to find all points in nyc_sidewalks within the cone
    logger.info("Performing spatial join between traffic and sidewalk data using directional cones...")
    traffic = gpd.sjoin(traffic, nyc_sidewalks, how='inner', predicate='intersects')
    
    # Restore the original geometry
    logger.debug("Restoring original point geometries...")
    traffic['geometry'] = traffic['original_geometry']
    
    # Drop the original_geometry column if no longer needed
    traffic = traffic.drop(columns=['original_geometry'])
    
else:
    logger.info("Direction-based processing is disabled - using all camera views without directional filtering")
    
    # When not using direction, just perform a buffer-based join
    logger.info(f"Creating circular buffers with {MAX_DISTANCE:.2f} feet radius")
    buffered_points = traffic.copy()
    buffered_points['geometry'] = buffered_points.geometry.buffer(MAX_DISTANCE)
    
    # Perform a spatial join to find all points in nyc_sidewalks within buffer
    logger.info("Performing spatial join between traffic and sidewalk data using circular buffers...")
    traffic = gpd.sjoin(buffered_points, nyc_sidewalks, how='inner', predicate='intersects')

# Log the number of joined points
logger.info(f"Spatial join completed with {len(traffic)} matching points")

# Log info about the num_pedestrians column
logger.info("Using num_pedestrians column for traffic calculations")
if 'num_pedestrians' in traffic.columns:
    logger.debug(f"num_pedestrians range: {traffic['num_pedestrians'].min()}-{traffic['num_pedestrians'].max()}, mean: {traffic['num_pedestrians'].mean():.2f}")
else:
    logger.error("Required num_pedestrians column not found in the dataset")
    logger.info(f"Available columns: {', '.join(traffic.columns)}")
    sys.exit(1)

# %%
# get average traffic per sidewalk
logger.info("Calculating median pedestrian traffic per sidewalk...")
avg_traffic_by_sidewalk = traffic.groupby('point_index')[['num_pedestrians']].median()
# rename column to meaningful name
avg_traffic_by_sidewalk.columns = ['pedestrians_median']
logger.debug(f"Created median traffic dataframe with shape {avg_traffic_by_sidewalk.shape}")

logger.info("Merging median traffic data with sidewalk information...")
avg_traffic_by_sidewalk = nyc_sidewalks.merge(avg_traffic_by_sidewalk, left_on='point_index', right_index=True, how='left')
logger.debug(f"Merged dataframe shape: {avg_traffic_by_sidewalk.shape}")

# %%
# get 95th percentile of traffic per sidewalk
logger.info("Calculating 95th percentile pedestrian traffic per sidewalk...")
traffic_95th = traffic.groupby('point_index')[['num_pedestrians']].quantile(0.95)
# rename column to meaningful name
traffic_95th.columns = ['pedestrians_95th']
traffic_95th = nyc_sidewalks.merge(traffic_95th, left_on='point_index', right_index=True, how='left')

# only keep relevant cols and point_index 
traffic_95th = traffic_95th[['point_index', 'pedestrians_95th']]
logger.debug(f"Created 95th percentile dataframe with shape {traffic_95th.shape}")

# merge 95th percentile data with average traffic data
logger.info("Merging median and 95th percentile traffic data...")
avg_traffic_by_sidewalk = avg_traffic_by_sidewalk.merge(traffic_95th, on='point_index', how='left')
logger.success(f"Successfully created complete traffic statistics dataframe with shape {avg_traffic_by_sidewalk.shape}")

# Check for missing data
zero_crowdedness_count = (avg_traffic_by_sidewalk['pedestrians_median'].isna()).sum()
total_points = avg_traffic_by_sidewalk.shape[0]
zero_crowdedness_percentage = zero_crowdedness_count / total_points * 100

if zero_crowdedness_percentage > 10:
    logger.warning(f"High percentage of missing crowdedness data: {zero_crowdedness_percentage:.2f}% of sidewalks ({zero_crowdedness_count}/{total_points})")
else:
    logger.info(f"Missing crowdedness data: {zero_crowdedness_percentage:.2f}% of sidewalks ({zero_crowdedness_count}/{total_points})")

# %%
# write average traffic to disk ewalk_all.parquet"
output_path = "../data/nyc/processed/avg_traffic_by_sidewalk_all.parquet"
logger.info(f"Writing traffic statistics to {output_path}")
# make a gdf from avg_traffic_by_sidewalk
avg_traffic_by_sidewalk = gpd.GeoDataFrame(avg_traffic_by_sidewalk, geometry=avg_traffic_by_sidewalk['geometry'])
# write to geoparquet 
avg_traffic_by_sidewalk.to_parquet(output_path, index=False)
logger.success(f"Data successfully saved to {output_path}")

# %%


