# %% [markdown]
# ## Claustrophobic Streets
# Work by Matt Franchi
# 
# 
# ### Notebook Description: 
# In this notebook, we construct an aggregated map of 'space_taken' in New York City, defined as various types of street furniture that were encountered on walks around various neighborhoods. A limitation of this notebook is that we are unable to capture certain types of space_taken with data, ie. streetside dining. Furthermore, some data lacks precision at the 50-foot granularity with which we segmentize sidewalks (ie., sidewalk scaffolding). Nonetheless, we hope this notebook accurately captures the density of street space_taken density across the entirety of New York City. All datasets used come from NYC OpenData. 
# 
# ### Performance Notes: 
# We run this notebook on a compute node with 64GB RAM and 8 CPUs. 
# 
# 
# 

# %% [markdown]
# ### Module Imports 

# %%
import pandas as pd 
import geopandas as gpd 
import osmnx as ox 
from shapely import wkt 

import matplotlib.pyplot as plt 

from tqdm import tqdm 
from glob import glob 


import logging 
# add logger name, time and date to logger messages
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("street-space_taken")



logger.info("Initialization complete.")

# %%
# the projected CRS to convert coordinates into much more accurate positioning data, using the Long Island State Plane
PROJ_CRS = 'EPSG:2263'

# the maximum distance to search for a nearby street segment. Since we segmentize by 50 feet, we can search within 25 feet
MAX_DISTANCE=25

CUTOFF= pd.to_datetime("2023-12-02")



# %% [markdown]
# ### Loading Segmentized NYC Sidewalk Graph 

# %%




# %%



nyc_sidewalks = gpd.read_parquet("../data/nyc/processed/segmentized_with_widths.parquet")
print(nyc_sidewalks)

# stored in feet, need to convert to meters
nyc_sidewalks['shape_width'] = nyc_sidewalks['width'] * 0.3048


# set first column to be named 'point index' 
nyc_sidewalks.columns = ['point_index'] + list(nyc_sidewalks.columns[1:])

# %%
#nyc_sidewalks = gpd.GeoDataFrame(nyc_sidewalks, geometry=nyc_sidewalks['geometry'].apply(wkt.loads), crs='EPSG:2263')

# %%
# we buffer each point by 25 feet, creating a 50-diameter circle centered at the point. This captures nearby space_taken. 
nyc_sidewalks['geometry'] = nyc_sidewalks['geometry'].buffer(MAX_DISTANCE, cap_style=3)

# %%


# %% [markdown]
# ### Load NYC Neighborhoods

# %%
ntas_nyc = gpd.read_file("../data/nyc/sf/nynta2020_24b/nynta2020.shp").to_crs(PROJ_CRS)
un = ntas_nyc[ntas_nyc.NTAName == 'United Nations']
un_crop = gpd.sjoin(nyc_sidewalks, un, predicate='intersects')

fig, ax = plt.subplots(figsize=(10,10))

un_crop.plot(ax=ax, color='blue', alpha=0.5)
un.boundary.plot(ax=ax, color='red')


# %%
# sort neighbrhoods by area
ntas_nyc['area'] = ntas_nyc.geometry.area
ntas_nyc.sort_values('area', ascending=True)

# %% [markdown]
# ### Sidewalk Scaffolding 

# %%
# read DoB active scaffolding permits 
scaffolding_permits = pd.read_csv("../data/nyc/sf/dob_active_sheds.csv", engine='pyarrow')
scaffolding_permits = gpd.GeoDataFrame(scaffolding_permits, geometry=gpd.points_from_xy(scaffolding_permits['Longitude Point'], scaffolding_permits['Latitude Point']), crs='EPSG:4326')
scaffolding_permits = scaffolding_permits.to_crs(PROJ_CRS)

# %%
scaffolding_permits['First Permit Date']  = pd.to_datetime(scaffolding_permits['First Permit Date'])

scaffolding_permits = scaffolding_permits[scaffolding_permits['First Permit Date'] <= CUTOFF]
scaffolding_permits['First Permit Date'].describe()


# %% [markdown]
# ### Bus Stop Shelters

# %%
# read bus stop shelters 
bus_stop_shelters = gpd.read_file("../data/nyc/sf/Bus Stop Shelters.geojson").to_crs(PROJ_CRS)
bus_stop_shelters['latitude'] = bus_stop_shelters['latitude'].astype(float)
bus_stop_shelters['longitude'] = bus_stop_shelters['longitude'].astype(float)

# Bus stop installation date is not present, so filtering is out-of-scope

# %% [markdown]
# ### Trash Cans / Waste Baskets 

# %%
# load trash cans 
trash_cans = gpd.read_file("../data/nyc/sf/DSNY Litter Basket Inventory_20240525.geojson").to_crs(PROJ_CRS)
trash_cans['longitude'] = trash_cans.geometry.centroid.to_crs('EPSG:4326').x
trash_cans['latitude'] = trash_cans.geometry.centroid.to_crs('EPSG:4326').y

# trash can installation date is not present, so filtering is out-of-scope

# %% [markdown]
# ### LinkNYC Kiosks

# %%
# load linknyc
linknyc = gpd.read_file("../data/nyc/sf/LinkNYC_Kiosk_Locations_20240525.csv")
linknyc = gpd.GeoDataFrame(linknyc, geometry=gpd.points_from_xy(linknyc['Longitude'], linknyc['Latitude']), crs='EPSG:4326').to_crs(PROJ_CRS)

linknyc['Installation Complete'] = pd.to_datetime(linknyc['Installation Complete'])
linknyc = linknyc[linknyc['Installation Complete'] <= CUTOFF]
linknyc['Installation Complete'].describe()


# %% [markdown]
# ### Bicycle Parking Shelters 

# %%
# load bicycle parking shelters 
bicycle_parking_shelters = gpd.read_file("../data/nyc/sf/Bicycle Parking Shelters.geojson").to_crs(PROJ_CRS)
bicycle_parking_shelters['build_date'] = pd.to_datetime(bicycle_parking_shelters['build_date'])
bicycle_parking_shelters = bicycle_parking_shelters[bicycle_parking_shelters['build_date'] <= CUTOFF]
bicycle_parking_shelters['build_date'].describe()


# %% [markdown]
# ### Bicycle Racks

# %%
# load bicycle racks 
bicycle_racks = gpd.read_file("../data/nyc/sf/Bicycle Parking.geojson").to_crs(PROJ_CRS)
bicycle_racks['date_inst'] = pd.to_datetime(bicycle_racks['date_inst'])
bicycle_racks = bicycle_racks[bicycle_racks['date_inst'] <= CUTOFF]
bicycle_racks['date_inst'].describe()



# %% [markdown]
# ### CityBench

# %%
# load citybench
citybench = pd.read_csv("../data/nyc/sf/City_Bench_Locations__Historical__20240525.csv")
citybench = gpd.GeoDataFrame(citybench, geometry=gpd.points_from_xy(citybench['Longitude'], citybench['Latitude']), crs='EPSG:4326').to_crs(PROJ_CRS)
citybench['Installati'] = pd.to_datetime(citybench['Installati'])
citybench = citybench[citybench['Installati'] <= CUTOFF]
citybench['Installati'].describe()


# %% [markdown]
# ### Street Trees 

# %%
# load trees 
trees = pd.read_csv("../data/nyc/sf/Forestry_Tree_Points.csv", engine='pyarrow')
trees = gpd.GeoDataFrame(trees, geometry=wkt.loads(trees['Geometry']), crs='EPSG:4326').to_crs(PROJ_CRS)
trees['CreatedDate'] = pd.to_datetime(trees['CreatedDate'])
trees = trees[trees['CreatedDate'] <= CUTOFF]
trees['CreatedDate'].describe()



# %% [markdown]
# ### News Stands

# %%
# load newsstands 
newsstands = pd.read_csv("../data/nyc/sf/NewsStands.csv", engine='pyarrow')
newsstands = gpd.GeoDataFrame(newsstands, geometry=wkt.loads(newsstands['the_geom']), crs='EPSG:4326').to_crs(PROJ_CRS)
newsstands['Built_Date'] = pd.to_datetime(newsstands['Built_Date'])
newsstands = newsstands[newsstands['Built_Date'] <= CUTOFF]
newsstands['Built_Date'].describe() 

# %% [markdown]
# ### Parking Meters

# %%
# load parking meters 
parking_meters = pd.read_csv("../data/nyc/sf/Parking_Meters_Locations_and_Status_20240604.csv")
parking_meters = gpd.GeoDataFrame(parking_meters, geometry=wkt.loads(parking_meters['Location']), crs='EPSG:4326').to_crs(PROJ_CRS)

# parking meter installation date is not present, so filtering is out-of-scope

# %% [markdown]
# ### Fire Hydrants

# %%
# load hydrants 
hydrants = gpd.read_file("../data/nyc/sf/NYCDEP Citywide Hydrants.geojson").to_crs(PROJ_CRS) 

# hydrant installation date is not present, so filtering is out-of-scope


# %% [markdown]
# ### Street Signs 

# %%
# load street signs 
street_signs = pd.read_csv("../data/nyc/sf/Street_Sign_Work_Orders_20240721.csv", engine='pyarrow')

# only keep 'Current' record type 
street_signs = street_signs[street_signs['record_type'] == 'Current']
street_signs['order_completed_on_date'] = pd.to_datetime(street_signs['order_completed_on_date'])
street_signs = street_signs[street_signs['order_completed_on_date'] <= CUTOFF]
street_signs = gpd.GeoDataFrame(street_signs, geometry=gpd.points_from_xy(street_signs['sign_x_coord'], street_signs['sign_y_coord']), crs='EPSG:2263')
street_signs['order_completed_on_date'].describe()

# %% [markdown]
# ### Bollards 

# %%
# load bollards 
bollards = pd.read_csv("../data/nyc/sf/Traffic_Bollards_Tracking_and_Installations_20240721.csv", engine='pyarrow')
bollards['Date'] = pd.to_datetime(bollards['Date'])
bollards = bollards[bollards['Date'] <= CUTOFF]
bollards['Date'].describe()

# we choose not to process bollards, as locations need to be geocoded. Latitude/Longitude is not present in the dataset.

# %% [markdown]
# ### Subset of Street Poles that are Leased by Telecommunications Companies

# %%
# load mobile telecommunications-leased street poles dataset 
mobile_telecom_poles = pd.read_csv("../data/nyc/sf/Mobile_Telecommunications_Franchise_Pole_Reservation_Locations_20240721.csv", engine='pyarrow')
mobile_telecom_poles['Installation Date'] = pd.to_datetime(mobile_telecom_poles['Installation Date'])
mobile_telecom_poles = mobile_telecom_poles[mobile_telecom_poles['Installation Date'] <= CUTOFF]

mobile_telecom_poles = gpd.GeoDataFrame(mobile_telecom_poles, geometry=gpd.points_from_xy(mobile_telecom_poles['Longitude'], mobile_telecom_poles['Latitude']), crs='EPSG:4326').to_crs(PROJ_CRS)

mobile_telecom_poles['Installation Date'].describe()

# plottable, but unsure how many poles are leased, compared to total distribution of poles. With this uncertainty, we choose not to process this dataset.


# %% [markdown]
# ### Spatial Joining of space_takens to Sidewalk Graph 

# %%
# sjoin nearest bus stops and trash cans to sidewalk
len_before = len(nyc_sidewalks)
bus_stop_shelters = gpd.sjoin(nyc_sidewalks, bus_stop_shelters, )
logger.info(f"Missing {len(bus_stop_shelters[bus_stop_shelters['index_right'].isna()])} bus stop shelters.")

# %%
# sjoin nearest trash cans to sidewalk
len_before = len(trash_cans)
trash_cans = gpd.sjoin(nyc_sidewalks, trash_cans, )
logger.info(f"Removed {len_before - len(trash_cans)} trash cans that are not on sidewalks.")

# %%
# sjoin nearest linknyc to sidewalk
len_before = len(linknyc)
linknyc = gpd.sjoin(nyc_sidewalks, linknyc, )
logger.info(f"LinkNYC: {len_before} -> {len(linknyc)}")

# %%
# sjoin nearest citybench 
len_before = len(citybench)
citybench = gpd.sjoin(nyc_sidewalks, citybench, )
logger.info(f"Citybench: {len_before} -> {len(citybench)}")

# %%
# sjoint nearest bicycle parking shelters to sidewalk
len_before = len(bicycle_parking_shelters)
bicycle_parking_shelters = gpd.sjoin(nyc_sidewalks, bicycle_parking_shelters, )
logger.info(f"Bicycle Parking Shelters: {len_before} -> {len(bicycle_parking_shelters)}")

# %%
# sjoin nearest bicycle racks to sidewalk
len_before = len(bicycle_racks)
bicycle_racks = gpd.sjoin(nyc_sidewalks, bicycle_racks, )
logger.info(f"Bicycle Racks: {len_before} -> {len(bicycle_racks)}")

# %%
# sjoin nearest trees to sidewalk
len_before = len(trees)
trees = gpd.sjoin(nyc_sidewalks, trees, )
logger.info(f"Trees: {len_before} -> {len(trees)}")


# %%
# sjoin nearest newsstands to sidewalk
len_before = len(newsstands)
newsstands = gpd.sjoin(nyc_sidewalks, newsstands, )
logger.info(f"Newsstands: {len_before} -> {len(newsstands)}")


# %%
BUFFER=100 
# buffer scaffolding_permits points, then sjoin to sidewalks
scaffolding_permits.geometry = scaffolding_permits.geometry.buffer(BUFFER)
scaffolding_permits = gpd.sjoin(nyc_sidewalks, scaffolding_permits, predicate='intersects')

# %%
# sjoin nearest parking meters to sidewalk
len_before = len(parking_meters)
parking_meters = gpd.sjoin(nyc_sidewalks, parking_meters, )
logger.info(f"Parking Meters: {len_before} -> {len(parking_meters)}")


# %%
# sjoin nearest hydrants to sidewalk
len_before = len(hydrants)
hydrants = gpd.sjoin(nyc_sidewalks, hydrants, )
logger.info(f"Hydrants: {len_before} -> {len(hydrants)}")

# %%
# sjoin nearest street signs to sidewalk
len_before = len(street_signs)
street_signs = gpd.sjoin(nyc_sidewalks, street_signs, )
logger.info(f"Street Signs: {len_before} -> {len(street_signs)}")


# %% [markdown]
# ### Aggregating Counts by Point on Sidewalk Graph 

# %%
# now, get number of bus stops, trash cans, linknyc, citybench, bicycle parking shelters, and bicycle racks per sidewalk
bus_stop_counts = bus_stop_shelters.groupby('point_index').size().reset_index(name='bus_stop_count').fillna(0)
trash_can_counts = trash_cans.groupby('point_index').size().reset_index(name='trash_can_count').fillna(0)
linknyc_counts = linknyc.groupby('point_index').size().reset_index(name='linknyc_count').fillna(0)
citybench_counts = citybench.groupby('point_index').size().reset_index(name='citybench_count').fillna(0)
bicycle_parking_shelter_counts = bicycle_parking_shelters.groupby('point_index').size().reset_index(name='bicycle_parking_shelter_count').fillna(0)
bicycle_rack_counts = bicycle_racks.groupby('point_index').size().reset_index(name='bicycle_rack_count').fillna(0)
tree_counts = trees.groupby('point_index').size().reset_index(name='tree_count').fillna(0)
newsstand_counts = newsstands.groupby('point_index').size().reset_index(name='newsstand_count').fillna(0)
parking_meter_counts = parking_meters.groupby('point_index').size().reset_index(name='parking_meter_count').fillna(0)
hydrant_counts = hydrants.groupby('point_index').size().reset_index(name='hydrant_count').fillna(0)
street_sign_counts = street_signs.groupby('point_index').size().reset_index(name='street_sign_count').fillna(0)


# %%
# merge scaffolding in 
scaffolding_counts = scaffolding_permits.groupby('point_index').size().reset_index(name='scaffolding_permit_count').fillna(0)

# %%
# merge counts to nyc_sidewalks
nyc_sidewalks = nyc_sidewalks.merge(bus_stop_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(trash_can_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(linknyc_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(citybench_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(bicycle_parking_shelter_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(bicycle_rack_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(tree_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(newsstand_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(parking_meter_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(hydrant_counts, on='point_index', how='left')
nyc_sidewalks = nyc_sidewalks.merge(street_sign_counts, on='point_index', how='left')


# %%
# merge scaffolding in 
nyc_sidewalks = nyc_sidewalks.merge(scaffolding_counts, on='point_index', how='left')

# %% [markdown]
# ### Final Cleanup and Weighting

# %%
nyc_sidewalks = nyc_sidewalks.fillna(0)

# %%
nyc_sidewalks.describe([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99])

# %%
# naive weights based on predicted area of different space_takens 
weights = { 
    # from https://www.nycstreetdesign.info/furniture/bus-stop-shelter#:~:text=Narrow%3A%2014'%20L%20x%203,x%208'%2D11%22%20H
    'bus_stop_count': 70,
     # from https://onlyny.com/products/dsny-home-waste-basket?variant=40281355190356
    'trash_can_count': 4, 
    # https://www.nycstreetdesign.info/furniture/linknyc-kiosk
    'linknyc_count': 2.67, 
    # https://www.nycstreetdesign.info/furniture/citybench
    'citybench_count': 12.8125,
    'bicycle_parking_shelter_count': 2,
    # from https://www.nycstreetdesign.info/furniture/cityrack, but not very precise
    'bicycle_rack_count': 1.5,
    # can we compute this at the individual level? 
    'tree_count': 9,
    # from https://www.nycstreetdesign.info/furniture/newsstand, using large 
    'newsstand_count': 72, 
    # from https://www.nycstreetdesign.info/furniture/munimeter
    'parking_meter_count': 2.05,
    # this one is difficult... will come back to this. 
    'scaffolding_permit_count': 9,
    # from http://www.firehydrant.org/pictures/nyc.html. IS THIS RIGHT? 
    'hydrant_count': 0.08,
    # rough estimate
    'street_sign_count': 0.05,
}

# %%
# create a 'space_taken' metric that is the sum of all street space_taken features
nyc_sidewalks['space_taken'] = 0
for feature, weight in weights.items():
    # convert weight from sqft to sqm
    nyc_sidewalks['space_taken'] += nyc_sidewalks[feature] * weight * 0.092903

print(nyc_sidewalks['space_taken'].describe([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]))

# %%
# clamp distribution to 5th and 95th percentile
#nyc_sidewalks['space_taken'] = nyc_sidewalks['space_taken'].clip(lower=nyc_sidewalks['space_taken'].quantile(0.01), upper=nyc_sidewalks['space_taken'].quantile(0.99))

# %%

# %%
# now, compute 'available navigable space'. 
# we define this as the total sidewalk area represented by a point N (so, the area of N)
# minus the space_taken by N.

POINT_LENGTH = 50 * 0.3048
nyc_sidewalks['total_space'] = nyc_sidewalks['shape_width'] * POINT_LENGTH
nyc_sidewalks['available_space'] = nyc_sidewalks['total_space'] - nyc_sidewalks['space_taken']

# clip at 0 
nyc_sidewalks['available_space'] = nyc_sidewalks['available_space'].clip(lower=0.01)


print(nyc_sidewalks['available_space'].describe().apply(lambda x: format(x, 'f')))

# %% [markdown]
# ### Visualization and Saving Data 

# %%
# map sidewalk and color by space_taken 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

fig, ax = plt.subplots(figsize=(20, 20))
nyc_sidewalks.plot(column='available_space', ax=ax, legend=True, cmap='cividis', markersize=0.25, 
                  legend_kwds={'label': "Available, Navigable Sidewalk Space", 'orientation': 'horizontal', 
                              'shrink': 0.5, 'pad': 0.01})
ax.set_axis_off()

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Save the visualization
plt.savefig("figures/available_space_nyc.png", dpi=300, bbox_inches='tight', pad_inches=0)
logger.info("Visualization saved to figures/available_space_nyc.png")

# %%
# Save data as parquet format
output_dir = "../data/nyc/claustrophobia"
os.makedirs(output_dir, exist_ok=True)

# Save as Parquet with WKB encoding for geometry
parquet_path = os.path.join(output_dir, "nyc_sidewalks_space.parquet")
logger.info(f"Saving data to Parquet: {parquet_path}")
try:
    nyc_sidewalks.to_parquet(parquet_path, compression='snappy', index=False)
    logger.info(f"Successfully saved {len(nyc_sidewalks)} records to {parquet_path}")
except Exception as e:
    logger.error(f"Failed to save parquet file: {e}")
    # Try alternative approach with explicit WKB conversion if needed
    try:
        logger.info("Attempting to save with explicit WKB encoding...")
        from shapely import wkb
        # Clone dataframe to avoid modifying original
        nyc_sidewalks_wkb = nyc_sidewalks.copy()
        # Convert geometry to WKB
        nyc_sidewalks_wkb['geometry'] = nyc_sidewalks_wkb.geometry.apply(lambda geom: wkb.dumps(geom))
        nyc_sidewalks_wkb.to_parquet(parquet_path, compression='snappy', index=False)
        logger.info(f"Successfully saved with explicit WKB encoding to {parquet_path}")
    except Exception as e2:
        logger.error(f"Failed to save with explicit WKB encoding: {e2}")

logger.info("Data export completed")

# %%



