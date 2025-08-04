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


PROJ_CRS="EPSG:3627"


# (0b) import modules 
sys.path.append("../../../")
from utils.logger import get_logger

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
except ImportError as e:
    log.error(f"Error importing RAPIDS libraries: {e}")
    sys.exit(1)


# (0c) load data 
sidewalk_planimetric = geobase.read_geodataframe(SIDEWALK_PLANIMETRIC)
sidewalk_planimetric = geobase.ensure_crs(sidewalk_planimetric, PROJ_CRS)

pedestrian_ramps = geobase.read_geodataframe(PEDESTRIAN_RAMPS)
pedestrian_ramps = geobase.ensure_crs(pedestrian_ramps, PROJ_CRS)

park_properties = geobase.read_geodataframe(PARKS_PROPERTIES)
park_properties = geobase.ensure_crs(park_properties, PROJ_CRS)

street_centerlines = geobase.read_geodataframe(STREET_CENTERLINES)
street_centerlines = geobase.ensure_crs(street_centerlines, PROJ_CRS)

sidewalk_centerlines = geobase.read_geodataframe(SIDEWALK_CENTERLINES)
sidewalk_centerlines = geobase.ensure_crs(sidewalk_centerlines, PROJ_CRS)

# (1) start with raw sidewalk geojson and nyc street centerlines. 

# (2) generate intersection corner segments: 
# (2a) find points where two streets intersect. 
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

