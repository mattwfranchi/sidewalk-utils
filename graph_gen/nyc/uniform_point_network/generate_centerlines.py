#!/usr/bin/env python3
"""
Standalone script to generate centerlines from sidewalk polygons and save as geoparquet.
This should be run once to create the block-level centerline network.
"""

import sys
import os
import fire
import warnings
sys.path.append('/share/ju/sidewalk_utils')

from centerline_helper import CenterlineHelper
from utils.logger import get_logger
from pandarallel import pandarallel

def generate_centerlines(input_path: str = "/share/ju/sidewalk_utils/data/nyc/_raw/Sidewalk.geojson",
                        output_path: str = "/share/ju/sidewalk_utils/data/nyc/processed/nyc_sidewalk_centerlines.parquet"):
    """
    Generate centerlines from sidewalk polygons and save as geoparquet.
    
    Parameters:
    -----------
    input_path : str
        Path to sidewalk polygon GeoJSON file
    output_path : str
        Path to save the centerlines geoparquet file
    """
    logger = get_logger("GenerateCenterlines")
    
    logger.info("=" * 60)
    logger.info("GENERATING CENTERLINES FROM SIDEWALK POLYGONS")
    logger.info("=" * 60)
    
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    # Initialize centerline helper
    centerline_helper = CenterlineHelper()
    
    # Initialize parallel processing
    pandarallel.initialize(progress_bar=True, nb_workers=8, use_memory_fs=True)
    
    # Generate centerlines
    centerlines_gdf = centerline_helper.convert_polygons_to_centerlines(
        input_path=input_path,
        output_path=output_path
    )
    
    if centerlines_gdf is not None:
        logger.success(f"✅ Successfully generated {len(centerlines_gdf)} centerlines")
        logger.info(f"✅ Saved to: {output_path}")
        
        # Print some statistics
        logger.info("Centerline statistics:")
        logger.info(f"  - Total centerlines: {len(centerlines_gdf)}")
        logger.info(f"  - Average length: {centerlines_gdf.geometry.length.mean():.2f} feet")
        logger.info(f"  - Total length: {centerlines_gdf.geometry.length.sum():.2f} feet")
        logger.info(f"  - Unique parent_ids: {centerlines_gdf['parent_id'].nunique()}")
        
        return True
    else:
        logger.error("❌ Failed to generate centerlines")
        return False

if __name__ == "__main__":
    fire.Fire(generate_centerlines) 