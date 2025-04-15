#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import sys

# Import the logger
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)
logger.setLevel("DEBUG")

# Constants
PROJ_CRS = 'EPSG:2263'
DESIRED_SPACE_BETWEEN_PEDS = 6

def main():
    logger.info("Starting claustrophobia metric calculation")
    
    # Load NYC PUMA data
    logger.info("Loading NYC PUMA data")
    pumas_nyc = gpd.read_file("../data/nyc/sf/tl_2022_36_puma20/tl_2022_36_puma20.shp").to_crs(PROJ_CRS)
    pumas_nyc = pumas_nyc[pumas_nyc['NAMELSAD20'].str.contains('NYC')]
    logger.debug(f"Loaded {len(pumas_nyc)} PUMA regions for NYC")
    
    # Load demographic data
    logger.info("Loading demographic data")
    dp05 = pd.read_csv("../data/nyc/sf/ACSDP1Y2023.DP05-2024-09-24T165218.csv", index_col=0)
    dp05 = dp05[dp05.columns[dp05.columns.str.contains('NYC')].tolist()]
    logger.debug(f"Loaded general demographic data with {len(dp05)} rows")
    
    # Load disability data
    logger.info("Loading disability data")
    s1810 = pd.read_csv("../data/nyc/sf/ACSST1Y2023.S1810-2024-09-24T170627.csv", index_col=0)
    s1810 = s1810[s1810.columns[s1810.columns.str.contains('NYC')].tolist()]
    s1810.index = s1810.index.str.replace('\xa0','')
    
    # Drop all columns with 'Margin of Error' in the name
    s1810 = s1810[s1810.columns[~s1810.columns.str.contains('Margin of Error')].tolist()]
    
    # Extract disability and total data
    s1810_disability = s1810[s1810.columns[s1810.columns.str.contains('With a disability')].tolist()]
    s1810_total = s1810[s1810.columns[s1810.columns.str.contains('Total')].tolist()]
    
    # Process disability data
    logger.info("Processing disability data")
    s1810_disability.columns = s1810_disability.columns.str.split(';').map(lambda x: x[0])
    s1810_disability = s1810_disability.map(lambda x: int(x.strip().replace(',','').replace('±','').strip().replace('NaN', '-1').replace('N','-1')) if isinstance(x, str) else x)
    s1810_disability = s1810_disability.dropna().T
    logger.debug(f"Processed disability data with {s1810_disability.shape[0]} rows and {s1810_disability.shape[1]} columns")
    
    # Process total population data
    logger.info("Processing total population data")
    s1810_total.columns = s1810_total.columns.str.split(';').str[0]
    s1810_total = s1810_total.map(lambda x: int(x.strip().replace(',','').replace('±','').strip().replace('NaN', '-1').replace('N','-1').replace('(X)','-1')) if isinstance(x, str) else x)
    s1810_total = s1810_total.dropna().T
    logger.debug(f"Processed total population data with {s1810_total.shape[0]} rows and {s1810_total.shape[1]} columns")
    
    # Define constants for specific demographic groups
    VISION_DIFFICULTY = 'With a vision difficulty'
    AMBULATORY_DIFFICULTY = 'With an ambulatory difficulty'
    UNDER_5 = 'Under 5 years'
    AGE_5_TO_17 = '5 to 17 years'
    AGE_18_TO_34 = '18 to 34 years'
    AGE_35_TO_64 = '35 to 64 years'
    AGE_65_TO_74 = '65 to 74 years'
    AGE_75_PLUS = '75 years and over'
    TOTAL = 'Total civilian noninstitutionalized population'
    
    # Extract specific demographic data
    with_disability_distribution = s1810_disability[[VISION_DIFFICULTY, AMBULATORY_DIFFICULTY]]
    age_distribution = s1810_total[[UNDER_5, AGE_5_TO_17, AGE_18_TO_34, AGE_35_TO_64, AGE_65_TO_74, AGE_75_PLUS, TOTAL]]
    
    # Combine demographic data
    logger.info("Combining demographic data")
    population = pd.concat([with_disability_distribution, age_distribution], axis=1)
    
    population['child'] = population[UNDER_5] + population[AGE_5_TO_17]
    population['older'] = population[AGE_65_TO_74] + population[AGE_75_PLUS]
    population['no_qualifiers'] = population[AGE_18_TO_34] + population[AGE_35_TO_64]
    population['blind'] = population[VISION_DIFFICULTY]
    population['ambulant'] = population[AMBULATORY_DIFFICULTY]
    population['total'] = population[TOTAL]
    
    # Load pedestrian profiles
    logger.info("Loading pedestrian profiles")
    pedestrian_profiles = pd.read_csv("../data/nyc/sf/pedestrian_profiles.csv", index_col=0)
    PROFILES_TO_DROP = ['wheelchair','stroller']
    pedestrian_profiles = pedestrian_profiles.drop(PROFILES_TO_DROP, axis=0)
    logger.debug(f"Loaded pedestrian profiles with {len(pedestrian_profiles)} profiles after dropping {len(PROFILES_TO_DROP)} profiles")
    
    # Normalize population data
    logger.info("Normalizing population data")
    population = population[['child', 'older', 'no_qualifiers', 'blind', 'ambulant', 'total']]
    population = population.div(population['total'], axis=0)
    population = population.drop('total', axis=1)
    
    # Compute average total sidewalk width needed per PUMA
    logger.info("Computing average sidewalk width needed")
    population_arr = population[['child', 'older', 'no_qualifiers', 'blind', 'ambulant']].values
    pedestrian_profiles_arr = pedestrian_profiles['tsw_per_ped'].values
    tsw = population_arr @ pedestrian_profiles_arr
    tsw = pd.Series(tsw, index=population.index)
    logger.debug(f"Computed TSW for {len(tsw)} PUMA regions")
    
    # Load space data
    logger.info("Loading sidewalk space data")
    nyc_sidewalks = gpd.read_parquet('../data/nyc/claustrophobia/nyc_sidewalks_space.parquet')
    logger.debug(f"Loaded sidewalk space data with {len(nyc_sidewalks)} records")
    
    # Load traffic data
    logger.info("Loading traffic data")
    traffic = gpd.read_parquet('../data/nyc/processed/avg_traffic_by_sidewalk_all.parquet')
    logger.debug(f"Loaded traffic data with {len(traffic)} records")
    
    # Merge traffic data with sidewalks
    logger.info("Merging traffic with sidewalk data")
    nyc_sidewalks = nyc_sidewalks.merge(traffic, on='point_index', how='left', suffixes=('', '_traffic'))
    
    # Spatial join sidewalks to PUMAs
    logger.info("Spatially joining sidewalks to PUMAs")
    nyc_sidewalks = gpd.sjoin(nyc_sidewalks, pumas_nyc, how='left', predicate='intersects')
    
    # Merge total sidewalk width (tsw) on PUMA
    logger.info("Merging TSW data with sidewalks")
    nyc_sidewalks = nyc_sidewalks.merge(tsw.to_frame('tsw'), left_on='NAMELSAD20', right_index=True, how='left')
    
    # Calculate space needed for pedestrians
    logger.info("Calculating space needed for pedestrians")
    
    nyc_sidewalks['space_needed_for_peds_95th'] = nyc_sidewalks['pedestrians_95th'] * (nyc_sidewalks['tsw'] + DESIRED_SPACE_BETWEEN_PEDS)**2
    nyc_sidewalks['space_needed_for_peds_median'] = nyc_sidewalks['pedestrians_median'] * (nyc_sidewalks['tsw'] + DESIRED_SPACE_BETWEEN_PEDS)**2
    
    # Define claustrophobia calculation functions
    def compute_claustrohpobia_95th(row):
        if row['available_space'] == 0:
            return row['space_needed_for_peds_95th']
        if row['space_needed_for_peds_95th'] == 0:
            return 1/row['available_space']
        
        return row['space_needed_for_peds_95th'] / row['available_space']
    
    def compute_claustrohpobia_median(row):
        if row['available_space'] == 0:
            return row['space_needed_for_peds_median']
        if row['space_needed_for_peds_median'] == 0:
            return 1/row['available_space']
        
        return row['space_needed_for_peds_median'] / row['available_space']
    
    # Calculate claustrophobia metrics
    logger.info("Calculating claustrophobia metrics")
    nyc_sidewalks['claustrophobia_95th'] = nyc_sidewalks.apply(compute_claustrohpobia_95th, axis=1)
    nyc_sidewalks['claustrophobia_median'] = nyc_sidewalks.apply(compute_claustrohpobia_median, axis=1)
    
    # Clip values to avoid outliers
    logger.info("Clipping extreme values")
    nyc_sidewalks['claustrophobia_95th'] = nyc_sidewalks['claustrophobia_95th'].clip(
        lower=nyc_sidewalks['claustrophobia_95th'].quantile(0.001), 
        upper=nyc_sidewalks['claustrophobia_95th'].quantile(0.999)
    )
    nyc_sidewalks['claustrophobia_median'] = nyc_sidewalks['claustrophobia_median'].clip(
        lower=nyc_sidewalks['claustrophobia_median'].quantile(0.001), 
        upper=nyc_sidewalks['claustrophobia_median'].quantile(0.999)
    )
    
    # Generate summary statistics
    q_95th = nyc_sidewalks['claustrophobia_95th'].describe([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99])
    q_median = nyc_sidewalks['claustrophobia_median'].describe([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99])
    
    logger.info("95th percentile claustrophobia statistics:")
    for idx in q_95th.index:
        logger.debug(f"  {idx}: {q_95th[idx]:.3f}")
    
    logger.info("Median claustrophobia statistics:")
    for idx in q_median.index:
        logger.debug(f"  {idx}: {q_median[idx]:.3f}")
    
    # Select final columns
    logger.info("Selecting final output columns")
    nyc_sidewalks = nyc_sidewalks[['point_index', 'geometry', 'claustrophobia_95th', 'claustrophobia_median', 'NAMELSAD20',
                                  'tsw', 'pedestrians_median', 'pedestrians_95th', 'available_space', 'total_space', 'space_taken']]
    
    # Save results
    output_path = '../data/nyc/claustrophobia/nyc_sidewalks_claustrophobia.parquet'
    logger.info(f"Saving results to {output_path}")
    nyc_sidewalks.to_parquet(output_path, index=False)
    
    logger.success("Claustrophobia metric calculation completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in claustrophobia calculation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
