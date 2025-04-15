#!/usr/bin/env python
"""
Test driver for segmentize.py that processes sampled census tracts.
Runs the segmentize process for sidewalk files in the tracts/ directory.
"""

import os
import sys
import glob
import time
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging

# Add the parent directory to the path so we can import SidewalkSegmentizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmentize import SidewalkSegmentizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("segmentize_test.log"), logging.StreamHandler()]
)
logger = logging.getLogger("segmentize_tester")

def find_tract_sidewalk_pairs(tracts_dir="./tracts"):
    """Find all matching tract and sidewalk parquet files in the tracts directory."""
    # Get all sidewalk files
    sidewalk_files = glob.glob(os.path.join(tracts_dir, "sidewalks_*.parquet"))
    pairs = []
    
    for sidewalk_file in sidewalk_files:
        # Extract the unique identifier from the sidewalk filename
        basename = os.path.basename(sidewalk_file)
        unique_id = basename.replace("sidewalks_", "")
        
        # Find the matching tract file
        tract_file = os.path.join(tracts_dir, f"tract_{unique_id}")
        
        if os.path.exists(tract_file):
            pairs.append((tract_file, sidewalk_file))
        else:
            logger.warning(f"No matching tract file found for {sidewalk_file}")
    
    return pairs

def process_tract_sidewalk_pair(pair, output_dir="./output", segmentation_distance=50):
    """Process a tract-sidewalk file pair using SidewalkSegmentizer."""
    tract_file, sidewalk_file = pair
    
    # Extract tract information for the output filename
    basename = os.path.basename(tract_file).replace("tract_", "")
    tract_id = basename.split(".")[0]  # Remove file extension
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, f"segmentized_{tract_id}")
    
    logger.info(f"Processing tract-sidewalk pair: {tract_id}")
    logger.info(f"Input sidewalk file: {sidewalk_file}")
    logger.info(f"Output file: {output_file}")
    
    # Initialize the segmentizer
    segmentizer = SidewalkSegmentizer()
    
    # Process the sidewalk data
    start_time = time.time()
    success = segmentizer.process(
        i=sidewalk_file,
        o=output_file,
        compute_adj=True,
        segmentation_distance=segmentation_distance,
        point_adjacency=True
    )
    elapsed_time = time.time() - start_time
    
    if success:
        logger.info(f"Processing completed successfully in {elapsed_time:.2f} seconds")
        
        # Load the output file and log statistics
        try:
            result = gpd.read_parquet(output_file)
            logger.info(f"Output statistics: {len(result)} points generated")
            return True
        except Exception as e:
            logger.error(f"Error reading output file: {e}")
            return False
    else:
        logger.error(f"Processing failed for {tract_id}")
        return False

def main():
    """Main function to run the segmentize tests."""
    logger.info("Starting segmentize tester")
    
    # Find all tract-sidewalk pairs
    pairs = find_tract_sidewalk_pairs()
    logger.info(f"Found {len(pairs)} tract-sidewalk pairs to process")
    
    if not pairs:
        logger.error("No tract-sidewalk pairs found. Please run geometries.ipynb first.")
        return
    
    # Create results table
    results = []
    
    # Process each pair
    for i, pair in enumerate(pairs):
        logger.info(f"Processing pair {i+1} of {len(pairs)}")
        
        tract_file, sidewalk_file = pair
        tract_id = os.path.basename(tract_file).replace("tract_", "").split(".")[0]
        
        try:
            start_time = time.time()
            success = process_tract_sidewalk_pair(pair)
            elapsed_time = time.time() - start_time
            
            # Add to results
            results.append({
                "tract_id": tract_id,
                "success": success,
                "processing_time": elapsed_time
            })
            
            # Don't overwhelm the system, brief pause between runs
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing pair {tract_id}: {e}")
            results.append({
                "tract_id": tract_id,
                "success": False,
                "processing_time": None,
                "error": str(e)
            })
    
    # Create and save summary report
    results_df = pd.DataFrame(results)
    success_rate = results_df["success"].mean() * 100
    
    logger.info(f"Testing complete. Success rate: {success_rate:.1f}%")
    logger.info(f"Average processing time: {results_df['processing_time'].mean():.2f} seconds")
    
    # Save results to CSV
    results_df.to_csv("segmentize_test_results.csv", index=False)
    logger.info("Results saved to segmentize_test_results.csv")

if __name__ == "__main__":
    main()