import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import os
from shapely.geometry import Point
from tqdm import tqdm
from inequality.gini import Gini

# Optional: momepy for spatial uniformity
try:
    import momepy
    HAS_MOME = True
except ImportError:
    HAS_MOME = False

CRS = 'EPSG:2263'
SQFT_PER_SQMILE = 27878400  # 5280*5280

def load_boundaries(boundary_path):
    gdf = gpd.read_file(boundary_path)
    if gdf.crs != CRS:
        gdf = gdf.to_crs(CRS)
    return gdf


def filter_intersecting_geos(geo_gdf, points_gdf, geo_id_col):
    """
    Filter geographies to only include those that intersect with the point network.
    """
    print(f"  Filtering {len(geo_gdf)} geographies for intersection with network...")
    
    # Create a spatial index for the points for efficient intersection testing
    from shapely.strtree import STRtree
    point_tree = STRtree(list(points_gdf.geometry))
    
    intersecting_indices = []
    for idx, row in tqdm(geo_gdf.iterrows(), total=len(geo_gdf), desc="Checking intersections"):
        poly = row.geometry
        # Find points that intersect with this polygon
        intersecting_points = point_tree.query(poly)
        if len(intersecting_points) > 0:
            intersecting_indices.append(idx)
    
    filtered_gdf = geo_gdf.iloc[intersecting_indices].reset_index(drop=True)
    print(f"  Kept {len(filtered_gdf)} geographies that intersect with network")
    
    return filtered_gdf


def assign_points_to_geos(points_gdf, geo_gdf, geo_id_col):
    # Spatial join: assign each point to a polygon
    print(f"  Spatial join: using column '{geo_id_col}' from boundary file")
    print(f"  Boundary file columns: {list(geo_gdf.columns)}")
    print(f"  Points to assign: {len(points_gdf)}")
    
    # Ensure we have the right columns for the join
    join_columns = [geo_id_col, 'geometry']
    if geo_id_col not in geo_gdf.columns:
        print(f"  WARNING: Column '{geo_id_col}' not found in boundary file!")
        print(f"  Available columns: {list(geo_gdf.columns)}")
        # Try to find a reasonable ID column
        possible_id_cols = [col for col in geo_gdf.columns if any(keyword in col.lower() for keyword in ['id', 'code', 'name', 'label'])]
        if possible_id_cols:
            geo_id_col = possible_id_cols[0]
            print(f"  Using alternative column: '{geo_id_col}'")
            join_columns = [geo_id_col, 'geometry']
        else:
            # Use first non-geometry column
            non_geom_cols = [col for col in geo_gdf.columns if col != 'geometry']
            if non_geom_cols:
                geo_id_col = non_geom_cols[0]
                print(f"  Using first available column: '{geo_id_col}'")
                join_columns = [geo_id_col, 'geometry']
    
    joined = gpd.sjoin(points_gdf, geo_gdf[join_columns], how='left', predicate='within')
    
    # Debug: check results
    assigned_count = joined[geo_id_col].notna().sum()
    print(f"  Points successfully assigned: {assigned_count}/{len(points_gdf)}")
    if assigned_count > 0:
        print(f"  Sample assigned IDs: {joined[geo_id_col].dropna().head().tolist()}")
    
    return joined


def spatial_uniformity_metrics(points_joined_gdf, geo_gdf, geo_id_col):
    # For each polygon, compute:
    # - point count
    # - area (sqft)
    # - density (points per sq mile)
    # - mean nearest neighbor distance (optional, momepy, in feet)
    results = []
    
    # Group points by geography
    points_by_geo = points_joined_gdf.groupby(geo_id_col)
    print(f"  Found {len(points_by_geo)} geographies with points")
    
    for idx, row in tqdm(geo_gdf.iterrows(), total=len(geo_gdf)):
        poly = row.geometry
        geo_id = row[geo_id_col]
        area_sqft = poly.area  # EPSG:2263 is in feet
        area_sqmi = area_sqft / SQFT_PER_SQMILE
        
        # Get points for this geography
        if geo_id in points_by_geo.groups:
            pts_in_poly = points_by_geo.get_group(geo_id)
            n_points = len(pts_in_poly)
            density = n_points / area_sqmi if area_sqmi > 0 else np.nan
            
            # Calculate mean nearest neighbor distance
            mean_nnd = np.nan
            if HAS_MOME and n_points > 1:
                try:
                    mean_nnd = momepy.NearestNeighborDistance(pts_in_poly, 'geometry').series.mean()
                except Exception:
                    mean_nnd = np.nan
        else:
            n_points = 0
            density = 0.0
            mean_nnd = np.nan
            
        results.append({
            geo_id_col: geo_id,
            'n_points': n_points,
            'area_sqft': area_sqft,
            'area_sqmi': area_sqmi,
            'density_points_per_sqmi': density,
            'mean_nnd_ft': mean_nnd
        })
    return pd.DataFrame(results)


def maup_metrics(uniformity_metrics_df, geo_id_col):
    # MAUP: compute Gini coefficient of point density across polygons
    densities = uniformity_metrics_df['density_points_per_sqmi'].dropna().values
    if len(densities) == 0:
        return {'gini': np.nan, 'std': np.nan, 'mean': np.nan}
    gini_val = Gini(densities).g
    return {'gini': gini_val, 'std': np.std(densities), 'mean': np.mean(densities)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate sampled point network for MAUP and spatial uniformity.")
    parser.add_argument('--network', type=str, required=True, help='Path to sampled point network (parquet or geojson)')
    parser.add_argument('--cb', type=str, default='/share/ju/sidewalk_utils/data/nyc/geo/cb-nyc-2020.geojson', help='Path to census blocks geojson')
    parser.add_argument('--ct', type=str, default='/share/ju/sidewalk_utils/data/nyc/geo/ct-nyc-2020.geojson', help='Path to census tracts geojson')
    parser.add_argument('--nta', type=str, default='/share/ju/sidewalk_utils/data/nyc/geo/nta-nyc-2020.geojson', help='Path to NTA geojson')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save results')
    args = parser.parse_args()

    # Load sampled network
    print(f"Loading sampled network from {args.network}")
    if args.network.endswith('.parquet'):
        points_gdf = gpd.read_parquet(args.network)
    else:
        points_gdf = gpd.read_file(args.network)
    if points_gdf.crs != CRS:
        points_gdf = points_gdf.to_crs(CRS)
    print(f"Loaded {len(points_gdf)} points")

    # Load boundaries
    print("Loading boundaries...")
    cb_gdf = load_boundaries(args.cb)
    ct_gdf = load_boundaries(args.ct)
    nta_gdf = load_boundaries(args.nta)
    
    print(f"Census blocks: {len(cb_gdf)} features, columns: {list(cb_gdf.columns)}")
    print(f"Census tracts: {len(ct_gdf)} features, columns: {list(ct_gdf.columns)}")
    print(f"NTAs: {len(nta_gdf)} features, columns: {list(nta_gdf.columns)}")

    # Filter geographies that do not intersect with the network
    print("\nFiltering geographies that do not intersect with the network...")
    
    # Determine column names first
    cb_id_col = None
    for possible_col in ['BCTCB2020', 'GEOID', 'BLOCKID', 'BLOCKCE', 'GEOID20', 'GEOID10']:
        if possible_col in cb_gdf.columns:
            cb_id_col = possible_col
            break
    if cb_id_col is None:
        cb_id_col = cb_gdf.columns[0]
    
    ct_id_col = 'BoroCT2020' if 'BoroCT2020' in ct_gdf.columns else ct_gdf.columns[0]
    nta_id_col = 'NTACode' if 'NTACode' in nta_gdf.columns else nta_gdf.columns[0]
    
    # Filter geographies
    cb_gdf = filter_intersecting_geos(cb_gdf, points_gdf, cb_id_col)
    ct_gdf = filter_intersecting_geos(ct_gdf, points_gdf, ct_id_col)
    nta_gdf = filter_intersecting_geos(nta_gdf, points_gdf, nta_id_col)

    # Assign points to geographies
    print("\nAssigning points to census blocks...")
    points_cb = assign_points_to_geos(points_gdf, cb_gdf, cb_id_col)
    
    print("\nAssigning points to census tracts...")
    points_ct = assign_points_to_geos(points_gdf, ct_gdf, ct_id_col)
    
    print("\nAssigning points to NTAs...")
    points_nta = assign_points_to_geos(points_gdf, nta_gdf, nta_id_col)

    # Compute metrics
    print("\nComputing spatial uniformity metrics for blocks...")
    cb_metrics = spatial_uniformity_metrics(points_cb, cb_gdf, cb_id_col)
    print("\nComputing spatial uniformity metrics for tracts...")
    ct_metrics = spatial_uniformity_metrics(points_ct, ct_gdf, ct_id_col)
    print("\nComputing spatial uniformity metrics for NTAs...")
    nta_metrics = spatial_uniformity_metrics(points_nta, nta_gdf, nta_id_col)

    # Compute MAUP metrics
    print("\nComputing MAUP metrics for blocks...")
    cb_maup = maup_metrics(cb_metrics, cb_id_col)
    print("Computing MAUP metrics for tracts...")
    ct_maup = maup_metrics(ct_metrics, ct_id_col)
    print("Computing MAUP metrics for NTAs...")
    nta_maup = maup_metrics(nta_metrics, nta_id_col)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    cb_metrics.to_csv(os.path.join(args.output_dir, 'cb_uniformity_metrics.csv'), index=False)
    ct_metrics.to_csv(os.path.join(args.output_dir, 'ct_uniformity_metrics.csv'), index=False)
    nta_metrics.to_csv(os.path.join(args.output_dir, 'nta_uniformity_metrics.csv'), index=False)
    with open(os.path.join(args.output_dir, 'maup_summary.txt'), 'w') as f:
        f.write('Census Block MAUP: ' + str(cb_maup) + '\n')
        f.write('Census Tract MAUP: ' + str(ct_maup) + '\n')
        f.write('NTA MAUP: ' + str(nta_maup) + '\n')
    print("Done. Results saved to:", args.output_dir)

if __name__ == "__main__":
    main() 