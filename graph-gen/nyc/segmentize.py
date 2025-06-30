import os
import pandas as pd
import fire
from functools import wraps
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union, Callable, Any
from shapely import Point
from pathlib import Path
import geopandas as gpd

# Import constants
from data.nyc.c import PROJ_FT
from geo_processor_base import GeoDataProcessor
from user import INSTALL_DIR

# Import CPU fallbacks and specialized utils
from segmentize_cpu import SegmentizeCPUFallbacks
# Import GPU implementations and utils
from segmentize_gpu import SegmentizeGPUImplementations
from segmentize_utils import (
    segmentize_and_extract_points,
    prepare_segmentized_dataframe,
    compute_adjacency,
    consolidate_corner_points
)

# Type aliases for improved readability
GeoDataFrame = gpd.GeoDataFrame
PathLike = Union[str, Path]

@dataclass
class ProcessingContext:
    """
    Context object to hold processing state for method chaining
    
    This object tracks all parameters and intermediate results during the
    segmentization workflow. It's used both by the process method and
    the fluent interface methods.
    """
    input_path: PathLike
    output_path: Optional[PathLike] = None
    segmentation_distance: float = 50
    adj_tolerance: float = 5
    compute_adj: bool = True
    point_adjacency: bool = True
    point_distance_threshold: Optional[float] = None
    sidewalks_data: Optional[GeoDataFrame] = None  # Original sidewalk geometries
    segmentized_points: Optional[GeoDataFrame] = None  # Points after segmentization
    result: Optional[GeoDataFrame] = None  # Final processed result
    start_time: Optional[pd.Timestamp] = None  # For performance tracking
    success: bool = False  # Whether processing succeeded

def rapids_or_fallback(fallback_method_name: str) -> Callable:
    """
    Decorator to check for RAPIDS availability and fall back to a CPU implementation.
    
    This decorator attempts to use GPU-accelerated methods first via the RAPIDS ecosystem
    (cuDF, cuSpatial). If those libraries aren't available, it automatically falls back
    to the CPU implementation defined in the SegmentizeCPUFallbacks class.
    
    Parameters:
    -----------
    fallback_method_name : str
        Name of the method to call from SegmentizeCPUFallbacks if RAPIDS is not available
        
    Returns:
    --------
    function
        Decorated function that automatically falls back to CPU implementation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            try:
                # Try to import RAPIDS libraries
                import cudf
                import cuspatial
                self.logger.info("RAPIDS libraries found, using GPU acceleration")
                return func(self, *args, **kwargs)
            except ImportError as e:
                # If import fails, run the fallback function
                self.logger.warning(f"RAPIDS libraries not available, falling back to CPU implementation: {e}")
                
                # Initialize the CPU fallbacks class if not already done
                if not hasattr(self, '_cpu_fallbacks'):
                    self._cpu_fallbacks = SegmentizeCPUFallbacks(self)
                
                # Call the appropriate method from the fallbacks class
                fallback_method = getattr(self._cpu_fallbacks, fallback_method_name)
                return fallback_method(*args, **kwargs)
        return wrapper
    return decorator

class SidewalkSegmentizer(GeoDataProcessor):
    """
    Tool for segmentizing sidewalk geometries into points for network analysis.
    """

    def __init__(self) -> None:
        """Initialize the SidewalkSegmentizer with its own logger."""
        super().__init__(name=__name__)
        
        # Prevent duplicate logging by disabling propagation on one logger
        self.logger.propagate = False  # This prevents messages from being passed up to parent loggers
        
        # Initialize CPU fallbacks and GPU implementations (will be lazily created when needed)
        self._cpu_fallbacks: Optional[SegmentizeCPUFallbacks] = None
        self._gpu_implementations: Optional[SegmentizeGPUImplementations] = None
        self._current_ctx: Optional[ProcessingContext] = None  # For fluent interface
    
    # Core utility methods
    def convert_segments_to_points(self, gdf: GeoDataFrame, distance: float = 50) -> GeoDataFrame:
        """
        Convert sidewalk segments into regularly spaced points
        
        This method takes linear sidewalk geometries and converts them into
        evenly spaced points along each segment, based on the specified distance.
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input geodataframe with sidewalk geometries
        distance : float
            Distance between points in segmentation (feet)
            
        Returns:
        --------
        GeoDataFrame
            DataFrame containing point geometries derived from the input segments
        """
        return segmentize_and_extract_points(gdf, distance=distance, logger=self.logger)
        
    def prepare_final_dataframe(self, points_gdf: GeoDataFrame, source_gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Prepare the final point dataframe with attributes from source segments
        
        Takes the segmentized points and transfers attributes from the original
        sidewalk segments that those points were generated from.
        
        Parameters:
        -----------
        points_gdf : GeoDataFrame
            GeoDataFrame with segmentized point geometries
        source_gdf : GeoDataFrame
            Original geodataframe with attributes to transfer
            
        Returns:
        --------
        GeoDataFrame
            Points with source attributes attached
        """
        return prepare_segmentized_dataframe(points_gdf, source_gdf, logger=self.logger)
        
    def compute_segment_adjacency(self, gdf: GeoDataFrame, tolerance: float = 0.1) -> GeoDataFrame:
        """
        Compute adjacency relationships between sidewalk segments
        
        Determines which sidewalk segments are adjacent to each other within a
        specified tolerance distance. This creates the base adjacency network
        before segmentization into points.
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input geodataframe with sidewalk geometries
        tolerance : float
            Distance tolerance for considering segments adjacent (feet)
            
        Returns:
        --------
        GeoDataFrame
            Input geodataframe with additional 'adjacent_ids' column
        """
        return compute_adjacency(gdf, tolerance=tolerance, logger=self.logger)
        
    def merge_nearby_corner_points(self, gdf: GeoDataFrame, min_distance: float = 10) -> GeoDataFrame:
        """
        Merge points that are too close together at corners
        
        After segmentization, points near corners/intersections can be very close 
        to each other. This method consolidates them to avoid redundancy and improve
        network topology.
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input geodataframe with point geometries
        min_distance : float
            Minimum distance between points - any closer will be merged (feet)
            
        Returns:
        --------
        GeoDataFrame
            Points with nearby corner points consolidated
        """

        # check if gdf is a geodataframe, if not exit with appropriate error 
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a GeoDataFrame")
        if gdf.empty:   
            raise ValueError("Input GeoDataFrame is empty")

        return consolidate_corner_points(gdf, min_distance=min_distance, logger=self.logger)

    # GPU-accelerated methods with CPU fallbacks
    @rapids_or_fallback(fallback_method_name='sidewalk_network_filter_cpu')
    def filter_points_with_sidewalk_network(self, segmentized_points: GeoDataFrame, 
                                           og_sidewalk_file_path: PathLike) -> GeoDataFrame:
        """
        Filter segmentized points using GPU-accelerated point-in-polygon operations
        
        Ensures that all points created during segmentization fall within actual
        sidewalk polygons, removing any that may be outside due to geometric operations.
        Uses GPU acceleration when available.
        
        Parameters:
        -----------
        segmentized_points : GeoDataFrame
            Points to filter
        og_sidewalk_file_path : str or Path
            Path to original sidewalk polygons file
            
        Returns:
        --------
        GeoDataFrame
            Filtered points guaranteed to be within sidewalk polygons
        """
        # Import GPU libraries (decorator ensures these imports work)
        import cudf
        import cuspatial
        
        # Convert to string if it's a Path
        sidewalk_path = str(og_sidewalk_file_path)
        
        # Lazily initialize GPU implementations if needed
        if not hasattr(self, '_gpu_implementations') or self._gpu_implementations is None:
            self._gpu_implementations = SegmentizeGPUImplementations(self)
            
        # Delegate to GPU implementation
        return self._gpu_implementations.sidewalk_network_filter(
            segmentized_points, 
            sidewalk_path
        )

    @rapids_or_fallback(fallback_method_name='compute_point_adjacency_parallel')
    def compute_point_level_adjacency(self, segmentized_points: GeoDataFrame, source_gdf: GeoDataFrame, 
                                    point_distance_threshold: float = 10, 
                                    batch_size: int = 1000) -> GeoDataFrame:
        """
        Compute adjacency between individual points based on segment adjacency
        
        After segmentizing segments into points, this method establishes which points
        are connected to each other, creating a network graph of points. It uses
        the segment-level adjacency info to efficiently process only relevant
        point pairs, rather than checking all possible combinations.
        
        Uses GPU acceleration when available, with automatic fallback to CPU.
        
        Parameters:
        -----------
        segmentized_points : GeoDataFrame
            The segmentized points with level_0 indicating the parent segment
        source_gdf : GeoDataFrame
            The original segments with adjacent_ids column from segment adjacency computation
        point_distance_threshold : float
            Maximum distance between points to be considered adjacent (in feet)
        batch_size : int
            Size of segment pair batches to process (adjust based on GPU memory)
            
        Returns:
        --------
        GeoDataFrame
            Points with point-level adjacency information in 'point_adjacent_ids' column
        """
        # Import GPU libraries (decorator ensures these imports work)
        import cudf
        import cuspatial
        
        # Lazily initialize GPU implementations if needed
        if not hasattr(self, '_gpu_implementations') or self._gpu_implementations is None:
            self._gpu_implementations = SegmentizeGPUImplementations(self)
            
        # Delegate to GPU implementation
        return self._gpu_implementations.compute_point_adjacency(
            segmentized_points, 
            source_gdf,
            point_distance_threshold=point_distance_threshold,
            batch_size=batch_size
        )

    #-------------- PROCESSING PIPELINE STEPS ---------------#
    
    def setup_processing(self, ctx: ProcessingContext) -> ProcessingContext:
        """
        Set up processing parameters and log them
        
        First step in the processing pipeline that validates input parameters,
        calculates defaults, and prepares for processing.
        """
        self.logger.info(f"Processing parameters:")
        self.logger.info(f"  - Input path: {ctx.input_path}")
        self.logger.info(f"  - Segmentation distance: {ctx.segmentation_distance} feet")
        
        # Calculate intelligent default for point_distance_threshold if not specified
        if ctx.point_distance_threshold is None:
            # Use 1.01x the segmentation distance as the default threshold
            ctx.point_distance_threshold = 1.1 * ctx.segmentation_distance
            self.logger.info(f"  - Using auto-calculated point distance threshold: {ctx.point_distance_threshold:.2f} feet (1.01 × segmentation distance)")
        else:
            self.logger.info(f"  - Using manual point distance threshold: {ctx.point_distance_threshold} feet")
        
        # Set default output path if not provided
        if ctx.output_path is None:
            input_dir = os.path.dirname(str(ctx.input_path))
            input_basename = os.path.basename(str(ctx.input_path)).split('.')[0]
            ctx.output_path = os.path.join(input_dir, f"{input_basename}_segmentized.parquet")
            self.logger.info(f"Using default output path: {ctx.output_path}")
        else:
            self.logger.info(f"Output path: {ctx.output_path}")
        
        self.logger.info("Starting sidewalk segmentization process")
        ctx.start_time = pd.Timestamp.now()
        return ctx
    
    def load_sidewalk_data(self, ctx: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Load sidewalk data from input path
        
        Loads the sidewalk geometries from the specified file and ensures
        they're in the correct coordinate system.
        """
        use_sidewalk_widths = "sidewalkwidths" in str(ctx.input_path).lower()
        self.logger.info(f"Data source identified as: {'sidewalk widths' if use_sidewalk_widths else 'standard sidewalks'}")
        
        ctx.sidewalks_data = self.read_geodataframe(
            ctx.input_path, 
            crs=PROJ_FT, 
            name="sidewalk data"
        )
        
        if ctx.sidewalks_data is None:
            self.logger.error("Failed to load input data")
            return None
            
        return ctx
    
    def simplify_sidewalk_geometries(self, ctx: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Simplify sidewalk geometries to improve processing efficiency
        
        Reduces the complexity of sidewalk geometries by removing redundant vertices
        while preserving their essential shape.
        """
        if ctx.sidewalks_data is None:
            self.logger.error("No sidewalk data available to simplify")
            return None
            
        self.logger.info("Simplifying geometries")
        ctx.sidewalks_data = self.simplify_geometries(ctx.sidewalks_data, tolerance=10)
        
        if ctx.sidewalks_data is None:
            self.logger.error("Failed to simplify geometries")
            return None
            
        return ctx
    
    def calculate_segment_adjacency(self, ctx: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Calculate adjacency relationships between sidewalk segments
        
        Determines which sidewalk segments are adjacent to each other. This information
        is later used to establish connectivity between points derived from different
        segments.
        """
        if ctx.sidewalks_data is None:
            self.logger.error("No sidewalk data available to calculate adjacency")
            return None
            
        if ctx.compute_adj:
            self.logger.info("Computing segment-level adjacency relationships")
            ctx.sidewalks_data = self.compute_segment_adjacency(ctx.sidewalks_data, tolerance=ctx.adj_tolerance)
            
            if ctx.sidewalks_data is None:
                self.logger.error("Failed to compute segment adjacency")
                return None
                
        return ctx
    
    def segmentize_sidewalks(self, ctx: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Convert sidewalk geometries into regularly spaced points
        
        Transforms the sidewalk line or polygon geometries into a series of points
        spaced at regular intervals (defined by segmentation_distance).
        """
        if ctx.sidewalks_data is None:
            self.logger.error("No sidewalk data available to segmentize")
            return None
            
        ctx.segmentized_points = self.convert_segments_to_points(
            ctx.sidewalks_data, 
            distance=ctx.segmentation_distance
        )
        
        if ctx.segmentized_points is None:
            self.logger.error("Failed to segmentize data")
            return None
            
        self.logger.info(f"Segmentized into {len(ctx.segmentized_points)} points")
        return ctx
    
    def merge_corner_points(self, ctx: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Consolidate closely spaced points at corners and intersections
        
        Merges points that are too close together to avoid redundancy in the network
        and create better topology at intersections.
        """
        if ctx.segmentized_points is None:
            self.logger.error("No segmentized points available to merge corners")
            return None
            
        # FIX: First ensure we have a proper GeoDataFrame before any geometry operations
        if not isinstance(ctx.segmentized_points, gpd.GeoDataFrame):
            self.logger.warning("Converting result to GeoDataFrame before consolidation")
            try:
                # Check if it's a GeoSeries (common case after segmentization)
                if isinstance(ctx.segmentized_points, gpd.GeoSeries):
                    self.logger.info("Converting GeoSeries to GeoDataFrame")
                    ctx.segmentized_points = gpd.GeoDataFrame(
                        geometry=ctx.segmentized_points,
                        crs=ctx.segmentized_points.crs
                    )
                else:
                    # Try to find possible geometry column in a regular DataFrame
                    points_df = ctx.segmentized_points
                    geometry_col = None
                    
                    # Try to find a column with Point objects
                    if 0 in points_df.columns and isinstance(points_df[0].iloc[0], Point):
                        geometry_col = 0
                    else:
                        # Check other columns
                        for col in points_df.columns:
                            if isinstance(points_df[col].iloc[0], Point):
                                geometry_col = col
                                break
                                
                    if geometry_col is not None:
                        self.logger.info(f"Using column '{geometry_col}' as geometry")
                        ctx.segmentized_points = gpd.GeoDataFrame(
                            points_df, 
                            geometry=geometry_col,
                            crs=PROJ_FT
                        )
                    else:
                        self.logger.error("Could not find geometry column")
                        return None
            except Exception as e:
                self.logger.error(f"Failed to convert to GeoDataFrame: {e}")
                return None

        # Now consolidate points to avoid duplication at corners
        if ctx.segmentation_distance is not None:
            consolidation_distance = ctx.segmentation_distance * 0.3
            self.logger.info(f"Consolidating closely clustered points (threshold: {consolidation_distance}ft)")
            ctx.segmentized_points = self.merge_nearby_corner_points(
                ctx.segmentized_points, 
                min_distance=consolidation_distance
            )
            
            if ctx.segmentized_points is None:
                self.logger.error("Failed to consolidate corner points")
                return None
                
        return ctx
    
    def filter_points_to_sidewalks(self, ctx: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Filter points to ensure they fall within actual sidewalk areas
        
        Uses the original sidewalk polygons to validate that all points
        created during segmentization fall within legitimate sidewalk areas.
        """
        if ctx.segmentized_points is None:
            self.logger.error("No segmentized points available to filter")
            return None
            
        # Save the original geometry column name and check for actual Point objects
        orig_geom_col = ctx.segmentized_points.geometry.name if isinstance(ctx.segmentized_points, gpd.GeoDataFrame) else None
        
        # Log information about input data structure for debugging
        if isinstance(ctx.segmentized_points, gpd.GeoDataFrame):
            self.logger.info(f"Input to filter is a GeoDataFrame with geometry column '{ctx.segmentized_points.geometry.name}'")
            self.logger.info(f"Available columns: {ctx.segmentized_points.columns.tolist()}")
        else:
            self.logger.info(f"Input to filter is not a GeoDataFrame: {type(ctx.segmentized_points)}")
        
        # Apply the filtering operation
        ctx.segmentized_points = self.filter_points_with_sidewalk_network(
            ctx.segmentized_points, 
            f"{INSTALL_DIR}/data/nyc/_raw/Sidewalk.geojson"
        )
        
        if ctx.segmentized_points is None:
            self.logger.error("Failed to filter points")
            return None
            
        # Log filtered data structure for debugging
        self.logger.info(f"Filtered result type: {type(ctx.segmentized_points)}")
        self.logger.info(f"Filtered result columns: {ctx.segmentized_points.columns.tolist()}")
        
        # Ensure geometry column is properly set after filtering
        if isinstance(ctx.segmentized_points, gpd.GeoDataFrame):
            # Check if geometry column is properly set
            if ctx.segmentized_points.geometry.name is None or ctx.segmentized_points.geometry.name not in ctx.segmentized_points.columns:
                self.logger.warning("Geometry column lost during filtering, attempting to restore...")
                try:
                    # Try to restore original geometry column
                    if orig_geom_col and orig_geom_col in ctx.segmentized_points.columns:
                        self.logger.info(f"Restoring original geometry column '{orig_geom_col}'")
                        ctx.segmentized_points = gpd.GeoDataFrame(
                            ctx.segmentized_points,
                            geometry=orig_geom_col,
                            crs=PROJ_FT
                        )
                    elif 'geometry' in ctx.segmentized_points.columns:
                        self.logger.info("Using 'geometry' column as geometry")
                        ctx.segmentized_points = gpd.GeoDataFrame(
                            ctx.segmentized_points,
                            geometry='geometry',
                            crs=PROJ_FT
                        )
                    else:
                        # Try to find any column with Point objects as a last resort
                        for col in ctx.segmentized_points.columns:
                            if len(ctx.segmentized_points) > 0 and isinstance(ctx.segmentized_points[col].iloc[0], Point):
                                self.logger.info(f"Found column '{col}' containing Point objects, using as geometry")
                                ctx.segmentized_points = gpd.GeoDataFrame(
                                    ctx.segmentized_points,
                                    geometry=col,
                                    crs=PROJ_FT
                                )
                                break
                        else:
                            self.logger.warning("Could not restore geometry column automatically")
                except Exception as e:
                    self.logger.error(f"Failed to restore geometry column: {e}")
                    # Continue anyway, we'll try again in the next step
        else:
            self.logger.warning("Filter result is not a GeoDataFrame, attempting to convert")
            try:
                # Check if it's a pandas DataFrame with a 'geometry' column
                if hasattr(ctx.segmentized_points, 'columns') and 'geometry' in ctx.segmentized_points.columns:
                    self.logger.info("Converting DataFrame with 'geometry' column to GeoDataFrame")
                    ctx.segmentized_points = gpd.GeoDataFrame(
                        ctx.segmentized_points, 
                        geometry='geometry',
                        crs=PROJ_FT
                    )
                # Check if it's a pandas DataFrame with any column containing Point objects
                elif hasattr(ctx.segmentized_points, 'columns'):
                    for col in ctx.segmentized_points.columns:
                        if len(ctx.segmentized_points) > 0 and isinstance(ctx.segmentized_points[col].iloc[0], Point):
                            self.logger.info(f"Found column '{col}' with Point objects, using as geometry")
                            ctx.segmentized_points = gpd.GeoDataFrame(
                                ctx.segmentized_points,
                                geometry=col,
                                crs=PROJ_FT
                            )
                            break
                    else:
                        self.logger.error("Could not find any column with Point objects")
                        return None
                else:
                    self.logger.error("Filter result is not a DataFrame and could not be converted")
                    return None
            except Exception as e:
                self.logger.error(f"Failed to convert filter result to GeoDataFrame: {e}")
                return None
                
        # Final verification
        if not isinstance(ctx.segmentized_points, gpd.GeoDataFrame):
            self.logger.error("Failed to ensure result is a GeoDataFrame after filtering")
            return None
            
        self.logger.info(f"Final filtered result: GeoDataFrame with {len(ctx.segmentized_points)} points, " +
                        f"geometry column '{ctx.segmentized_points.geometry.name}'")
        
        return ctx
    
    def establish_point_adjacency(self, ctx: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Establish adjacency relationships between individual points
        """
        if ctx.segmentized_points is None:
            self.logger.error("No segmentized points available to establish adjacency")
            return None
            
        if ctx.sidewalks_data is None:
            self.logger.error("No sidewalk data available for adjacency computation")
            return None
            
        # Fix: Ensure geometry column is properly set before adjacency computation
        if not isinstance(ctx.segmentized_points, gpd.GeoDataFrame):
            self.logger.error("Segmentized points is not a GeoDataFrame")
            return None
        
        # Check if geometry column is properly set
        if ctx.segmentized_points.geometry.name is None or ctx.segmentized_points.geometry.name not in ctx.segmentized_points.columns:
            self.logger.warning("Geometry column not properly set, attempting to fix...")
            try:
                # Try to identify the geometry column
                if 'geometry' in ctx.segmentized_points.columns:
                    self.logger.info("Using 'geometry' column as geometry")
                    ctx.segmentized_points = gpd.GeoDataFrame(
                        ctx.segmentized_points,
                        geometry='geometry',
                        crs=PROJ_FT
                    )
                else:
                    # Try to find a column with Point objects
                    for col in ctx.segmentized_points.columns:
                        if isinstance(ctx.segmentized_points[col].iloc[0], Point):
                            self.logger.info(f"Using column '{col}' as geometry")
                            ctx.segmentized_points = gpd.GeoDataFrame(
                                ctx.segmentized_points,
                                geometry=col,
                                crs=PROJ_FT
                            )
                            break
                    else:
                        self.logger.error("Could not find geometry column")
                        return None
            except Exception as e:
                self.logger.error(f"Failed to fix geometry column: {e}")
                return None
                
        if ctx.compute_adj and ctx.point_adjacency:
            # Validate segment adjacency data before computing point adjacency
            if 'adjacent_ids' not in ctx.sidewalks_data.columns:
                self.logger.warning("No segment adjacency data found, computing it now")
                ctx.sidewalks_data = self.compute_segment_adjacency(ctx.sidewalks_data, tolerance=ctx.adj_tolerance)
                if ctx.sidewalks_data is None:
                    self.logger.error("Failed to compute segment adjacency")
                    return None
            
            # Log segment adjacency statistics for debugging
            if 'adjacent_ids' in ctx.sidewalks_data.columns:
                segment_adj_counts = ctx.sidewalks_data['adjacent_ids'].apply(len)
                avg_segment_adj = segment_adj_counts.mean()
                max_segment_adj = segment_adj_counts.max()
                isolated_segments = (segment_adj_counts == 0).sum()
                self.logger.info(f"Segment adjacency stats: avg={avg_segment_adj:.2f}, max={max_segment_adj}, isolated={isolated_segments}")
            
            self.logger.info("Computing point-level adjacency relationships")
            ctx.segmentized_points = self.compute_point_level_adjacency(
                ctx.segmentized_points, 
                ctx.sidewalks_data,
                point_distance_threshold=ctx.point_distance_threshold
            )
            
            if ctx.segmentized_points is None:
                self.logger.error("Failed to compute point adjacency")
                return None
            
            # Validate point adjacency results
            if 'point_adjacent_ids' in ctx.segmentized_points.columns:
                point_adj_counts = ctx.segmentized_points['point_adjacent_ids'].apply(len)
                avg_point_adj = point_adj_counts.mean()
                max_point_adj = point_adj_counts.max()
                isolated_points = (point_adj_counts == 0).sum()
                self.logger.info(f"Point adjacency validation: avg={avg_point_adj:.2f}, max={max_point_adj}, isolated={isolated_points}")
                
                # Warn if too many points are isolated
                isolation_rate = isolated_points / len(ctx.segmentized_points)
                if isolation_rate > 0.1:  # More than 10% isolated
                    self.logger.warning(f"High isolation rate: {isolation_rate:.1%} of points are isolated")
                    self.logger.warning("Consider increasing point_distance_threshold or checking segment adjacency")
                    
        return ctx
    
    def prepare_final_data(self, ctx: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Prepare the final dataframe with attributes from original segments
        
        Creates the final point network by attaching attributes from the original
        sidewalk segments to their corresponding points.
        """
        if ctx.segmentized_points is None:
            self.logger.error("No segmentized points available to prepare final data")
            return None
            
        if ctx.sidewalks_data is None:
            self.logger.error("No sidewalk data available for attribute transfer")
            return None
            
        ctx.result = self.prepare_final_dataframe(ctx.segmentized_points, ctx.sidewalks_data)
        
        if ctx.result is None:
            self.logger.error("Failed to prepare final dataframe")
            return None
            
        # Ensure result is in the correct CRS
        self.logger.info(f"Ensuring output is in {PROJ_FT}")
        prev_crs = ctx.result.crs
        ctx.result = self.ensure_crs(ctx.result, PROJ_FT)
        
        if ctx.result is None:
            self.logger.error("Failed to ensure correct CRS")
            return None
            
        if prev_crs != PROJ_FT:
            self.logger.info(f"CRS transformed from {prev_crs} to {PROJ_FT}")
            bounds = ctx.result.total_bounds
            self.logger.info(f"Output bounds [minx, miny, maxx, maxy]: {bounds}")
            
        return ctx
    
    def save_output_data(self, ctx: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Save the final point network to output file
        
        Writes the processed point network with all attributes and adjacency
        information to disk in the specified format.
        """
        if ctx.result is None:
            self.logger.error("No result data available to save")
            return None
            
        if ctx.output_path is None:
            self.logger.error("No output path specified")
            return None
            
        self.logger.info(f"Saving {len(ctx.result)} segmentized points to {ctx.output_path}")
        success = self.save_geoparquet(ctx.result, ctx.output_path)
        
        if not success:
            self.logger.error("Failed to save output file")
            return None
            
        ctx.success = True
        return ctx
    
    def report_statistics(self, ctx: ProcessingContext) -> ProcessingContext:
        """
        Report final statistics about the processing results
        
        Calculates and logs performance metrics, data transformation statistics,
        and other useful information about the completed process.
        """
        if ctx.result is None or ctx.sidewalks_data is None or ctx.start_time is None:
            self.logger.warning("Incomplete context data, skipping detailed statistics")
            return ctx
            
        end_time = pd.Timestamp.now()
        elapsed_time = (end_time - ctx.start_time).total_seconds()
        points_per_second = len(ctx.result) / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info(f"Processing statistics:")
        self.logger.info(f"  - Input features: {len(ctx.sidewalks_data)}")
        self.logger.info(f"  - Output points: {len(ctx.result)}")
        self.logger.info(f"  - Points-to-feature ratio: {len(ctx.result)/len(ctx.sidewalks_data):.1f}")
        self.logger.info(f"  - Processing time: {elapsed_time:.1f} seconds")
        self.logger.info(f"  - Processing speed: {points_per_second:.1f} points/second")
        
        # Add network statistics if adjacency was computed
        if 'point_adjacent_ids' in ctx.result.columns:
            avg_connections = ctx.result['point_adjacent_ids'].apply(len).mean()
            max_connections = ctx.result['point_adjacent_ids'].apply(len).max()
            isolated_points = (ctx.result['point_adjacent_ids'].apply(len) == 0).sum()
            
            self.logger.info(f"  - Network statistics:")
            self.logger.info(f"    - Average connections per point: {avg_connections:.2f}")
            self.logger.info(f"    - Maximum connections for a point: {max_connections}")
            self.logger.info(f"    - Isolated points: {isolated_points} ({isolated_points/len(ctx.result)*100:.1f}%)")
        
        self.logger.success("Sidewalk segmentization completed successfully")
        return ctx
    
    def execute_processing_pipeline(self, ctx: ProcessingContext) -> bool:
        """
        Execute the complete processing pipeline with error handling
        
        This is the main orchestrator method that runs each step in the
        processing pipeline in sequence, with proper error handling.
        
        Parameters:
        -----------
        ctx : ProcessingContext
            Processing context with input parameters and state
            
        Returns:
        --------
        bool
            True if processing was successful, False otherwise
        """
        # Define the processing pipeline steps in order
        pipeline_steps: List[Tuple[Callable[[ProcessingContext], Optional[ProcessingContext]], str]] = [
            (self.setup_processing, "Setting up processing parameters"),
            (self.load_sidewalk_data, "Loading sidewalk data"),
            (self.simplify_sidewalk_geometries, "Simplifying geometries"),
            (self.calculate_segment_adjacency, "Calculating segment adjacency"),
            (self.segmentize_sidewalks, "Converting sidewalks to points"),
            (self.merge_corner_points, "Merging corner points"),
            (self.filter_points_to_sidewalks, "Filtering points to sidewalk areas"),
            (self.establish_point_adjacency, "Establishing point adjacency network"),
            (self.prepare_final_data, "Preparing final data structure"),
            (self.save_output_data, "Saving output data"),
            (self.report_statistics, "Reporting statistics")
        ]
        
        # Execute each step in sequence
        for step_func, step_description in pipeline_steps:
            self.logger.info(f"STEP: {step_description}")
            step_start = pd.Timestamp.now()
            
            # Execute the step
            ctx = step_func(ctx)
            
            # Calculate step duration
            step_duration = (pd.Timestamp.now() - step_start).total_seconds()
            
            # Check for failure
            if ctx is None:
                self.logger.error(f"Pipeline failed during: {step_description} after {step_duration:.2f} seconds")
                return False
                
            self.logger.info(f"Completed in {step_duration:.2f} seconds")
                
        return ctx.success
    
    def process(self, i: PathLike, o: Optional[PathLike] = None,
               segmentation_distance: float = 50, compute_adj: bool = True, adj_tolerance: float = 0.1,
               point_adjacency: bool = True, point_distance_threshold: Optional[float] = None) -> bool:
        """
        Process sidewalk data to segmentize into points
        
        Main entry point for command-line processing. Takes sidewalk geometries and
        converts them into a network of regularly spaced points with adjacency relationships.
        
        Args:
            i: Path to input sidewalk data
            o: Path to save output data (default: derived from input path)
            segmentation_distance: Distance between points in segmentation, feet (default: 50)
            compute_adj: Whether to compute adjacency relationships (default: True)
            adj_tolerance: Distance tolerance for considering geometries adjacent (default: 0.1 feet)
            point_adjacency: Whether to compute point-level adjacency (default: True)
            point_distance_threshold: Maximum distance between points to be considered adjacent (default: 1.01 * segmentation_distance)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create context object with all parameters
            ctx = ProcessingContext(
                input_path=i,
                output_path=o,
                segmentation_distance=segmentation_distance,
                adj_tolerance=adj_tolerance,
                compute_adj=compute_adj,
                point_adjacency=point_adjacency,
                point_distance_threshold=point_distance_threshold
            )
            
            # Run the processing pipeline
            return self.execute_processing_pipeline(ctx)
            
        except Exception as e:
            self.logger.error(f"Unhandled exception in process method: {e}")
            return False
            
    # Alternative fluent interface implementation
    def with_input(self, input_path: PathLike) -> 'SidewalkSegmentizer':
        """
        Start a fluent processing chain with an input file
        
        First method in the fluent interface chain that specifies the input data source.
        
        Parameters:
        -----------
        input_path : str or Path
            Path to the input sidewalk data file
            
        Returns:
        --------
        SidewalkSegmentizer
            Self reference for method chaining
        """
        ctx = ProcessingContext(input_path=input_path)
        self._current_ctx = ctx
        return self
        
    def with_output(self, output_path: PathLike) -> 'SidewalkSegmentizer':
        """
        Set output path for the processing chain
        
        Specifies where the processed point network should be saved.
        
        Parameters:
        -----------
        output_path : str or Path
            Path where output data should be written
            
        Returns:
        --------
        SidewalkSegmentizer
            Self reference for method chaining
        """
        if self._current_ctx is None:
            raise ValueError("No processing context. Call with_input() first.")
        self._current_ctx.output_path = output_path
        return self
        
    def with_segmentation_distance(self, distance: float) -> 'SidewalkSegmentizer':
        """
        Set segmentation distance for the processing chain
        
        Controls how densely spaced the points will be along sidewalk segments.
        
        Parameters:
        -----------
        distance : float
            Distance between points in feet
            
        Returns:
        --------
        SidewalkSegmentizer
            Self reference for method chaining
        """
        if self._current_ctx is None:
            raise ValueError("No processing context. Call with_input() first.")
        self._current_ctx.segmentation_distance = distance
        return self
        
    def with_adjacency(self, compute: bool = True, tolerance: float = 0.1) -> 'SidewalkSegmentizer':
        """
        Configure segment adjacency computation for the processing chain
        
        Controls whether segment-level adjacency should be computed and the
        tolerance for determining adjacency.
        
        Parameters:
        -----------
        compute : bool
            Whether to compute segment adjacency
        tolerance : float
            Distance threshold for considering segments adjacent (feet)
            
        Returns:
        --------
        SidewalkSegmentizer
            Self reference for method chaining
        """
        if self._current_ctx is None:
            raise ValueError("No processing context. Call with_input() first.")
        self._current_ctx.compute_adj = compute
        self._current_ctx.adj_tolerance = tolerance
        return self
        
    def with_point_adjacency(self, compute: bool = True, threshold: Optional[float] = None) -> 'SidewalkSegmentizer':
        """
        Configure point-level adjacency computation for the processing chain
        
        Controls whether point-level adjacency should be computed and the
        distance threshold for determining which points are adjacent.
        
        Parameters:
        -----------
        compute : bool
            Whether to compute point adjacency
        threshold : float, optional
            Maximum distance between points to be considered adjacent (feet)
            If None, will use 1.01 * segmentation_distance
            
        Returns:
        --------
        SidewalkSegmentizer
            Self reference for method chaining
        """
        if self._current_ctx is None:
            raise ValueError("No processing context. Call with_input() first.")
        self._current_ctx.point_adjacency = compute
        self._current_ctx.point_distance_threshold = threshold
        return self
        
    def run(self) -> bool:
        """
        Run the processing chain with the configured parameters
        
        Final method in the fluent interface chain that executes the
        processing pipeline with all configured parameters.
        
        Returns:
        --------
        bool
            True if processing was successful, False otherwise
        """
        try:
            if self._current_ctx is None:
                self.logger.error("No processing context configured. Call with_input() first.")
                return False
                
            ctx = self._current_ctx
            self._current_ctx = None  # Clear for next chain
            return self.execute_processing_pipeline(ctx)
        except Exception as e:
            self.logger.error(f"Unhandled exception in fluent processing chain: {e}")
            self._current_ctx = None  # Clear for next chain
            return False

if __name__ == "__main__":
    # Use the Fire CLI library to expose the SidewalkSegmentizer class
    fire.Fire(SidewalkSegmentizer)