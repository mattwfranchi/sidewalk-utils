# Add these functions to the class or as standalone utilities

import shapely.wkb as swkb
import base64
import pandas as pd

def shapely_to_wkb_base64(geometry):
    """Convert a Shapely geometry to base64-encoded WKB string"""
    if geometry is None:
        return None
    wkb_data = swkb.dumps(geometry)
    return base64.b64encode(wkb_data).decode('ascii')

def wkb_base64_to_shapely(wkb_str):
    """Convert a base64-encoded WKB string back to Shapely geometry"""
    if not wkb_str:
        return None
    wkb_data = base64.b64decode(wkb_str)
    return swkb.loads(wkb_data)

# Example usage when loading from parquet:
def load_with_shapely_geometries(parquet_file):
    """Load a parquet file and convert WKB strings back to Shapely geometries"""
    df = pd.read_parquet(parquet_file)
    
    if 'shapely_boxes' in df.columns:
        def convert_box_array(box_array):
            # Check if it's a NumPy array or list
            if box_array is None:
                return []
                
            # Convert each string in the array to a Shapely object
            result = []
            for box_str in box_array:
                if box_str is not None and isinstance(box_str, str):
                    try:
                        geometry = wkb_base64_to_shapely(box_str)
                        result.append(geometry)
                    except Exception as e:
                        print(f"Error decoding geometry: {e}")
            
            return result
        
        # Apply the conversion
        df['shapely_boxes'] = df['shapely_boxes'].apply(convert_box_array)
    
    return df