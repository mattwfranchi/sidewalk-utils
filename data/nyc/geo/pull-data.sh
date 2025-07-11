# Street Flooding Project 
# Developer: Matt Franchi, @mattwfranchi 
# Cornell Tech 

# This script pulls the necessary geographic datasets for the creation of our analysis dataframe. 

# Geographic Boundaries 

## 2020 NYC Census Tracts, water areas clipped
wget -O ct-nyc-2020.geojson 'https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Census_Tracts_for_2020_US_Census/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=pgeojson'

## 2020 NYC Census Tracts, including water areas
wget -O ct-nyc-wi-2020.geojson 'https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Census_Tracts_for_2020_US_Census_Water_Included/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=pgeojson'

## 2020 NYC Census Blocks, including water areas 
wget -O cb-nyc-wi-2020.geojson 'https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Census_Blocks_for_2020_US_Census_Water_Included/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=pgeojson'

## 2020 NYC Census Blocks, water areas clipped
wget -O cb-nyc-2020.geojson 'https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Census_Blocks_for_2020_US_Census/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=pgeojson'

## NYC Integer 1 foot Digital Elevation Model Raster 
wget -O nyc-1ft-dem.zip 'https://sa-static-customer-assets-us-east-1-fedramp-prod.s3.amazonaws.com/data.cityofnewyork.us/NYC_DEM_1ft_Int.zip'
unzip -d data nyc-1ft-dem.zip

## NYC 2020 Neighborhood Tabulation Areas 
wget -O nta-nyc-2020.geojson 'https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson'

# delete zip
rm nyc-1ft-dem.zip