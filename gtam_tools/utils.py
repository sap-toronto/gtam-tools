import geopandas as gpd
from shapely.wkb import loads, dumps


def format_gdf(gdf: gpd.GeoDataFrame, *, index_col: str = None) -> gpd.GeoDataFrame:
    """A function to prepare a GeoDataFrame for usage.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to format.
        index_col (str, optional): Defaults to ``None``. The name of the column in `gdf` to use as the index.

    Returns:
        gpd.GeoDataFrame
    """
    gdf = gdf.to_crs(epsg=26917)
    gdf.columns = gdf.columns.str.lower()
    if index_col is not None:
        gdf = gdf.set_index(index_col.lower()).sort_index()
    gdf['geometry'] = gdf['geometry'].apply(lambda x: loads(dumps(x, output_dimension=2)))  # flatten 3d to 2d

    return gdf
