import geopandas as gpd
import re
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


def _tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def _alphanum_key(s):
    """Turn a string into a list of string and number chunks (eg. "z23a" -> ["z", 23, "a"])"""
    return [_tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    l = l.copy()
    l.sort(key=_alphanum_key)
    return l
