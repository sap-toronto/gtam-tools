import geopandas as gpd
from shapely.wkb import loads, dumps
from warnings import warn

from balsa.routines import sort_nicely as human_sort


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


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    warn('This method is deprecated and will be removed in future releases. Please use a newer version of this method from `balsa.routines.general` instead.', DeprecationWarning, stacklevel=2)
    return human_sort(l)
