import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.wkb import loads, dumps
from typing import Union


def format_gdf(gdf: gpd.GeoDataFrame, index_col: str = None) -> gpd.GeoDataFrame:
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


def read_tts_cross_tabulation_file(fp: Union[str, Path], column_format: bool = False) -> pd.DataFrame:
    """A function to read a TTS Cross Tabulation file downloaded from the DMG Data Retrieval System.

    Args:
        fp (Union[str, Path]): File path to the TTS Cross Tabulation file.
        column_format (bool, optional): Defaults to ``False``. Identifies if the file is in column format.

    Returns:
        pd.DataFrame
    """
    fp = Path(fp)
    assert fp.exists(), f'File `{fp.as_posix()}` not found.'

    # Determine query properties
    row_att = None
    table_att = None
    table_headings = []
    with open(fp) as f:
        for i, line in enumerate(f, start=1):
            if row_att is None and 'Row:' in line:  # only find the first instance
                row_att = line.split('-')[-1].strip()
            if table_att is None and 'Table:' in line:  # only find the first instance
                table_att = line.split('-')[-1].strip()
            if 'TABLE' in line:
                table_headings.append((i, line.split(':')[-1].replace(table_att, '').strip()[1:-1]))
        n_lines = i + 1

    if column_format:
        # Read data from file
        if table_att is not None and len(table_headings) > 0:
            table = []
            for i in range(0, len(table_headings)):
                start_row, table_name = table_headings[i]
                end_row = table_headings[i + 1][0] if i < len(table_headings) - 1 else n_lines + 1

                skip_rows = start_row + 1
                n_rows = end_row - start_row - 4

                df = pd.read_csv(fp, skipinitialspace=True, skiprows=skip_rows, nrows=n_rows, delim_whitespace=True)
                df[table_att] = table_name
                table.append(df)
            table = pd.concat(table, axis=0, ignore_index=True)
        else:
            header_row = None
            with open(fp) as f:
                for i, line in enumerate(f, start=1):
                    if line.strip().startswith(row_att):
                        header_row = i
                        break
            table = pd.read_csv(fp, skipinitialspace=True, skiprows=header_row - 1, delim_whitespace=True)
    else:
        raise NotImplementedError

    return table
