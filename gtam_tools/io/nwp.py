import numpy as np
import pandas as pd
from pathlib import Path
from typing import Hashable, List, Tuple, Union
import zipfile

from balsa.logging import get_model_logger

logger = get_model_logger('gtam_tools.io')


def read_nwp_base_network(nwp_fp: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A function to read the base network from a Network Package file (exported from Emme using the TMG Toolbox) into
    DataFrames.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of DataFrames containing the nodes and links
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    header_nodes, header_links, last_line = None, None, None
    with zipfile.ZipFile(nwp_fp) as zf:
        for i, line in enumerate(zf.open('base.211'), start=1):
            line = line.strip().decode('utf-8')
            if line.startswith('c'):
                continue  # Skip comment lines
            if line.startswith('t nodes'):
                header_nodes = i
            elif line.startswith('t links'):
                header_links = i
        last_line = i

        # Read nodes
        n_rows = header_links - header_nodes - 2
        data_types = {
            'c': str, 'Node': np.int64, 'X-coord': float, 'Y-coord': float, 'Data1': float, 'Data2': float,
            'Data3': float, 'Label': str
        }
        nodes = pd.read_csv(zf.open('base.211'), index_col='Node', dtype=data_types, skiprows=header_nodes,
                            nrows=n_rows, delim_whitespace=True)
        nodes.columns = nodes.columns.str.lower()
        nodes.columns = nodes.columns.str.strip()
        nodes.index.name = 'node'
        nodes.rename(columns={'x-coord': 'x', 'y-coord': 'y'}, inplace=True)
        nodes['is_centroid'] = nodes['c'] == 'a*'
        nodes.drop('c', axis=1, inplace=True)

        # Read links
        n_rows = last_line - header_links - 1
        links = pd.read_csv(zf.open('base.211'), index_col=['From', 'To'], skiprows=header_links, nrows=n_rows,
                            delim_whitespace=True, low_memory=False)
        links.columns = links.columns.str.lower()
        links.columns = links.columns.str.strip()
        links.index.names = ['inode', 'jnode']
        mask_mod = links['c'] == 'm'
        n_modified_links = len(links[mask_mod])
        if n_modified_links > 0:
            logger.warning(f'Ignored {n_modified_links} modification records in the links table')
        links = links[~mask_mod].drop('c', axis=1)
        if 'typ' in links.columns:
            links.rename(columns={'typ': 'type'}, inplace=True)
        if 'lan' in links.columns:
            links.rename(columns={'lan': 'lanes'}, inplace=True)
        dtypes = {
            'length': float, 'modes': str, 'type': int, 'lanes': int, 'vdf': int, 'data1': float, 'data2': float,
            'data3': float
        }
        links = links.astype(dtypes)

    return nodes, links


def read_nwp_exatts_list(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read the extra attributes present in a Network Package file (exported from Emme using the TMG
    Toolbox) into DataFrames.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    with zipfile.ZipFile(nwp_fp) as zf:
        df = pd.read_csv(zf.open('exatts.241'), index_col=False)
        df.columns = df.columns.str.strip()
        df['type'] = df['type'].astype('category')

    return df


def read_nwp_link_attributes(nwp_fp: Union[str, Path], *, attributes: Union[str, List[str]] = None) -> pd.DataFrame:
    """A function to read link attributes from a Network Package file exported from Emme using the TMG Toolbox.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.
        attributes (Union[str, List[str]], optional): Defaults to ``None``. Names of link attributes to extract. Note
            that ``'inode'`` and ``'jnode'`` will be included by default.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    if attributes is not None:
        if isinstance(attributes, Hashable):
            attributes = [attributes]
        elif isinstance(attributes, list):
            pass
        else:
            raise RuntimeError

    with zipfile.ZipFile(nwp_fp) as zf:
        df = pd.read_csv(zf.open('exatt_links.241'))
        df.columns = df.columns.str.strip()
        df.set_index(['inode', 'jnode'], inplace=True)

    if attributes is not None:
        df = df[attributes].copy()

    return df


def read_nwp_traffic_results(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read the traffic assignment results from a Network Package file (exported from Emme using the TMG
    Toolbox) into DataFrames.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    with zipfile.ZipFile(nwp_fp) as zf:
        df = pd.read_csv(zf.open('link_results.csv'), index_col=['i', 'j'])
        df.index.names = ['inode', 'jnode']

    return df


def read_nwp_traffic_results_at_countpost(nwp_fp: Union[str, Path], countpost_att: str) -> pd.DataFrame:
    """A function to read the traffic assignment results at countposts from a Network Package file (exported from Emme
    using the TMG Toolbox) into DataFrames.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.
        countpost_att (str): The name of the extra link attribute containing countpost identifiers. Results will be
            filtered using this attribute.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    if not countpost_att.startswith('@'):
        countpost_att = f'@{countpost_att}'

    countpost_links = read_nwp_link_attributes(nwp_fp, countpost_att)
    countpost_links = countpost_links[countpost_links[countpost_att] > 0]

    results = read_nwp_traffic_results(nwp_fp)
    results = results[results.index.isin(countpost_links.index)].copy()

    return results


def read_nwp_transit_line_volumes(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read the transit assignment boardings and max volumes from a Network Package file (exported from
    Emme using the TMG Toolbox) into DataFrames.

    Note:
        Transit line names in Emme must adhere to the TMG NCS16 for this function to work properly.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    assert nwp_fp.exists(), f'File `{nwp_fp.as_posix()}` not found.'

    with zipfile.ZipFile(nwp_fp) as zf:
        data_types = {'line': str, 'transit_boardings': float, 'transit_volume': float}
        df = pd.read_csv(zf.open('segment_results.csv'), usecols=data_types.keys(), dtype=data_types)
        df['operator'] = (df['line'].str[:2]).str.replace('\d+', '')
        df['route'] = df['line'].str.replace(r'\D', '').astype(int)
        df = df.groupby(['operator', 'route']).agg({'transit_boardings': 'sum', 'transit_volume': 'max'})
        df.rename(columns={'transit_boardings': 'boardings', 'transit_volume': 'max_volume'}, inplace=True)

    return df
