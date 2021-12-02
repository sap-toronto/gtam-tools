import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union
from warnings import warn

import balsa.routines.io.nwp as nwp_tools

deprecated_msg = 'This method is deprecated and will be removed in future releases. Please use a newer version of this method from `balsa.routines.io.nwp` instead.'


def read_nwp_base_network(nwp_fp: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A function to read the base network from a Network Package file (exported from Emme using the TMG Toolbox) into
    DataFrames.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of DataFrames containing the nodes and links
    """
    warn(deprecated_msg, DeprecationWarning, stacklevel=2)
    return nwp_tools.read_nwp_base_network(nwp_fp)


def read_nwp_exatts_list(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read the extra attributes present in a Network Package file (exported from Emme using the TMG
    Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    warn(deprecated_msg, DeprecationWarning, stacklevel=2)
    return nwp_tools.read_nwp_exatts_list(nwp_fp)


def read_nwp_link_attributes(nwp_fp: Union[str, Path], *, attributes: Union[str, List[str]] = None) -> pd.DataFrame:
    """A function to read link attributes from a Network Package file (exported from Emme using the TMG Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.
        attributes (Union[str, List[str]], optional): Defaults to ``None``. Names of link attributes to extract. Note
            that ``'inode'`` and ``'jnode'`` will be included by default.

    Returns:
        pd.DataFrame
    """
    warn(deprecated_msg, DeprecationWarning, stacklevel=2)
    return nwp_tools.read_nwp_link_attributes(nwp_fp, attributes=attributes)


def read_nwp_traffic_results(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read the traffic assignment results from a Network Package file (exported from Emme using the TMG
    Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    warn(deprecated_msg, DeprecationWarning, stacklevel=2)
    return nwp_tools.read_nwp_traffic_results(nwp_fp)


def read_nwp_traffic_results_at_countpost(nwp_fp: Union[str, Path], countpost_att: str) -> pd.DataFrame:
    """A function to read the traffic assignment results at countposts from a Network Package file (exported from Emme
    using the TMG Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.
        countpost_att (str): The name of the extra link attribute containing countpost identifiers. Results will be
            filtered using this attribute.

    Returns:
        pd.DataFrame
    """
    warn(deprecated_msg, DeprecationWarning, stacklevel=2)
    return nwp_tools.read_nwp_traffic_results_at_countpost(nwp_fp, countpost_att)


def read_nwp_transit_network(nwp_fp: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A function to read the transit network from a Network Package file (exported from Emme using the TMG Toolbox)
    into DataFrames.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of DataFrames containing the transt lines and segments.
    """
    warn(deprecated_msg, DeprecationWarning, stacklevel=2)
    return nwp_tools.read_nwp_transit_network(nwp_fp)


def read_nwp_transit_result_summary(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read and summarize the transit assignment boardings and max volumes from a Network Package file
    (exported from Emme using the TMG Toolbox) by operator and route.

    Note:
        Transit line names in Emme must adhere to the TMG NCS16 for this function to work properly.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    warn(deprecated_msg, DeprecationWarning, stacklevel=2)
    return nwp_tools.read_nwp_transit_result_summary(nwp_fp)


def read_nwp_transit_station_results(nwp_fp: Union[str, Path], station_line_nodes: List[int]) -> pd.DataFrame:
    """A function to read and summarize the transit boardings (on) and alightings (offs) at stations from a Network
    Package file (exported from Emme using the TMG Toolbox).

    Note:
        Ensure that station nodes being specified are on the transit line itself and are not station centroids.

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    warn(deprecated_msg, DeprecationWarning, stacklevel=2)
    return nwp_tools.read_nwp_transit_station_results(nwp_fp, station_line_nodes)


def read_nwp_transit_segment_results(nwp_fp: Union[str, Path]) -> pd.DataFrame:
    """A function to read and summarize the transit segment boardings, alightings, and volumes from a Network Package
    file (exported from Emme using the TMG Toolbox).

    Args:
        nwp_fp (Union[str, Path]): File path to the network package.

    Returns:
        pd.DataFrame
    """
    warn(deprecated_msg, DeprecationWarning, stacklevel=2)
    return nwp_tools.read_nwp_transit_segment_results(nwp_fp)
