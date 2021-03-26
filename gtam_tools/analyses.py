import pandas as pd
from typing import Union

from balsa.routines import tlfd
from cheval import LinkedDataFrame


def extract_tflds(trips_df: Union[pd.DataFrame, LinkedDataFrame], agg_col: str, *, impedance_col: str = 'impedance',
                  bin_start: int = 0, bin_end: int = 200, bin_step: int = 2) -> pd.DataFrame:
    """A function to extract TLFDs from the model results trip table.

    Args:
        trips_df (Union[pd.DataFrame, LinkedDataFrame]): The trips table from the Microsim results. Ideally, this table
            would be from ``MicrosimData.trips``.
        agg_col (str): The name of the column in the trips table to plot TLFDs by category/group.
        impedance_col (str, optional): Defaults to ``'impedance'``. The column in ``trips_df`` containing the impedances
            to use for calculating TFLDs.
        bin_start (int): Defaults is ``0``. The minimum bin value.
        bin_end (int): Defaults to ``200``. The maximum bin value.
        bin_step (int): Default is ``2``. The size of each bin.

    Returns:
        pd.DataFrame: The TLFDs from the model results trip table in wide format.
    """
    model_tlfds = {}
    for label, subset in trips_df.groupby(agg_col):
        fd = tlfd(subset[impedance_col], bin_start=bin_start, bin_end=bin_end, bin_step=bin_step,
                  weights=subset['weight'], intrazonal=subset['o_zone'] == subset['d_zone'], label_type='MULTI',
                  include_top=True)
        model_tlfds[label] = fd
    model_tlfds = pd.DataFrame(model_tlfds)
    model_tlfds.columns.name = agg_col

    return model_tlfds


def extract_e2e_linkages(table: Union[pd.DataFrame, LinkedDataFrame], ensembles: pd.Series, *, agg_col: str = None,
                         orig_col: str = 'o_zone', dest_col: str = 'd_zone'):
    """A function to extract ensemble-to-ensemble linkages from the model results.

    Args:
        table (Union[pd.DataFrame, LinkedDataFrame]): The table from the Microsim results. Ideally, this table
            would be from ``MicrosimData.trips`` or ``MicrosimData.persons``.
        ensembles (pd.Series): The TAZ to ensemble definition to use.
        agg_col (str, optional): Defaults to ``None``. The name of the column in the trips table to aggregate by
            category/group.
    """
    df = table.copy()
    df['o_ensemble'] = ensembles.reindex(df[orig_col]).values
    df['d_ensemble'] = ensembles.reindex(df[dest_col]).values

    usecols = ['o_ensemble', 'd_ensemble']
    if agg_col is not None:
        usecols.append(agg_col)

    e2e_df = df.groupby(usecols).agg({'weight': 'sum'})
    if agg_col is not None:
        e2e_df = e2e_df.unstack().fillna(0)
        e2e_df.columns = e2e_df.columns.droplevel(0)
    else:
        e2e_df.rename(columns={'weight': 'trips'}, inplace=True)

    return e2e_df
