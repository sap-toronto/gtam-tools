import pandas as pd

from balsa.routines import tlfd
from cheval import LinkedDataFrame


def extract_tflds(trips_df: LinkedDataFrame, agg_col: str, *, impedance_method: str = 'manhattan', bin_start: int = 0,
                  bin_end: int = 200, bin_step: int = 2) -> pd.DataFrame:
    """A function to extract TLFDs from the model results trip table.

    Args:
        trips_df (LinkedDataFrame): The trips table from the Microsim results. Ideally, this table would be from
            ``MicrosimData.trips``.
        agg_col (str): The name of the column in the trips table to plot TLFDs by category/group.
        impedance_method (str, optional): Defaults to ``'manhattan'``. The impedance method to calculate TLFDs. Must be
            a valid impedance availble in ``MicrosimData.impedances``.
        bin_start (int): Defaults is ``0``. The minimum bin value.
        bin_end (int): Defaults to ``200``. The maximum bin value.
        bin_step (int): Default is ``2``. The size of each bin.

    Returns:
        pd.DataFrame: The TLFDs from the model results trip table in wide format.
    """
    model_tlfds = {}
    for label, subset in trips_df.groupby(agg_col):
        impedances = getattr(subset.imped, impedance_method)
        fd = tlfd(impedances, bin_start=bin_start, bin_end=bin_end, bin_step=bin_step, weights=subset['weight'],
                  intrazonal=subset['o_zone'] == subset['d_zone'], label_type='MULTI', include_top=True)
        model_tlfds[label] = fd
    model_tlfds = pd.DataFrame(model_tlfds)
    model_tlfds.columns.name = agg_col

    return model_tlfds


def extract_e2e_trips(trips_df: LinkedDataFrame, ensembles: pd.Series, *, agg_col: str = None):
    """A function to extract ensemble-to-ensemble trips from the model results trip table.

    Args:
        trips_df (LinkedDataFrame): The trips table from the Microsim results. Ideally, this table would be from
            ``MicrosimData.trips``.
        ensembles (pd.Series): The TAZ to ensemble definition to use.
        agg_col (str, optional): Defaults to ``None``. The name of the column in the trips table to aggregate by
            category/group.
    """
    trips = trips_df.copy()
    trips['o_ensemble'] = ensembles.reindex(trips['o_zone']).values
    trips['d_ensemble'] = ensembles.reindex(trips['d_zone']).values

    usecols = ['o_ensemble', 'd_ensemble']
    if agg_col is not None:
        usecols.append(agg_col)

    e2e_trips = trips.groupby(usecols).agg({'weight': 'sum'})
    if agg_col is not None:
        e2e_trips = e2e_trips.unstack().fillna(0)
        e2e_trips.columns = e2e_trips.columns.droplevel(0)
    else:
        e2e_trips.rename(columns={'weight': 'trips'}, inplace=True)

    return e2e_trips
