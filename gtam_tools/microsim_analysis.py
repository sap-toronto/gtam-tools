from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from balsa.logging import get_model_logger
from balsa.routines import tlfd
from cheval import LinkedDataFrame
from pyproj import CRS

from .enums import TimeFormat, ZoneNums
from .microsim_data import MicrosimData

PREPPED_DATA = Tuple[LinkedDataFrame, List[str], List[str], List[str], List[str]]


class MicrosimAnalysis(object):

    _microsim_data: MicrosimData
    _ensemble_mindex: pd.MultiIndex
    _ensemble_names: Optional[pd.DataFrame]
    _partition_col: str

    _logger = get_model_logger('gtam_tools.MicrosimAnalysis')

    _activity_groups = {
        'home': ['Home', 'ReturnFromWork'],
        'work': ['PrimaryWork', 'SecondaryWork', 'WorkBasedBusiness'],
        'school': ['School'],
        'market': ['Market', 'JointMarket'],
        'other': ['IndividualOther', 'JointOther']
    }

    @property
    def microsim_data_loaded(self) -> bool:
        return self._microsim_data is not None

    @property
    def microsim_data(self) -> MicrosimData:
        return self._microsim_data

    def __init__(self, run_folder: Union[str, PathLike], shortest_path_skim: Union[str, PathLike],
                 zones_file: Union[str, PathLike], *, zones_crs: Union[str, CRS] = None, taz_col: str = 'gta06',
                 partition_col: str = 'pd', coord_unit: float = 0.001,
                 time_format: TimeFormat = TimeFormat.MINUTE_DELTA,
                 max_internal_zone: Union[int, Enum] = ZoneNums.MAX_INTERNAL):
        """Processes GTAModel Microsim results.

        Args:
            run_folder (Union[str, PathLike]): The path to the GTAModel Microsim Results folder
            shortest_path_skim (Union[str, PathLike]): The path to the shortest path skim matrix file.
            zones_file (Union[str, PathLike]): The path to a file containing zone coordinates. Accepts CSVs with x, y
                coordinate columns, or a zones shapefile. If providing a shapefile, it will calculate the zone
                coordinate based on the centroid of each zone polygon.
            zones_crs (Union[str, CRS], optional): Defaults to ``None``. The coordinate reference system for
                ``zones_file``. Only applicable if ``zones_file`` is a CSV file. Value can be anything accepted by
                ``pyproj.CRS.from_user_input()``.
            taz_col (str, optional): Defaults to ``None``. Name of the TAZ column in ``zones_file``.
            partition_col (str, optional): Defaults to ``'pd'``. Name of the partition/ensembles column in
                ``zones_file``.
            coord_unit (float, optional): Defaults to ``0.001``. A value to adjust distance values with.
            time_format (TimeFormat, optional): Defaults to ``TimeFormat.MINUTE_DELTA``. Specify the time format in the
                Microsim results. Used for parsing times in the trip modes file.
            max_internal_zone (Union[int, Enum], optional): Defaults to ``ZoneNums.MAX_INTERNAL``. The highest internal
                zone number allowed. Argument accepts an integer or an Enum attribute (e.g. ``ZoneNums.MAX_INTERNAL``).
        """
        self._logger.tip('Initializing MicrosimAnalysis')

        shortest_path_skim = Path(shortest_path_skim)
        if not shortest_path_skim.exists():
            raise FileNotFoundError(f'Shortest path skim `{shortest_path_skim.as_posix()}` not found')

        data = MicrosimData.load_folder(
            run_folder, time_format=time_format, zones_file=zones_file, taz_col=taz_col, zones_crs=zones_crs,
            coord_unit=coord_unit, max_internal_zone=max_internal_zone
        )
        data.add_impedance_skim('network', shortest_path_skim, scale_unit=coord_unit)
        data.add_zone_ensembles('ensemble', zones_file, taz_col, ensemble_col=partition_col)
        self._microsim_data = data

        ensembles = sorted(data.zone_coordinates['ensemble'].unique())
        self._ensemble_mindex = pd.MultiIndex.from_product([ensembles, ensembles], names=['o_ensemble', 'd_ensemble'])

        self._partition_col = partition_col

        self._logger.tip('MicrosimAnalysis ready!')

    # region Loaders

    def load_model_activity_data(self, *, attach_time_period: bool = False,
                                 impedance_type: str = None) -> LinkedDataFrame:
        df = self.microsim_data.trips.copy()
        df['o_activity'] = ''
        df['d_activity'] = ''
        for label, activities in self._activity_groups.items():
            df.loc[df['o_act'].isin(activities), 'o_activity'] = label
            df.loc[df['d_act'].isin(activities), 'd_activity'] = label
        if impedance_type is not None:
            df['impedance'] = getattr(df.imped, impedance_type)
        df = df.loc[df.person.age >= 11].copy()

        if attach_time_period:
            index_cols = ['household_id', 'person_id', 'trip_id']
            mask_hh = self.microsim_data.trip_modes['household_id'].isin(df['household_id'])
            mask_per = self.microsim_data.trip_modes['person_id'].isin(df['person_id'])
            trip_periods = self.microsim_data.trip_modes.loc[mask_hh & mask_per].copy()
            trip_periods = trip_periods.groupby(index_cols).head(1).set_index(index_cols)['time_period']

            trip_index = pd.MultiIndex.from_frame(df[['household_id', 'person_id', 'trip_id']])
            df['time_period'] = trip_periods.reindex(trip_index).to_numpy()

        df.drop([
            'household_id', 'person_id', 'trip_id', 'o_act', 'd_act', 'repetitions', 'purpose', 'direction'
        ], axis=1, inplace=True)

        return df

    # endregion

    # region Utilities

    @staticmethod
    def _check_col_arg(table: Union[pd.DataFrame, LinkedDataFrame], col: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(col, str):
            col = [col]
        elif isinstance(col, Iterable):
            if not all([isinstance(s, str) for s in col]):
                raise TypeError('All elements of the iterable `col` must be string')
        else:
            raise TypeError(f'Invalid `col` type ({type(col)})')

        if not all([c in table for c in col]):
            raise ValueError('`col` value(s) not found in table')

        return col

    @staticmethod
    def _eval_masks(table: Union[pd.DataFrame, LinkedDataFrame], masks: List[str]) -> np.ndarray:
        table_mask = None
        for mask in masks:
            result = table.evaluate(mask) if isinstance(table, LinkedDataFrame) else table.eval()
            if result.dtype != bool:
                raise RuntimeError(f'`{mask}` does not evaluate to a boolean mask')
            if table_mask is None:
                table_mask = result
            else:
                table_mask = table_mask & result
        return np.array(table_mask)

    @staticmethod
    def _calc_tlfds(table: Union[pd.DataFrame, LinkedDataFrame], o_col: str, d_col: str, result_name: str, *,
                    col: List[str] = None, **kwargs) -> pd.DataFrame:
        if col is None:
            df = tlfd(
                table['impedance'], weights=table['weight'], intrazonal=table[o_col] == table[d_col], **kwargs
            ).to_frame(name=result_name)
        else:
            df = []
            for value, subset in table.groupby(col):
                subset_fd = tlfd(
                    subset['impedance'], weights=subset['weight'], intrazonal=subset[o_col] == subset[d_col],
                    **kwargs
                ).to_frame(name=result_name).reset_index()
                if len(col) == 1:
                    subset_fd[col[0]] = value
                else:
                    for label, subvalue in zip(col, value):
                        subset_fd[label] = subvalue
                df.append(subset_fd)
            df: pd.DataFrame = pd.concat(df, axis=0, ignore_index=True)
            df.set_index(['from', 'to', *col], inplace=True)

        return df

    # endregion

    # region Summaries

    def _process_drivers(self, table: Union[pd.DataFrame, LinkedDataFrame], weight_col: str, *,
                         col: List[str] = None) -> pd.DataFrame:
        data = self.microsim_data
        if col is None:
            df = table.groupby('home_zone')[weight_col].sum().to_frame(name='n_drivers')
        else:
            df = table.pivot_table(values=weight_col, index='home_zone', columns=col, aggfunc='sum', fill_value=0)
        df['ensemble'] = data.zone_coordinates['ensemble'].reindex(df.index).to_numpy()
        df = df.groupby('ensemble').sum()
        df.index = df.index.astype(int)
        df.index.name = self._partition_col
        if col is None:
            df = pd.DataFrame(df)
        else:
            df = df.stack().to_frame(name='n_drivers')

        return df

    def summarize_drivers_hhlds(self, *, hhld_col: Union[str, Iterable[str]] = None,
                                masks: List[str] = None) -> pd.DataFrame:
        df = self.microsim_data.households
        if hhld_col is not None:
            hhld_col = self._check_col_arg(df, hhld_col)
        if masks is not None:
            df = df.loc[self._eval_masks(df, masks)].copy()
        df['drivers_w'] = df['drivers'] * df['weight']

        return self._process_drivers(df, 'drivers_w', col=hhld_col)

    def summarize_drivers_pers(self, *, pers_col: Union[str, Iterable[str]] = None,
                               masks: List[str] = None) -> pd.DataFrame:
        data = self.microsim_data
        df: LinkedDataFrame = data.persons.loc[data.persons['license']]
        if pers_col is not None:
            pers_col = self._check_col_arg(df, pers_col)
            pers_col = [c for c in pers_col if c != 'license']
        if masks is not None:
            df = df.loc[self._eval_masks(df, masks)].copy()

        return self._process_drivers(df, 'weight', col=pers_col)

    def summarize_ao(self, *, hhld_col: Union[str, Iterable[str]] = None, masks: List[str] = None,
                     clip_upper: int = None) -> pd.DataFrame:
        data = self.microsim_data
        df: LinkedDataFrame = data.households.copy()
        if hhld_col is not None:
            hhld_col = self._check_col_arg(df, hhld_col)
        if masks is not None:
            df = df.loc[self._eval_masks(df, masks)].copy()
        df['ensemble'] = data.zone_coordinates['ensemble'].reindex(df['home_zone']).to_numpy()
        if clip_upper is not None:
            df['vehicles'] = df['vehicles'].clip(upper=clip_upper)

        index_cols = 'ensemble' if hhld_col is None else ['ensemble', *hhld_col]
        df = df.pivot_table(values='weight', index=index_cols, columns='vehicles', aggfunc='sum', fill_value=0)
        if hhld_col is None:
            df.index = df.index.astype(int)
            df.index.name = self._partition_col
        else:
            df.index = df.index.set_levels(df.index.levels[0].astype(int), level=0)
            df.index.names = [self._partition_col, *hhld_col]
        df = df.stack().to_frame(name='n_households')

        return df

    def summarize_porpow_porpos_e2e(self, dest_col: str, *, pers_col: Union[str, Iterable[str]] = None,
                                    masks: List[str] = None, convert_categorical_to_str: bool = True) -> pd.DataFrame:
        data = self.microsim_data
        df: LinkedDataFrame = data.persons.copy()
        if pers_col is not None:
            pers_col = self._check_col_arg(df, pers_col)
            if convert_categorical_to_str:
                for col in pers_col:
                    if df[col].dtype.name == 'category':
                        df[col] = df[col].astype(str)
        mask = df[dest_col] > 0
        if masks is not None:
            mask = mask & self._eval_masks(df, masks)
        df = df.loc[mask].copy()

        cols = ['home_zone', dest_col] if pers_col is None else ['home_zone', dest_col, *pers_col]
        df = df.groupby(cols, as_index=False)['weight'].sum()
        df['o_ensemble'] = data.zone_coordinates['ensemble'].reindex(df['home_zone']).to_numpy().astype(int)
        df['d_ensemble'] = data.zone_coordinates['ensemble'].reindex(df[dest_col]).fillna(0).to_numpy().astype(int)
        df['d_ensemble'] = np.where(df[dest_col] == ZoneNums.ROAMING, -1, df['d_ensemble'])
        cols = ['o_ensemble', 'd_ensemble'] if pers_col is None else ['o_ensemble', 'd_ensemble', *pers_col]
        df = df.groupby(cols)['weight'].sum().to_frame(name='n_linkages')
        if pers_col is None:
            df.index.names = [f'{self._partition_col}_o', f'{self._partition_col}_d']
        else:
            df.index.names = [f'{self._partition_col}_o', f'{self._partition_col}_d', *pers_col]

        return df

    def summarize_porpow_porpos_tlfd(self, dest_col: str, *, pers_col: Union[str, Iterable[str]] = None,
                                     masks: List[str] = None, impedance_type: str = 'network',
                                     convert_categorical_to_str: bool = True, **kwargs) -> pd.DataFrame:
        data = self.microsim_data
        df: LinkedDataFrame = data.persons
        if pers_col is not None:
            pers_col = self._check_col_arg(df, pers_col)
            if convert_categorical_to_str:
                for col in pers_col:
                    if df[col].dtype.name == 'category':
                        df[col] = df[col].astype(str)
        mask = df[dest_col] > 0
        if masks is not None:
            mask = mask & self._eval_masks(df, masks)
        df = df.loc[mask].copy()
        df.link_to(data.impedances, 'imped', on_self=['home_zone', dest_col])
        df['impedance'] = getattr(df.imped, impedance_type)

        return self._calc_tlfds(df, 'home_zone', dest_col, 'n_linkages', col=pers_col, **kwargs)

    def _prep_model_activity_data(self, *, att_col: Union[str, Iterable[str]] = None, masks: List[str] = None,
                                  model_data: LinkedDataFrame = None, attach_time_period: bool = False,
                                  impedance_type: str = None, convert_categorical_to_str: bool = True) -> PREPPED_DATA:
        if model_data is None:
            df = self.load_model_activity_data(attach_time_period=attach_time_period, impedance_type=impedance_type)
        else:
            df = model_data.copy()

        # Sort attributes
        hhld_col, pers_col, trip_col = [], [], []
        if att_col is not None:
            if isinstance(att_col, str):
                att_col = [att_col]
            elif isinstance(att_col, Iterable):
                pass
            else:
                raise TypeError(f'Invalid `att_col` type ({type(att_col)})')

            for col in att_col:
                if col in self.microsim_data.households:
                    hhld_col.append(col)
                elif col in self.microsim_data.persons:
                    pers_col.append(col)
                elif col in df:
                    trip_col.append(col)
                else:
                    raise ValueError(f'`{col}` not found in either the household, person, or trip tables')

        # Promote attributes from LinkedDataFrame links
        for col in hhld_col:
            df[col] = getattr(df.person.household, col)
        for col in pers_col:
            df[col] = getattr(df.person, col)
        if convert_categorical_to_str:
            for col in {*hhld_col, *pers_col, *trip_col}:
                if df[col].dtype.name == 'category':
                    df[col] = df[col].astype(str)

        if masks is not None:
            df = df.loc[self._eval_masks(df, masks)].copy()

        return df, att_col, hhld_col, pers_col, trip_col

    def summarize_model_activity_e2e(self, *, att_col: Union[str, Iterable[str]] = None, masks: List[str] = None,
                                     model_data: LinkedDataFrame = None, attach_time_period: bool = False,
                                     convert_categorical_to_str: bool = True) -> pd.DataFrame:
        data = self.microsim_data
        df, att_col, _, _, _ = self._prep_model_activity_data(
            att_col=att_col, masks=masks, model_data=model_data, attach_time_period=attach_time_period,
            convert_categorical_to_str=convert_categorical_to_str
        )

        df = df.groupby(['o_zone', 'd_zone', *att_col], as_index=False)['weight'].sum()
        df['o_ensemble'] = data.zone_coordinates['ensemble'].reindex(df['o_zone']).fillna(0).to_numpy().astype(int)
        df['d_ensemble'] = data.zone_coordinates['ensemble'].reindex(df['d_zone']).fillna(0).to_numpy().astype(int)
        df = df.groupby(['o_ensemble', 'd_ensemble', *att_col])['weight'].sum().to_frame(name='n_trips')
        df.index.names = [f'{self._partition_col}_o', f'{self._partition_col}_d', *att_col]

        return pd.DataFrame(df)

    def summarize_model_activity_tlfd(self, *, att_col: Union[str, Iterable[str]] = None, masks: List[str] = None,
                                      model_data: LinkedDataFrame = None, attach_time_period: bool = False,
                                      impedance_type: str = 'network', convert_categorical_to_str: bool = True,
                                      **kwargs) -> pd.DataFrame:
        df, att_col, _, _, _ = self._prep_model_activity_data(
            att_col=att_col, masks=masks, model_data=model_data, attach_time_period=attach_time_period,
            impedance_type=impedance_type, convert_categorical_to_str=convert_categorical_to_str
        )

        return self._calc_tlfds(df, 'o_zone', 'd_zone', 'n_trips', col=att_col, **kwargs)

    def summarize_modes_e2e(self, *, att_col: Union[str, Iterable[str]] = None, masks: List[str] = None,
                            convert_categorical_to_str: bool = True) -> pd.DataFrame:
        data = self.microsim_data
        df = data.trip_modes.copy()
        df['o_zone'] = df.trip.o_zone
        df['d_zone'] = df.trip.d_zone

        # Sort attributes
        hhld_col, pers_col, trip_col, mode_col = [], [], [], []
        if att_col is not None:
            if isinstance(att_col, str):
                att_col = [att_col]
            elif isinstance(att_col, Iterable):
                pass
            else:
                raise TypeError(f'Invalid `att_col` type ({type(att_col)})')

            for col in att_col:
                if col in self.microsim_data.households:
                    hhld_col.append(col)
                elif col in self.microsim_data.persons:
                    pers_col.append(col)
                elif col in self.microsim_data.trips:
                    trip_col.append(col)
                elif col in df:
                    mode_col.append(col)
                else:
                    raise ValueError(f'`{col}` not found in either the household, person, trip, or trip mode tables')

        # Promote attributes from LinkedDataFrame links
        for col in hhld_col:
            df[col] = getattr(df.person.household, col)
        for col in pers_col:
            df[col] = getattr(df.person, col)
        for col in trip_col:
            df[col] = getattr(df.trip, col)
        if convert_categorical_to_str:
            for col in {'mode', *hhld_col, *pers_col, *trip_col, *mode_col}:
                if df[col].dtype.name == 'category':
                    df[col] = df[col].astype(str)

        mask = df.person.age >= 11  # match TTS, as it doesn't have data for this group
        if masks is not None:
            mask = mask & self._eval_masks(df, masks)
        df = df.loc[mask].copy()

        df = df.groupby(['o_zone', 'd_zone', 'mode', *att_col], as_index=False)['full_weight'].sum()
        df['o_ensemble'] = data.zone_coordinates['ensemble'].reindex(df['o_zone']).fillna(0).to_numpy().astype(int)
        df['d_ensemble'] = data.zone_coordinates['ensemble'].reindex(df['d_zone']).fillna(0).to_numpy().astype(int)
        df = df.groupby(['o_ensemble', 'd_ensemble', 'mode', *att_col])['full_weight'].sum().to_frame(name='n_trips')
        df.index.names = [f'{self._partition_col}_o', f'{self._partition_col}_d', 'mode', *att_col]

        return pd.DataFrame(df)

    # endregion
