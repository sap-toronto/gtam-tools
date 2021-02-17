import geopandas as gpd
from itertools import product
import numpy as np
import pandas as pd
from pathlib import Path
import pkg_resources
from typing import Union

from balsa.routines import distance_matrix, read_mdf
from balsa.logging import get_model_logger
from cheval import LinkedDataFrame

from .enums import TimeFormat, ZoneNums


def _load_model_activity_pairs() -> pd.Series:
    stream = pkg_resources.resource_stream(__name__, 'resources/activity_pairs_model.csv')
    return pd.read_csv(stream, index_col=[0, 1], squeeze=True)


class MicrosimData(object):

    _households: LinkedDataFrame
    _persons: LinkedDataFrame
    _trips: LinkedDataFrame
    _trip_modes: LinkedDataFrame
    _trip_stations: LinkedDataFrame
    _trip_passengers: LinkedDataFrame
    _zone_coordinates: gpd.GeoDataFrame
    _impedances: pd.DataFrame

    _logger = get_model_logger('gtam_tools.MicrosimData')

    _household_spec = {
        'household_id': np.int64, 'home_zone': np.int16, 'weight': np.int64, 'persons': np.int16,
        'dwelling_type': np.int16, 'vehicles': np.int16, 'income_class': np.int16
    }

    _persons_spec = {
        'household_id': np.int64, 'person_id': np.int32, 'age': np.int16, 'sex': 'category', 'license': bool,
        'transit_pass': bool, 'employment_status': 'category', 'occupation': 'category', 'free_parking': bool,
        'student_status': 'category', 'work_zone': np.int16, 'school_zone': np.int16, 'weight': np.int64
    }

    _trips_spec = {
        'household_id': np.int64, 'person_id': np.int32, 'trip_id': np.int32, 'o_act': 'category', 'o_zone': np.int16,
        'd_act': 'category', 'd_zone': np.int16, 'weight': np.int64
    }

    _trip_modes_spec = {
        'household_id': np.int64, 'person_id': np.int32, 'trip_id': np.int32, 'mode': 'category', 'o_depart': str,
        'd_arrive': str, 'weight': np.int64
    }

    _trip_stations_spec = {
        'household_id': np.int64, 'person_id': np.int32, 'trip_id': np.int32, 'station': np.int16,
        'direction': 'category', 'weight': np.int64
    }

    _passenger_spec = {
        'household_id': np.int64, 'passenger_id': np.int32, 'passenger_trip_id': np.int32, 'driver_id': np.int32,
        'driver_trip_id': np.int32, 'weight': np.int64
    }

    @property
    def households_loaded(self):
        return self._households is not None

    @property
    def persons_loaded(self):
        return self._persons is not None

    @property
    def trips_loaded(self):
        return self._trips is not None

    @property
    def trip_modes_loaded(self):
        return self._trip_modes is not None

    @property
    def trip_stations_loaded(self):
        return self._trip_stations is not None

    @property
    def trips_passengers_loaded(self):
        return self._trip_passengers is not None

    @property
    def zone_coordinates_loaded(self):
        return self._zone_coordinates is not None

    @property
    def households(self) -> LinkedDataFrame:
        return self._households

    @property
    def persons(self) -> LinkedDataFrame:
        return self._persons

    @property
    def trips(self) -> LinkedDataFrame:
        return self._trips

    @property
    def trip_modes(self) -> LinkedDataFrame:
        return self._trip_modes

    @property
    def trip_stations(self) -> LinkedDataFrame:
        return self._trip_stations

    @property
    def trip_passengers(self) -> LinkedDataFrame:
        return self._trip_passengers

    @property
    def zone_coordinates(self) -> gpd.GeoDataFrame:
        return self._zone_coordinates

    @property
    def impedances(self) -> pd.DataFrame:
        return self._impedances

    @staticmethod
    def load_folder(run_folder: Union[str, Path], link_tables: bool = True, derive_additional_variables: bool = True,
                    time_format=TimeFormat.MINUTE_DELTA, zones_file: Union[str, Path] = None, taz_col: str = None,
                    zones_crs: str = None, to_crs: str = 'EPSG:26917', coord_unit: float = 0.001) -> 'MicrosimData':
        """Load GTAModel Microsim Result tables from a specified folder.

        Args:
            run_folder (Union[str, Path]): The path to the GTAModel Microsim Results folder
            link_tables (bool, optional): Defaults to ``True``. A flag to link result tables together. Please note that
                this option will take several minutes to complete.
            derive_additional_variables (bool, optional): Defaults to ``True``. A flag to derive additional variables
                based on the Microsim data. Requires ``link_tables=True``.
            time_format (TimeFormat, optional): Defaults to ``TimeFormat.MINUTE_DELTA``. Specify the time format in the
                Microsim results. Used for parsing times in the trip modes file.
            zones_file (Union[str, Path], optional): Defaults to ``None``. The path to a file containing zone
                coordinates. Accepts CSVs with x, y coordinate columns, or a zones shapefile. If providing a shapefile,
                it will calculate the zone coordinate based on the centroid of each zone polygon.
            taz_col (str, optional): Defaults to ``None``. Name of the TAZ column in ``zones_file``.
            zones_crs (Union[str, CRS], optional): Defaults to ``None``. The coordinate reference system for
                ``zones_file``. Only applicable if ``zones_file`` is a CSV file. Value can be anything accepted by
                ``pyproj.CRS.from_user_input()``.
            to_crs (str, optional): Defaults to ``'EPSG:26917'``. A coordinate reference system to reproject zone
                coordinates to. Value can be anything accepted by ``pyproj.CRS.from_user_input()``.
            coord_unit (float, optional): Defaults to ``0.001``. A value to adjust distance values with.

        Returns:
            MicrosimData
        """
        run_folder = Path(run_folder)
        assert run_folder.exists(), f'Run folder `{run_folder.as_posix()}` not found'
        assert run_folder.is_dir()

        if zones_file is not None:
            assert taz_col is not None, 'Please specify `taz_col`'

        if derive_additional_variables:
            assert link_tables, '`link_tables` must be enabled to derive additional variables'

        def _prep_file(name: str) -> Path:
            uncompressed = run_folder / (name + '.csv')
            compressed = run_folder / (name + '.csv.gz')

            if uncompressed.exists():
                return uncompressed
            if compressed.exists():
                return compressed

            raise FileExistsError(f'Could not find the `{name}` output file.')

        households_fp = _prep_file('households')
        persons_fp = _prep_file('persons')
        trips_fp = _prep_file('trips')
        trip_modes_fp = _prep_file('trip_modes')
        trip_stations_fp = _prep_file('trip_stations')

        try:
            fpass_fp = _prep_file('facilitate_passenger')
        except FileExistsError:
            fpass_fp = None

        if zones_file is not None:
            zones_file = Path(zones_file)
            if not zones_file.exists():
                raise FileExistsError(f'Zone attribute file could not be found at {zones_file.as_posix()}')

        MicrosimData._logger.tip(f'Loading Microsim Results from `{run_folder.as_posix()}`')

        data = MicrosimData()
        data._load_tables(households_fp, persons_fp, trips_fp, trip_modes_fp, trip_stations_fp, fpass_fp=fpass_fp,
                          zones_fp=zones_file, taz_col=taz_col, zones_crs=zones_crs, to_crs=to_crs)

        if data.zone_coordinates_loaded:
            data._calc_base_impedences(coord_unit)

        data._verify_integrity()

        if link_tables:
            data._link_tables()

        if derive_additional_variables:
            data._classify_times(time_format)
            data._derive_household_variables()
            data._derive_person_variables()
            data._derive_trip_variables()
            data._reweight_trips()

        MicrosimData._logger.tip('Microsim Results successfully loaded!')

        return data

    def add_impedance_skim(self, name: str, skim_fp: Union[str, Path], scale_unit: float = 1.0,
                           ignore_missing_ods: bool = False):
        """Add a skim from a matrix binary file as impedance values to the impedances table.

        Args:
            name (str): The reference name for the impedance skim.
            skim_fp (Union[str, Path]): The file path to the skim matrix binary file.
            scale_unit (float, optional): Defaults to ``1.0``. A scalar value to adjust skim values.
            ignore_missing_ods (bool, optional): Defaults to ``False``. A flag to ignore missing ODs. If ``True``, skim
                values for missing ODs will be set to zero.
        """
        assert self.zone_coordinates_loaded, 'Cannot add skim unless zone coordinates are loaded'

        skim_fp = Path(skim_fp)
        assert skim_fp.exists(), f'Skim file not found at `{skim_fp.as_posix()}`'

        skim_data = read_mdf(skim_fp, raw=False, tall=True)
        mask1 = skim_data.index.get_level_values(0) <= ZoneNums.MAX_INTERNAL
        mask2 = skim_data.index.get_level_values(1) <= ZoneNums.MAX_INTERNAL
        skim_data = skim_data[mask1 & mask2].reindex(self.impedances.index)

        skim_data = skim_data * scale_unit

        if ignore_missing_ods:
            skim_data.fillna(0, inplace=True)
        else:
            assert np.all(~skim_data.isna()), 'Skim is not compatible with the dataset zone system'

        self.impedances[name] = skim_data
        self._logger.report(f'Added `{name}` to impedances')

    def add_zone_ensembles(self, name: str, definition_fp: Union[str, Path], taz_col: str,
                           ensemble_col: str = 'ensemble', missing_val: int = 9999,
                           ensemble_names_fp: Union[str, Path] = None, ensemble_names_col: str = 'name'):
        """Add zone ensemble definitions to the zone coordinates table.

        Args:
            name (str): The reference name for the ensemble definitions.
            definition_fp (Union[str, Path]): The file path to the zone ensemble correspondence file. Can be a CSV
                file or shapefile.
            taz_col (str): Name of the TAZ column in ``definition_fp``.
            ensemble_col (str, optional): Defaults to ``'ensemble'``. Name of the ensembles column in ``definition_fp``.
            missing_val (int, optional): Defaults to ``9999``. A value to use for all TAZs without an
                assigned ensemble.
            ensemble_names_fp (Union[str, Path], optional): Defaults to ``None``. The file path to a CSV file containing
                zone ensemble names. The ensemble id column in this file must be the same as ``ensemble_col``.
            ensemble_names_col (str, optional): Defaults to ``'name'``. The name of the column containing the ensemble
                names. Will only be used if ``ensemble_names_fp`` is specified.
        """
        assert self.zone_coordinates_loaded, 'Cannot add zone ensembles unless zone coordinates are loaded'

        taz_col = taz_col.lower()
        ensemble_col = ensemble_col.lower()

        definition_fp = Path(definition_fp)
        assert definition_fp.exists(), f'Correspondence file not found at `{definition_fp.as_posix()}`'

        if definition_fp.suffix == '.csv':
            correspondence = pd.read_csv(definition_fp)
        elif definition_fp.suffix == '.shp':
            correspondence = gpd.read_file(definition_fp)
        else:
            raise RuntimeError(f'An unsupported zones file type was provided ({definition_fp.suffix})')

        correspondence.columns = correspondence.columns.str.lower()
        correspondence.set_index(taz_col, inplace=True)
        correspondence = correspondence[ensemble_col].reindex(self.zone_coordinates.index, fill_value=missing_val)
        correspondence = correspondence.to_frame(name=ensemble_col)

        if ensemble_names_fp is not None:
            ensemble_names_fp = Path(ensemble_names_fp)
            assert ensemble_names_fp.exists(), f'Ensemble names file not found at `{ensemble_names_fp.as_posix()}`'

            ensemble_names = pd.read_csv(ensemble_names_fp)

            correspondence = correspondence.merge(ensemble_names, how='left', on=ensemble_col)

        self.zone_coordinates[name] = correspondence[ensemble_col].values
        if ensemble_names_fp is not None:
            self.zone_coordinates[f'{name}_label'] = correspondence[ensemble_names_col]
        self._logger.report('Added ensembles to zone coordinates')

    @staticmethod
    def _load_zones_file(fp: Path, taz_col: str, zones_crs: str, to_crs: str) -> gpd.GeoDataFrame:
        if fp.suffix == '.csv':
            zones_df = pd.read_csv(fp)
            zones_df.columns = zones_df.columns.str.lower()
            zones_df = gpd.GeoDataFrame(zones_df, geometry=gpd.points_from_xy(zones_df['x'], zones_df['y']),
                                        crs=zones_crs)
        elif fp.suffix == '.shp':
            zones_df = gpd.read_file(fp.as_posix())
            zones_df.columns = zones_df.columns.str.lower()
            zones_df['geometry'] = zones_df.centroid
        else:
            raise RuntimeError(f'An unsupported zones file type was provided ({fp.suffix})')

        taz_col = taz_col.lower()
        zones_df[taz_col] = zones_df[taz_col].astype(int)

        zones_df = zones_df[zones_df[taz_col] <= ZoneNums.MAX_INTERNAL].copy()  # Keep internal zones only

        zones_df.set_index(taz_col, inplace=True)
        zones_df = zones_df.to_crs({'init': to_crs})  # reproject
        zones_df['x'] = zones_df.geometry.x
        zones_df['y'] = zones_df.geometry.y
        zones_df.sort_index(inplace=True)

        return zones_df[['geometry', 'x', 'y']].copy()

    def _load_tables(self, households_fp: Path, persons_fp: Path, trips_fp: Path, trip_modes_fp: Path,
                     trip_stations_fp: Path, fpass_fp: Path = None, zones_fp: Path = None, taz_col: str = None,
                     zones_crs: str = None, to_crs: str = 'EPSG:26917'):
        self._logger.info(f'Loading result tables')

        table = LinkedDataFrame.read_csv(households_fp, dtype=self._household_spec)
        self._logger.report(f'Loaded {len(table)} household entries')
        self._households = table

        table = LinkedDataFrame.read_csv(persons_fp, dtype=self._persons_spec)
        self._logger.report(f'Loaded {len(table)} person entries')
        self._persons = table

        table = LinkedDataFrame.read_csv(trips_fp, dtype=self._trips_spec)
        self._logger.report(f'Loaded {len(table)} trip entries')
        self._trips = table

        table = LinkedDataFrame.read_csv(trip_modes_fp, dtype=self._trip_modes_spec)
        self._logger.report(f'Loaded {len(table)} trip mode entries')
        self._trip_modes = table

        table = LinkedDataFrame.read_csv(trip_stations_fp, dtype=self._trip_stations_spec)
        self._logger.report(f'Loaded {len(table)} trip station entries')
        self._trip_stations = table

        if fpass_fp is not None:
            table = LinkedDataFrame.read_csv(fpass_fp, dtype=self._passenger_spec)
            self._logger.report(f'Loaded {len(table)} trip passenger entries')
            self._trip_passengers = table

        if zones_fp is not None:
            table = self._load_zones_file(zones_fp, taz_col, zones_crs, to_crs)
            self._logger.report(f'Loaded coordinates for {len(table)} internal zones')
            self._zone_coordinates = table

    def _classify_times(self, time_format: TimeFormat):
        assert self.trip_modes_loaded

        self._logger.info('Parsing time formats')

        table = self.trip_modes

        self._logger.debug('Parsing `o_depart`')
        table['o_depart_hr'] = self._convert_time_to_hours(table['o_depart'], time_format)

        self._logger.debug('Parsing `d_arrive`')
        table['d_arrive_hr'] = self._convert_time_to_hours(table['d_arrive'], time_format)

        self._logger.debug('Classifying `time_period`')
        table['time_period'] = self._classify_time_period(table['o_depart_hr'])

    @staticmethod
    def _convert_time_to_hours(column: pd.Series, time_format: TimeFormat) -> pd.Series:
        if time_format == time_format.MINUTE_DELTA:
            return MicrosimData._floordiv_minutes(column)
        elif time_format == time_format.COLON_SEP:
            return MicrosimData._convert_text_to_datetime(column)
        else:
            raise NotImplementedError(time_format)

    @staticmethod
    def _convert_text_to_datetime(s: pd.Series) -> pd.Series:
        colon_count = s.str.count(':')
        filtr = colon_count == 1

        new_time: pd.Series = s.copy()
        new_time.loc[filtr] += ':00'

        filtr = new_time.str.contains('-')
        if filtr.sum() > 0:
            new_time.loc[filtr] = "0:00:00"
            print(f"Found {filtr.sum()} cells with negative time. These have been corrected to 0:00:00")

        time_table = new_time.str.split(':', expand=True).astype(np.int8)
        hours = time_table.iloc[:, 0]

        return hours

    @staticmethod
    def _floordiv_minutes(column: pd.Series) -> pd.Series:
        converted = column.astype(np.float64)
        return (converted // 60).astype(np.int32)

    @staticmethod
    def _classify_time_period(start_hour: pd.Series) -> pd.Series:
        new_col = pd.Series('DN', index=start_hour.index)

        new_col.loc[start_hour.between(6, 8)] = 'AM'
        new_col.loc[start_hour.between(9, 14)] = 'MD'
        new_col.loc[start_hour.between(15, 18)] = 'PM'
        new_col.loc[start_hour.between(19, 23)] = 'EV'

        return new_col.astype('category')

    def _calc_base_impedences(self, coord_unit: float):
        self._logger.info('Calculating standard impedances from zone coordinates')

        methods = ['manhattan', 'euclidean']
        impedances = {
            method: distance_matrix(self._zone_coordinates['x'], self._zone_coordinates['y'], tall=True, method=method,
                                    coord_unit=coord_unit) for method in methods
        }

        self._impedances = pd.DataFrame(impedances)
        self._impedances.index.names = ['o', 'd']

    def _verify_integrity(self):
        if self.households_loaded and self.persons_loaded:
            hh_sizes = self.persons['household_id'].value_counts(dropna=False)
            isin = hh_sizes.index.isin(self.households['household_id'])
            n_homeless = hh_sizes.loc[~isin].sum()
            if n_homeless > 0:
                raise RuntimeError('Found `%s` persons with invalid or missing household IDs' % n_homeless)

    def _link_tables(self):
        self._logger.info('Precomputing table linkages')

        self.persons.link_to(self.households, 'household', on='household_id')
        self.households.link_to(self.persons, 'persons', on='household_id')
        self.persons['home_zone'] = self.persons.household.home_zone
        self._logger.debug('Linked households to persons')

        self.trips.link_to(self.persons, 'person', on=['household_id', 'person_id'])
        self._logger.debug('Linked persons to trips')

        self.trip_modes.link_to(self.trips, 'trip', on=['household_id', 'person_id', 'trip_id'])
        self.trips.link_to(self.trip_modes, 'modes', on=['household_id', 'person_id', 'trip_id'])
        self.trip_modes.link_to(self.persons, 'person', on=['household_id', 'person_id'])
        self.trip_modes.link_to(self.households, 'household', on='household_id')
        self._logger.debug('Linked trip modes to households, persons, and trips')

        self.trip_stations.link_to(self.trips, 'trip', on=['household_id', 'person_id', 'trip_id'])
        self._logger.debug('Linked trip stations to trips')

        if self.zone_coordinates_loaded:
            self.trips.link_to(self.impedances, 'imped', on_self=['o_zone', 'd_zone'])
            self._logger.debug('Linked impedances to trips')

    def _derive_household_variables(self):
        households = self.households

        self._logger.info('Deriving additional household variables')

        self._logger.debug('Summarizing `drivers`')
        households['drivers'] = households.persons.sum('license * 1')

        self._logger.debug('Classifying `auto_suff`')
        households['auto_suff'] = 'suff'
        households.loc[households.eval('drivers > vehicles'), 'auto_suff'] = 'insuff'
        households.loc[households['vehicles'] == 0, 'auto_suff'] = 'nocar'
        households['auto_suff'] = households['auto_suff'].astype('category')

    def _derive_person_variables(self):
        persons = self.persons

        self._logger.info('Deriving additional persons variables')

        self._logger.debug('Classifying `person_type`')
        persons['person_type'] = 'O'
        persons.loc[persons['age'] >= 65, 'person_type'] = 'R'  # Retired
        persons.loc[persons['employment_status'] == 'F', 'person_type'] = 'F'  # Full-time Worker
        persons.loc[persons['employment_status'] == 'P', 'person_type'] = 'P'  # Part-time Worker
        persons.loc[persons['age'] < 18, 'person_type'] = 'S'  # Student
        persons.loc[(persons['age'] >= 18) & (persons['student_status'].isin({'F', 'P'})), 'person_type'] = 'U'  # University/College
        persons['person_type'] = persons['person_type'].astype('category')

        self._logger.debug('Classifying `student_type`')
        persons['student_type'] = 'O'
        persons.loc[persons['age'] < 13, 'student_type'] = 'P'  # Primary
        persons.loc[persons['age'].between(13, 17), 'student_type'] = 'S'  # Secondary
        persons.loc[persons['person_type'] == 'U', 'student_type'] = 'U'  # University/College
        persons['student_type'] = persons['student_type'].astype('category')

        self._logger.debug('Classifying `occ_emp`')  # combine occupation and employment status into one identifier
        persons['occ_emp'] = 'O'
        occs = 'G P M S'.split(' ')
        emp_stat_groups = {'F': {'F', 'H'}, 'P': {'P', 'J'}}  # Only look at full or part time workers
        for occ, status in product(occs, emp_stat_groups.keys()):
            mask1 = persons['occupation'] == occ
            mask2 = persons['employment_status'].isin(emp_stat_groups[status])
            persons.loc[mask1 & mask2, 'occ_emp'] = f'{occ}{status}'
        persons['occ_emp'] = persons['occ_emp'].astype('category')

        self._logger.debug('Classifying `work_from_home`')
        persons['work_from_home'] = persons['employment_status'].isin({'H', 'J'})

    def _derive_trip_variables(self):
        trips = self.trips

        self._logger.info('Deriving addition trip variables')

        self._logger.debug('Classifying `purpose`')
        lookup_table = _load_model_activity_pairs()  # Essentially, `purpose` is being classified on `d_act`...
        indexer = pd.MultiIndex.from_arrays([trips['o_act'], trips['d_act']])
        trips['purpose'] = lookup_table.reindex(indexer, fill_value='NHB').values
        trips['purpose'] = trips['purpose'].astype('category')

        self._logger.debug('Classifying `direction`')
        orig_is_home = trips['o_act'] == 'Home'
        dest_is_home = trips['d_act'] == 'Home'
        trips['direction'] = 'NHB'
        trips.loc[orig_is_home & dest_is_home, 'direction'] = 'H2H'
        trips.loc[orig_is_home & ~dest_is_home, 'direction'] = 'Outbound'
        trips.loc[~orig_is_home & dest_is_home, 'direction'] = 'Inbound'
        trips['direction'] = trips['direction'].astype('category')

    def _reweight_trips(self):
        trips = self.trips
        trip_modes = self.trip_modes

        self._logger.info('Reweighting trips and trip mode tables')

        self._logger.debug('Summarizing `repetitions` in the trips table')
        trips['repetitions'] = trips.modes.sum('weight')

        self._logger.debug('Calculating `full_weight` in the trip modes table')
        trip_modes['full_weight'] = trip_modes.weight / trip_modes.trip.repetitions * trip_modes.trip.weight
