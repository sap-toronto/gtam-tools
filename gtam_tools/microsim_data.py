from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from balsa.logging import get_model_logger
from cheval import LinkedDataFrame


class TimeFormat(Enum):

    MINUTE_DELTA = 'minute_delta'
    COLON_SEP = 'colon_separated'


class MicrosimData(object):

    _households: LinkedDataFrame
    _persons: LinkedDataFrame
    _trips: LinkedDataFrame
    _trip_modes: LinkedDataFrame
    _trip_stations: LinkedDataFrame
    _trip_passengers: LinkedDataFrame

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
        'household_id': np.int64, 'person_id': np.int32, 'trip_id': np.int32, 'station': np.int16, 'direction': 'category',
        'weight': np.int64
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

    @staticmethod
    def load_folder(run_folder: Union[str, Path], link_tables: bool = True, reweight_trips: bool = True,
                    time_format=TimeFormat.MINUTE_DELTA) -> 'MicrosimData':
        """Load GTAModel Microsim Result tables from a specified folder.

        Args:
            run_folder (Union[str, Path]): The path to the GTAModel Microsim Results folder
            link_tables (bool, optional): Defaults to ``True``. A flag to link result tables together. Please note that
                this option will take several minutes to complete.
            reweight_trips (bool, optional): Defaults to ``True``. A flag to calculate final weights in the trip and
                trip mode tables.
            time_format (TimeFormat, optional): Defaults to ``TimeFormat.MINUTE_DELTA``. Specify the time format in the
                Microsim results. Used for parsing times in the trip modes file.

        Returns:
            MicrosimData
        """
        run_folder = Path(run_folder)
        assert run_folder.exists(), f'Run folder `{run_folder.as_posix()}` not found'
        assert run_folder.is_dir()

        if reweight_trips:
            assert link_tables, '`link_tables` must be enabled to reweight trips'

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

        data = MicrosimData()

        MicrosimData._logger.tip(f'Loading Microsim Results from `{run_folder.as_posix()}`')

        data._load_tables(households_fp, persons_fp, trips_fp, trip_modes_fp, trip_stations_fp, fpass_fp=fpass_fp)
        data._classify_times(time_format)
        data._verify_integrity()

        if link_tables:
            data._link_tables()

        if reweight_trips:
            data.reweight_trips()

        MicrosimData._logger.tip('Microsim Results successfully loaded!')

        return data

    def _load_tables(self, households_fp: Path, persons_fp: Path, trips_fp: Path, trip_modes_fp: Path,
                     trip_stations_fp: Path, fpass_fp: Path = None):
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

    def _classify_times(self, time_format: TimeFormat):
        assert self.trip_modes_loaded

        self._logger.info('Parsing time formats')

        table = self.trip_modes

        table['o_depart_hr'] = self._convert_time_to_hours(table['o_depart'], time_format)
        self._logger.debug('Parsed `o_depart`')

        table['d_arrive_hr'] = self._convert_time_to_hours(table['d_arrive'], time_format)
        self._logger.debug('Parsed `d_arrive`')

        table['time_period'] = self._classify_time_period(table['o_depart_hr'])
        self._logger.debug('Classified time periods')

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

    def reweight_trips(self):
        assert self.trips_loaded
        trips = self.trips
        trip_modes = self.trip_modes

        self._logger.info('Reweighting trips and trip mode tables')
        trips['repetitions'] = trips.modes.sum('weight')
        trip_modes['full_weight'] = trip_modes.weight / trip_modes.trip.repetitions * trip_modes.trip.weight
