import pandas as pd
import pkg_resources


def load_407_count_stations() -> pd.DataFrame:
    stream = pkg_resources.resource_stream(__name__, 'resource_data/407_cordon_count_countposts.csv')
    dtypes = {'countpost': int, 'region_name': str, 'station_id': str, 'station_desc': str}
    return pd.read_csv(stream, index_col=['countpost'], dtype=dtypes).sort_index()
