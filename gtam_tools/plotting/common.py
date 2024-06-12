from __future__ import annotations

import warnings
from typing import Any, Dict, Hashable, Union, List, Tuple

import pandas as pd
import numpy as np

from bokeh.layouts import column
from bokeh.models import Column, Div

def check_df_indices(controls_df: pd.DataFrame, results_df: pd.DataFrame):
    if not controls_df.index.equals(results_df.index):
        warnings.warn('Indices for \'controls_df\' and \'results_df\' are not identical, function may not produce desired results')
    if not controls_df.columns.equals(results_df.columns):
        warnings.warn('Columns for \'controls_df\' and \'results_df\' are not identical, function may not produce desired results')


def check_ref_label(ref_label: Union[str, List[str]], controls_df: pd.DataFrame, results_df: pd.DataFrame) -> List[str]:
    if ref_label is None:
       if not np.all(controls_df.index.names == results_df.index.names):
           raise RuntimeError('Unable to resolve different index names, pleace specify values for \'ref_label\' instead')
       if None in controls_df.index.names:
           raise RuntimeError('Some index levels in \'controls_df\' do not have names')
       if None in results_df.index.names:
           raise RuntimeError('Some index levels in \'results_df\' do not have names')
       ref_label = list(controls_df.index.names)
    elif isinstance(ref_label, Hashable):
        ref_label = [ref_label]
    elif isinstance(ref_label, List):
        pass
    else:
        raise RuntimeError('Invalid data type provided for \'ref_label\'')
    
    return ref_label


def prep_figure_params(x_label: str, y_label: str, tooltips: List[Tuple[Hashable, Hashable]], plot_width: int = None,
                        plot_height: int = None, scatterplot: bool = False) -> Dict[str, Any]:
    if scatterplot == False:
        figure_params = {
            'x_axis_label': x_label, 'y_axis_label': y_label, 'tooltips': tooltips, 'toolbar_location': 'above',
            'tools': 'pan,zoom_in,zoom_out,box_zoom,wheel_zoom,hover,save,reset', 'output_backend': 'webgl'
        }
    else:
        figure_params = {
            'x_axis_label': x_label, 'y_axis_label': y_label, 'tooltips': tooltips, 'toolbar_location': 'above',
            'tools': 'pan,zoom_in,zoom_out,box_zoom,wheel_zoom,save,reset', 'output_backend': 'webgl'
        }
    if plot_width is not None:
        figure_params['width'] = plot_width
    if plot_height is not None:
        figure_params['height'] = plot_height

    return figure_params


def wrap_figure_title(fig, figure_title : str):
    title = Div(text=f'<h2>{figure_title}</h2>')
    return column(children=[title, fig], sizing_mode='stretch_both')
