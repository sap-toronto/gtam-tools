from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from bokeh.core.enums import SizingModeType
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from numpy.typing import ArrayLike, NDArray

from .common import wrap_title
from .resources import SyncAxesOptions


def _core_get_histogram_data(data: ArrayLike, *, bin_step: int = 1, use_abs: bool = False) -> Tuple[NDArray, NDArray]:
    if use_abs:
        data = np.absolute(data)
    min_val = int(np.amin(data))
    max_val = int(np.amax(data)) + bin_step + 1
    return np.histogram(data, bins=range(min_val, max_val, bin_step))


def _core_create_histogram(hist: ArrayLike, edges: ArrayLike, **kwargs) -> figure:
    fig: figure = figure(**kwargs)
    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
    return fig


def histogram_plot(df: pd.DataFrame, data_col: str, *, facet_col: str = None, facet_col_wrap: int = 2,
                   facet_sync_axes: SyncAxesOptions = None, figure_title: str = None, bin_step: int = 1,
                   use_abs: bool = False, sizing_mode: SizingModeType = None, **kwargs):
    """Creates an interactive Bokeh-based histogram plot

    Args:
        df (pd.DataFrame): A DataFrame containing data to plot histogram(s) for
        data_col (str): The name of the column in ``df`` to plot histogram(s) for
        facet_col (str, optional): Defaults to ``None``. The name of the column to use for creating a facet plot.
        facet_col_wrap (int, optional): Defaults to ``2``. The number of columns to wrap subplots in the facet plot.
        facet_sync_axes (SyncAxesOptions, optional): Defaults to ``'both'``. Option to sync/link facet axes. Accepts one
            of ``['both', 'x', 'y']``. Set to None to disable linked facet plot axes.
        figure_title (str, optional): Defaults to ``None``. The chart title to use.
        bin_step (int, optional): Defaults to ``1``. The size of each histogram bin.
        use_abs (bool, optional): Defaults to ``False``. Use the absolute values
        sizing_mode (SizingModeType, optional): Defaults to ``None``. A Bokeh SizingModeType. How will the items in the
            layout resize to fill the available space. Please refer to Bokeh documentation for acceptable values.

    Returns:
        A pandas DataFrame and a Bokeh figure
    """

    if facet_col is None:
        hist, edges = _core_get_histogram_data(df[data_col], bin_step=bin_step, use_abs=use_abs)
        results = pd.DataFrame({'bin_start': edges[:-1], 'bin_end': edges[1:], 'frequency': hist})

        fig = _core_create_histogram(
            hist, edges, **kwargs, sizing_mode='stretch_both' if figure_title is None else sizing_mode
        )
    else:
        subplots, results, linked_axes = [], [], {}
        grouper = df.groupby(level=facet_col) if facet_col in df.index.names else df.groupby(facet_col)
        for i, (label, subset) in enumerate(grouper):
            hist, edges = _core_get_histogram_data(subset[data_col], bin_step=bin_step, use_abs=use_abs)
            data = pd.DataFrame({'bin_start': edges[:-1], 'bin_end': edges[1:], 'frequency': hist})
            data.insert(0, facet_col, label)
            results.append(data)

            fig = _core_create_histogram(hist, edges, title=label, **kwargs, **linked_axes)
            if i == 0:
                if facet_sync_axes in {'x', 'both'}:
                    linked_axes['x_range'] = fig.x_range
                if facet_sync_axes in {'y', 'both'}:
                    linked_axes['y_range'] = fig.y_range
            subplots.append(fig)
        results = pd.concat(results, axis=0)
        fig = gridplot(
            subplots, ncols=facet_col_wrap, sizing_mode='stretch_both' if figure_title is None else sizing_mode,
            toolbar_location='above'
        )

    if figure_title is not None:
        fig = wrap_title(fig, figure_title, sizing_mode=sizing_mode)

    return results, fig
