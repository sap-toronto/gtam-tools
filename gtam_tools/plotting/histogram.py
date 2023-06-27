from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from bokeh.core.enums import SizingModeType
from bokeh.layouts import gridplot
from bokeh.models import LayoutDOM as BokehFigure
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
                   title: str = None, bin_step: int = 1, use_abs: bool = False, sizing_mode: SizingModeType = None,
                   sync_axes: SyncAxesOptions = None, **kwargs) -> Tuple[pd.DataFrame, BokehFigure]:
    if facet_col is None:
        hist, edges = _core_get_histogram_data(df[data_col], bin_step=bin_step, use_abs=use_abs)
        results = pd.DataFrame({'bin_start': edges[:-1], 'bin_end': edges[1:], 'frequency': hist})

        fig = _core_create_histogram(hist, edges, **kwargs)
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
                if sync_axes in {'x', 'both'}:
                    linked_axes['x_range'] = fig.x_range
                if sync_axes in {'y', 'both'}:
                    linked_axes['y_range'] = fig.y_range
            subplots.append(fig)
        results = pd.concat(results, axis=0)
        fig = gridplot(subplots, ncols=facet_col_wrap, sizing_mode='stretch_both', toolbar_location='above')

    if title is not None:
        fig = wrap_title(fig, title, sizing_mode=sizing_mode)

    return results, fig
